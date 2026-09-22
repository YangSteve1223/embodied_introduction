#!/usr/bin/env python3
"""True 1-2B student: truncate OpenVLA's 32-layer Llama backbone to 8 layers."""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import CHECKPOINT, build_dataset, collator, forward_actions, load_policy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--teacher-cache',required=True); ap.add_argument('--max-steps',type=int,default=500); ap.add_argument('--layers',type=int,default=8); ap.add_argument('--grad-accum',type=int,default=4); ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--teacher-weight',type=float,default=.5); ap.add_argument('--seed',type=int,default=20260922); args=ap.parse_args()
    if not dist.is_initialized(): dist.init_process_group('nccl')
    rank=dist.get_rank(); world=dist.get_world_size(); torch.cuda.set_device(rank); device=torch.device('cuda',rank); random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    vla,processor,head,projector=load_policy(Path(CHECKPOINT),device=device); layers=vla.language_model.model.layers; full_layers=len(layers); keep_layers=list(layers[:args.layers]); del layers; vla.language_model.model.layers=torch.nn.ModuleList(keep_layers); del keep_layers; torch.cuda.empty_cache(); vla.config.text_config.num_hidden_layers=args.layers; vla.language_model.config.num_hidden_layers=args.layers; vla.train(); head.train(); projector.train();
    for p in vla.parameters(): p.requires_grad_(True)
    for p in head.parameters(): p.requires_grad_(True)
    for p in projector.parameters(): p.requires_grad_(True)
    trainable=list(vla.parameters())+list(head.parameters())+list(projector.parameters()); student_params=sum(p.numel() for p in trainable); model=DDP(vla,device_ids=[rank],find_unused_parameters=False) if world>1 else vla; opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01); ds=build_dataset(processor,('libero_spatial_no_noops','libero_object_no_noops'),'train',use_proprio=True); coll=collator(processor); cache=np.load(args.teacher_cache); run=Path(args.run_dir); run.mkdir(parents=True,exist_ok=True)
    if rank==0: (run/'config.json').write_text(json.dumps({**vars(args),'world_size':world,'full_layers':full_layers,'student_layers':args.layers,'student_parameters':student_params},indent=2))
    logs=[]; start=time.time()
    for step in range(1,args.max_steps+1):
        opt.zero_grad(set_to_none=True); gt_total=0.; keep_total=0.
        for _ in range(args.grad_accum):
            idx=random.randrange(len(ds)); item=ds[idx]; batch=coll([item]); pred=forward_actions(model.module if world>1 else model,head,projector,batch,train_mode=True); gt=torch.from_numpy(np.asarray(item['actions'])).to(device=device,dtype=pred.dtype).unsqueeze(0); gt_loss=torch.nn.functional.l1_loss(pred,gt); keep=torch.zeros((),device=device,dtype=pred.dtype)
            if str(item['dataset_name'])=='libero_spatial_no_noops':
                teacher=torch.from_numpy(cache[int(item['sample_index'])]).to(device=device,dtype=pred.dtype).unsqueeze(0); keep=torch.nn.functional.smooth_l1_loss(pred.float(),teacher.float()).to(pred.dtype)
            loss=((1-args.teacher_weight)*gt_loss+args.teacher_weight*keep)/args.grad_accum; loss.backward(); gt_total+=float(gt_loss.detach()); keep_total+=float(keep.detach())
        torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if rank==0 and (step%25==0 or step==1): logs.append({'step':step,'gt_l1':gt_total/args.grad_accum,'teacher_smooth_l1':keep_total/args.grad_accum}); print(logs[-1],flush=True)
        if rank==0 and step in (250,args.max_steps):
            target_model=model.module if world>1 else model; torch.save({'vla':{k:v.detach().cpu() for k,v in target_model.state_dict().items()},'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'projector':{k:v.detach().cpu() for k,v in projector.state_dict().items()},'layers':args.layers,'config':vars(args)},run/f'student_step_{step}.pt')
    if rank==0:
        (run/'train_metrics.json').write_text(json.dumps({'logs':logs,'elapsed_sec':time.time()-start,'student_parameters':student_params},indent=2)); (run/'SUCCESS').write_text(str(args.max_steps))
    if world>1: dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()

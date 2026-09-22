#!/usr/bin/env python3
"""Independent small visuomotor student distilled from OpenVLA teacher actions."""
import argparse, json, os, random
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class StudentDataset(Dataset):
    def __init__(self, root, suites, split, teacher_dir, vocab=None):
        self.items=[]; self.vocab=vocab or {}
        for suite in suites:
            b=Path(root)/suite; p=np.load(b/f'{split}_primary.npy'); w=np.load(b/f'{split}_wrist.npy'); q=np.load(b/f'{split}_proprio.npy'); a=np.load(b/f'{split}_actions.npy'); lang=np.load(b/f'{split}_language.npy',allow_pickle=True)
            specific=Path(teacher_dir)/f'{suite}_{split}.npy' if teacher_dir else Path('/does/not/exist')
            legacy=Path(teacher_dir)/'teacher_actions.npy' if teacher_dir else Path('/does/not/exist')
            has_teacher=specific.exists() or (suite=='libero_spatial_no_noops' and split=='train' and legacy.exists())
            t=np.load(specific if specific.exists() else legacy) if has_teacher else np.zeros_like(a)
            for i in range(len(a)):
                words=str(lang[i]).lower().replace('_',' ').split(); ids=[self.vocab.get(x,1) for x in words]
                bow=np.zeros(len(self.vocab)+2,np.float32)
                for j in ids: bow[j]+=1
                if ids: bow/=len(ids)
                self.items.append((p[i],w[i],q[i],a[i],t[i],bow,suite,float(has_teacher)))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,w,q,a,t,b,s,h=self.items[i]
        return torch.from_numpy(np.concatenate([p,w],axis=2)).permute(2,0,1).float()/255., torch.from_numpy(q).float(), torch.from_numpy(a).float(), torch.from_numpy(t).float(), torch.from_numpy(b).float(), s, torch.tensor(h)

class TinyPolicy(nn.Module):
    def __init__(self, vocab_dim):
        super().__init__(); self.encoder=nn.Sequential(nn.Conv2d(6,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,5,2,2),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d(1)); self.lang=nn.Sequential(nn.Linear(vocab_dim,64),nn.GELU()); self.proprio=nn.Sequential(nn.Linear(8,64),nn.GELU()); self.head=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,56))
    def forward(self,img,q,bow): return self.head(torch.cat([self.encoder(img).flatten(1),self.lang(bow),self.proprio(q)],1)).view(-1,8,7)

def vocab_from(root,suites):
    words=set()
    for suite in suites:
        for split in ['train','heldout']:
            x=np.load(Path(root)/suite/f'{split}_language.npy',allow_pickle=True)
            for s in x: words.update(str(s).lower().replace('_',' ').split())
    return {w:i+2 for i,w in enumerate(sorted(words))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--teacher-dir',required=True); ap.add_argument('--run-dir',required=True); ap.add_argument('--suites',default='libero_spatial_no_noops,libero_object_no_noops'); ap.add_argument('--max-steps',type=int,default=1000); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--teacher-weight',type=float,default=.5); ap.add_argument('--seed',type=int,default=20260922); ap.add_argument('--resume',default=''); args=ap.parse_args()
    dist.init_process_group('nccl'); rank=dist.get_rank(); world=dist.get_world_size(); torch.cuda.set_device(rank); device=torch.device('cuda',rank); random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    suites=tuple(args.suites.split(',')); vocab=vocab_from(args.data_root,suites); ds=StudentDataset(args.data_root,suites,'train',args.teacher_dir,vocab); sampler=DistributedSampler(ds,shuffle=True,seed=args.seed); loader=DataLoader(ds,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True); raw_model=TinyPolicy(len(vocab)+2).to(device); start_step=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False); raw_model.load_state_dict(ck['model'],strict=True); start_step=int(Path(args.resume).stem.rsplit('_',1)[-1]) if 'step_' in Path(args.resume).stem else 0
    model=DDP(raw_model,device_ids=[rank]); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); run=Path(args.run_dir); run.mkdir(parents=True,exist_ok=True)
    if rank==0: (run/'config.json').write_text(json.dumps({**vars(args),'world_size':world,'parameters':sum(p.numel() for p in model.module.parameters()),'vocab_size':len(vocab)+2},indent=2))
    it=iter(loader); logs=[]
    for step in range(start_step+1,args.max_steps+1):
        try: img,q,gt,teach,bow,_,has=next(it)
        except StopIteration: sampler.set_epoch(step); it=iter(loader); img,q,gt,teach,bow,_,has=next(it)
        img,q,gt,teach,bow,has=[x.to(device,non_blocking=True) for x in (img,q,gt,teach,bow,has)]; pred=model(img,q,bow); gt_loss=torch.nn.functional.l1_loss(pred,gt); t_raw=torch.nn.functional.smooth_l1_loss(pred,teach,reduction='none').mean((1,2)); t_loss=(t_raw*has).sum()/has.sum().clamp_min(1); loss=(1-args.teacher_weight)*gt_loss+args.teacher_weight*t_loss; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if rank==0 and (step%25==0 or step==1): logs.append({'step':step,'loss':float(loss.detach()),'gt_l1':float(gt_loss.detach()),'teacher_smooth_l1':float(t_loss.detach())}); print(logs[-1],flush=True)
        if rank==0 and step % 250 == 0: torch.save({'model':model.module.state_dict(),'vocab':vocab,'config':vars(args)},run/f'student_step_{step}.pt')
    if rank==0: torch.save({'model':model.module.state_dict(),'vocab':vocab,'config':vars(args)},run/'student_final.pt'); (run/'train_metrics.json').write_text(json.dumps({'logs':logs,'parameters':sum(p.numel() for p in model.module.parameters()),'world_size':world},indent=2)); (run/'SUCCESS').write_text(str(args.max_steps))
    dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()

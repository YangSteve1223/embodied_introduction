#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from common import CHECKPOINT,build_dataset,collator,forward_actions,load_policy
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--split',default='heldout'); ap.add_argument('--layers',type=int,default=8); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--max-samples',type=int,default=0); args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); vla,processor,head,projector=load_policy(Path(CHECKPOINT),device=device); layers=vla.language_model.model.layers; keep_layers=list(layers[:args.layers]); del layers; vla.language_model.model.layers=torch.nn.ModuleList(keep_layers); del keep_layers; torch.cuda.empty_cache(); vla.config.text_config.num_hidden_layers=args.layers; vla.language_model.config.num_hidden_layers=args.layers; ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); vla.load_state_dict(ck['vla'],strict=True); head.load_state_dict(ck['head'],strict=True); projector.load_state_dict(ck['projector'],strict=True); vla.eval();head.eval();projector.eval(); ds=build_dataset(processor,('libero_spatial_no_noops','libero_object_no_noops'),args.split,use_proprio=True); coll=collator(processor); rows=[]; start=time.time(); limit=args.max_samples or len(ds)
 for base in range(0,limit,args.batch_size):
  items=[ds[i] for i in range(base,min(base+args.batch_size,limit))]; pred=forward_actions(vla,head,projector,coll(items),train_mode=False).float().cpu().numpy()
  for j,item in enumerate(items):
   gt=np.asarray(item['actions'],dtype=np.float32); e=np.abs(pred[j]-gt); rows.append({'suite':str(item['dataset_name']),'task':str(item['sample_task']),'full_chunk_l1':float(e.mean()),'first_action_l1':float(e[0].mean()),'per_dimension_mae':e.mean(0).tolist()})
 suites=sorted(set(r['suite'] for r in rows)); per={s:float(np.mean([r['full_chunk_l1'] for r in rows if r['suite']==s])) for s in suites}; out={'config':vars(args),'elapsed_sec':time.time()-start,'sample_count':len(rows),'per_suite':per,'macro_suite_normalized_action_l1':float(np.mean(list(per.values()))),'first_action_step_l1':float(np.mean([r['first_action_l1'] for r in rows])),'per_dimension_mae':np.mean([r['per_dimension_mae'] for r in rows],0).tolist(),'rows':rows}; Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in out if k!='rows'},indent=2))
if __name__=='__main__': main()

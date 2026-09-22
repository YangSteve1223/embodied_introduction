#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from train_small_student_ddp import TinyPolicy, StudentDataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--run-dir',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',default='heldout'); ap.add_argument('--suites',default='libero_spatial_no_noops,libero_object_no_noops'); args=ap.parse_args()
    root=Path(args.data_root); suites=tuple(args.suites.split(',')); ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); vocab=ck['vocab']; ds=StudentDataset(root,suites,args.split,'',vocab); model=TinyPolicy(len(vocab)+2); model.load_state_dict(ck['model']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval(); rows=[]
    loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=2,pin_memory=device.type=='cuda')
    with torch.inference_mode():
        for img,q,gt,_,bow,ss,_ in loader:
            pred=model(img.to(device),q.to(device),bow.to(device)).cpu().numpy(); gt=gt.numpy(); e=np.abs(pred-gt)
            for j,s in enumerate(ss): rows.append({'suite':str(s),'full_chunk_l1':float(e[j].mean()),'first_action_l1':float(e[j,0].mean()),'per_dimension_mae':e[j].mean(0).tolist()})
    by={s:float(np.mean([r['full_chunk_l1'] for r in rows if r['suite']==s])) for s in suites}; out={'checkpoint':args.checkpoint,'split':args.split,'sample_count':len(rows),'per_suite':by,'macro_suite_normalized_action_l1':float(np.mean(list(by.values()))),'first_action_step_l1':float(np.mean([r['first_action_l1'] for r in rows])),'per_dimension_mae':np.mean([r['per_dimension_mae'] for r in rows],0).tolist(),'rows':rows}; Path(args.run_dir).mkdir(parents=True,exist_ok=True); (Path(args.run_dir)/f'{args.split}.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in out if k!='rows'},indent=2))
if __name__=='__main__': main()

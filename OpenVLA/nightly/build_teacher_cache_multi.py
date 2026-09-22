#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
from common import CHECKPOINT, MaterializedDataset, collator, forward_actions, load_policy

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--suites',default='libero_spatial_no_noops,libero_object_no_noops'); ap.add_argument('--split',default='train'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    vla,processor,head,projector=load_policy(Path(CHECKPOINT)); vla.eval(); head.eval(); projector.eval(); coll=collator(processor)
    manifest={'checkpoint':str(CHECKPOINT),'split':args.split,'suites':[]}
    for suite in [x for x in args.suites.split(',') if x]:
        ds=MaterializedDataset(suite,args.split,processor,use_proprio=True); preds=[]; ids=[]; start=time.time()
        for i in range(len(ds)):
            item=ds[i]; pred=forward_actions(vla,head,projector,coll([item]),train_mode=False)[0].float().cpu().numpy(); preds.append(pred)
            ids.append({'sample_index':int(i),'episode_id':str(item['sample_episode_id']),'timestep':int(item['sample_timestep'])})
            if (i+1)%25==0: print(f'{suite} cached {i+1}/{len(ds)}',flush=True)
        npy=out/f'{suite}_{args.split}.npy'; np.save(npy,np.stack(preds).astype(np.float32)); meta=out/f'{suite}_{args.split}.json'; meta.write_text(json.dumps({'checkpoint':str(CHECKPOINT),'suite':suite,'split':args.split,'count':len(preds),'ids':ids,'elapsed_sec':time.time()-start},indent=2)); manifest['suites'].append({'suite':suite,'count':len(preds),'npy':str(npy),'json':str(meta)})
    (out/f'manifest_{args.split}.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

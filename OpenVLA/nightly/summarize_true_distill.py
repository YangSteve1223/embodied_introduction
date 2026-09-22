#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
R=Path('/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260922')
base=json.load(open(R/'02_offline_baseline/baseline_heldout.json'))['metrics']
ev=json.load(open(R/'06_true_distill/small_student_ddp_1000/eval_1000/heldout.json'))
cfg=json.load(open(R/'06_true_distill/small_student_ddp_1000/config.json'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
m=ev
summary={'type':'independent_small_student_distillation','teacher':'official OpenVLA-7B fine-tuned LIBERO-Spatial checkpoint','student':'TinyPolicy: independent 6-channel CNN + bag-of-words language encoder + proprio MLP + action MLP','student_parameters':cfg['parameters'],'world_size':cfg['world_size'],'steps':cfg['max_steps'],'train_batch_size_per_rank':cfg['batch_size'],'teacher_weight':cfg['teacher_weight'],'teacher_targets':'official teacher action cache for Spatial train samples; Object uses ground-truth loss because teacher is not trained for Object','baseline_macro_l1':base['macro_suite_normalized_action_l1'],'student_macro_l1':m['macro_suite_normalized_action_l1'],'relative_change_vs_baseline':(m['macro_suite_normalized_action_l1']-base['macro_suite_normalized_action_l1'])/base['macro_suite_normalized_action_l1'],'student_per_suite':m['per_suite'],'student_first_action_l1':m['first_action_step_l1'],'student_checkpoint_sha256':sha(R/'06_true_distill/small_student_ddp_1000/student_step_1000.pt'),'limitations':['Student CNN was trained from scratch and is intentionally a minimal proof-of-concept, not a production VLA.','Language is a fixed vocabulary bag-of-words representation over the 20-task offline protocol.','No closed-loop simulator success was run.']}
(R/'06_true_distill/TRUE_DISTILL_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

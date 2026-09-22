# EXP-004

This directory contains the complete local/server workflow for the registered
mild combined-noise dose-response continuation experiment.

The scripts never connect to a server automatically.  Run them from the
repository root after synchronizing this directory and the already validated
EXP-002/EXP-003 code to the server.

## Main workflow

```bash
PY=/share/yangpengju-local/anaconda3/envs/maniskill/bin/python
ROOT=/share/yangpengju-local/embodied/embodied_introduction
RUNS=/share/yangpengju-local/embodied/runs/exp004
EXP002=/share/yangpengju-local/embodied/runs/exp002
EXP003=/share/yangpengju-local/embodied/runs/exp003

cd "$ROOT"
"$PY" -m exp004.launch_smoke \
  --config exp004/configs/experiment.json \
  --exp002-runs-root "$EXP002" \
  --output-root "$RUNS/smoke_$(date +%Y%m%d_%H%M%S)" \
  --python "$PY" --gpus 0 1 2 3 --timeout-seconds 1800

"$PY" -m exp004.launch_training \
  --config exp004/configs/experiment.json \
  --exp002-runs-root "$EXP002" \
  --output-root "$RUNS/formal_training" \
  --python "$PY" --gpus 0 1 2 3 --timeout-seconds 14400

"$PY" -m exp004.launch_evaluation \
  --config exp004/configs/experiment.json \
  --exp002-runs-root "$EXP002" \
  --exp003-runs-root "$EXP003" \
  --formal-runs-root "$RUNS/formal_training" \
  --output-root "$RUNS/formal_evaluation" \
  --python "$PY" --gpus 0 1 2 3 --stages primary secondary \
  --timeout-seconds 3600

"$PY" -m exp004.summarize \
  --config exp004/configs/experiment.json \
  --evaluation-root "$RUNS/formal_evaluation" \
  --output-json "$RUNS/exp004_summary.json"
```

The smoke is one bounded engineering gate.  It is never included in the
formal matrices.  The formal launcher runs three four-GPU waves, one wave per
training seed, and stops before the next wave if any job fails.  The evaluator
always completes all 60 primary cells before starting the 48 secondary cells.
Cross-severity evaluation is intentionally a separate optional command and is
not needed to close the primary experiment.

`--resume` only skips a run or cell with a complete output.  A non-empty
incomplete directory is an error requiring manual inspection; no script
overwrites old experiment output.

If the primary and secondary summary is already closed and GPU time remains,
the optional cross-severity diagnostic can be launched separately:

```bash
"$PY" -m exp004.launch_evaluation \
  --config exp004/configs/experiment.json \
  --exp002-runs-root "$EXP002" --exp003-runs-root "$EXP003" \
  --formal-runs-root "$RUNS/formal_training" \
  --output-root "$RUNS/formal_evaluation" \
  --python "$PY" --gpus 0 1 2 3 --stages cross \
  --timeout-seconds 3600
```

This produces the registered 1x/2x/3x diagnostic grid (162 cells) but is not
required by `summarize.py` and cannot change the pre-registered gates.

## Narrow Mac-to-server synchronization

From the local Mac, this copies only the new experiment and the ignore rule;
it does not touch EXP-002/003 or delete anything on the server:

```bash
cd /Users/yangda/Documents/TaskandWork/embodied_introduction
rsync -av --exclude='results/' --exclude='*.pt' --exclude='*.pth' \
  exp004/ act-server:/share/yangpengju-local/embodied/embodied_introduction/exp004/
rsync -av .gitignore \
  act-server:/share/yangpengju-local/embodied/embodied_introduction/.gitignore
```

Before running it, inspect the server tree with `ssh act-server 'cd
/share/yangpengju-local/embodied/embodied_introduction && git status --short'`.
If existing EXP-002/003 edits are present, leave them in place and use the
narrow sync above; do not run a blind pull or cleanup.

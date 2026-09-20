# EXP-003 — Calibrated interface-noise PPO post-training

EXP-003 continues the three successful EXP-002 clean policies under a paired
control/treatment design. The control receives clean PPO continuation. The
treatment receives the same continuation budget with a calibrated combined
observation/action-noise curriculum.

No script in this directory connects to a server automatically. The launchers
operate only after the user invokes them inside the server repository.

## Registered workflow

1. Run the single combined smoke gate. It covers unit tests, source evaluation,
   clean continuation, curriculum continuation, checkpoint writing, checkpoint
   reloading, and clean/noisy evaluation.
2. Run the 60-cell evaluation-only calibration matrix.
3. Summarize calibration and freeze the smallest eligible combined multiplier.
4. Launch six formal continuation runs: control/treatment for seeds
   1001, 1002, and 1003.
5. Evaluate original/control/treatment checkpoints in 36 formal cells.
6. Validate and summarize the formal result, then update the experiment record.

The earlier one-seed pilot stage was removed by user decision on 2026-09-20 to
reduce turnaround time. Formal periodic checkpoints remain diagnostic only;
the final checkpoint is always the primary result.

## Public and private files

Code, tests, configuration, and the design contract belong in Git. Runtime
artifacts belong under `/share/yangpengju-local/embodied/runs/exp003` on the
server and are copied back to `exp003/results/` locally. The local result
directory is ignored by Git.

## Main entry points

- `python -m exp003.smoke`: one bounded four-GPU smoke gate.
- `python -m exp003.launch_calibration`: parallel calibration evaluations.
- `python -m exp003.summarize_calibration`: validation and severity freeze.
- `python -m exp003.launch_formal_training`: six paired continuation runs.
- `python -m exp003.launch_formal_evaluation`: 36 formal evaluation cells.
- `python -m exp003.summarize_formal`: paired effect and gate report.

See `EXPERIMENT_PLAN.md` for the registered variables and interpretation rules.

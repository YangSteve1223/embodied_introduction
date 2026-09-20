# EXP-004 — Mild combined-noise dose-response PPO continuation

EXP-004 tests whether the EXP-003 failure was caused by an overly strong
combined-noise dose.  It continues the same three EXP-002 clean policies with
matched clean control C0 and combined-noise treatments C1, C2, and C3.

The only training variable is the combined interface-noise multiplier:

```text
C0 = 0x clean continuation
C1 = 1x combined observation + bounded action noise
C2 = 2x combined observation + bounded action noise
C3 = 3x combined observation + bounded action noise
C4 = EXP-003 treatment at 4x, update 100 reference only
```

All arms use the same task, source checkpoint, actor/critic architecture,
optimizer state, PPO hyperparameters, 2048 environments, 100-step rollouts,
30,720,000 continuation timesteps, three training seeds, and evaluation
protocol.  Treatment arms ramp linearly from 0 to their target over updates
1–20, then remain fixed through update 150.  C0 remains exactly clean.

The pre-registered endpoints are update 100 (primary) and update 150
(secondary).  Update 100 is comparable to EXP-003's 4x reference budget;
update 150 measures whether longer adaptation changes the result.  Periodic
checkpoints are diagnostic only.

For each treatment arm `Cs`, the matched effects under common combined-4x
evaluation are:

```text
robustness_gain_s = success(Cs, combined-4x) - success(C0, combined-4x)
clean_cost_s      = success(Cs, clean) - success(C0, clean)
```

A treatment passes only if mean robustness gain is at least +5 percentage
points, at least two of three seed effects are positive, mean clean cost is no
worse than -5 percentage points, noisy return is not below matched C0, and
noise-induced action clipping is not abnormal (5% threshold).

This remains state-only on-policy PPO.  It is not a visual VLA result, an
imitation-learning result, or an offline-RL result.

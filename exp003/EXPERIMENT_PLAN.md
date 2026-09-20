# EXP-003 registered experiment plan

## Research question

Starting from the same trained clean PegInsertionSide policy, does calibrated
interface-noise PPO continuation improve noisy-evaluation success relative to
an equal-budget clean continuation, without materially damaging clean success?

## Paired comparison

```text
Control:   EXP-002 clean checkpoint -> clean PPO continuation
Treatment: EXP-002 clean checkpoint -> combined-noise curriculum continuation
```

The paired arms share source actor/critic parameters, optimizer state, source
checkpoint, continuation seed, task, controller, environment count, rollout
length, PPO settings, additional timesteps, checkpoint schedule, and evaluation
protocol. The only treatment variable is interface noise during continuation.

EXP-002 did not save RNG or simulator state. Each pair therefore starts from a
fresh, identically seeded environment reset. Policy sampling uses the shared
PyTorch RNG stream; observation and action noise use separate generators so
the treatment does not consume policy-sampling random numbers.

## Stage A — evaluation-only calibration

Sources: EXP-002 `clean_seed1001`, `clean_seed1002`, and `clean_seed1003`.

```text
components: observation, action, combined
multipliers: 1x, 2x, 4x
canonical clean baseline: none / 0x
evaluation seeds: 20260920, 20260921
episodes per cell: 256
evaluation environments: 64
policy action: deterministic actor mean
reconfiguration_freq: 1
```

The canonical matrix contains 60 cells: three source checkpoints, two
evaluation seeds, and ten unique noise conditions. The duplicated 0x entries
of a rectangular 3x4 component table are represented by the same canonical
clean result.

The selected severity is the smallest combined multiplier satisfying all of:

- paired mean success drop between 10 and 25 percentage points;
- positive checkpoint-averaged drop for all three training seeds;
- mean success at least 5%, avoiding a complete floor result;
- mean noise-induced action clipping no greater than 5%;
- success and return remain interpretable under manual review.

If no multiplier passes, formal training is blocked pending noise redesign.

## Combined smoke gate

The single smoke round uses `clean_seed1002` and four GPUs. It runs:

- EXP-003 unit tests;
- source checkpoint clean and 1x-combined evaluation, 8 episodes each;
- control and treatment continuation with 16 environments, 5 rollout steps,
  two executed PPO updates, and checkpoint save/reload;
- clean and 1x-combined evaluation of both continuation checkpoints.

This is an engineering gate only. It is never included in calibration or
formal results.

## Stage B — formal paired continuation

The separate exploratory pilot was removed by user decision. Formal runs are:

```text
clean_seed1001 -> control_seed1001 / treatment_seed1001
clean_seed1002 -> control_seed1002 / treatment_seed1002
clean_seed1003 -> control_seed1003 / treatment_seed1003
```

Registered continuation settings:

```text
num_envs: 2048
num_steps: 100
batch_size: 204800
additional_timesteps: 20480000
continuation_updates: 100
checkpoint_interval_updates: 10
curriculum: rollout-level linear warmup for updates 1-20, then fixed
primary checkpoint: final checkpoint after update 100
```

PPO settings match the actual EXP-002 checkpoints, including
`target_kl = null`.

## Formal evaluation

For each training seed, evaluate the original source, control final checkpoint,
and treatment final checkpoint under clean and frozen calibrated-noisy
conditions. Each cell uses both fixed evaluation seeds and 256 episodes per
seed. The formal matrix therefore contains 36 JSON cells.

Primary effects:

```text
robustness_gain = treatment/noisy - control/noisy
clean_cost      = treatment/clean - control/clean
```

Engineering success requires:

- mean robustness gain at least +5 percentage points;
- at least two of three training seeds positive;
- mean clean cost no worse than -5 percentage points;
- no material return collapse;
- no abnormal clipping mechanism.

The original policy comparison separates the noise treatment effect from the
effect of additional clean PPO training time.

## Required closure report

After the formal experiment, report the research question, changed and fixed
variables, online data source, observation/action/reward/done semantics, PPO
loss, checkpoint lineage, ID/OOD metrics, seed variance, failure cases,
confounders, conclusion, and one next experimental change. Update
`EXPERIMENT_RECORD.md` at every evidence-bearing stage.

# FMCPE

Code from the paper "Flow Matching for Robust Simulation-Based Inference under Model Misspecification" (https://arxiv.org/abs/2509.23385).

## Setup

The environment is managed with [pixi](https://pixi.sh). Two environments are defined:

```bash
pixi shell          # CPU environment (local workstation)
pixi shell -e gpu   # CUDA 12.x environment (GPU clusters)
```

Or run a single command without entering a shell:

```bash
pixi run python run_benchmark.py ...        # CPU
pixi run -e gpu python run_benchmark.py ... # GPU
```

Python is pinned to 3.12. Core dependencies include PyTorch, JAX, `lampe`, `zuko`,
`sbi`, `numpyro`, and Hydra. See `pixi.toml` for the full specification.

> A legacy `requirements.txt` exists for a `pip`/`conda` install, but `pixi` is the
> supported path and the only one with a pinned lockfile (`pixi.lock`).

## Quick start

Experiments are driven by Hydra. The benchmark runs as a 4-phase pipeline; here is a
full single-task run (`pure_gaussian`):

```bash
EXP=comparison_benchmark
TASK=pure_gaussian
VARIANT=fm_pt_base
RESULTS=./results

# Phase 1 — foundation: simulation models (NPE, FMPE) + baselines
pixi run python run_benchmark.py command=train_foundation \
    +experiment=$EXP "tasks=[$TASK]" results_dir=$RESULTS

# Phase 2 — variant: train the calibration method across seeds × calibration sizes
pixi run python run_benchmark.py command=train_variant \
    +experiment=$EXP +variant=$VARIANT +variant_name=$VARIANT \
    "tasks=[$TASK]" results_dir=$RESULTS

# Phase 3 — evaluate everything on the held-out test set
pixi run python run_benchmark.py command=evaluate_all \
    +experiment=$EXP "tasks=[$TASK]" results_dir=$RESULTS

# Phase 4 — aggregate into CSV / LaTeX tables + summary report
pixi run python run_benchmark.py command=aggregate \
    +experiment=$EXP "tasks=[$TASK]" results_dir=$RESULTS
```

`command=run_all` runs phases 1, 3 and 4 in sequence (skipping variant training).

### Hydra CLI notes

- `+experiment=<name>` **must** use the leading `+` (the experiment group is loaded,
  not overridden).
- `+variant=<name>` loads the variant group; the top-level `variant_name` must also
  be set explicitly with `+variant_name=<name>`.
- Profile overrides use the group path: `+experiment/_profiles=sanity`.
- Quote task lists so the shell does not expand the brackets: `'tasks=[pure_gaussian]'`.

### Faster smoke test

```bash
pixi run python run_benchmark.py command=evaluate_all \
    +experiment=comparison_benchmark "tasks=[pure_gaussian]" \
    evaluation.num_joint_samples=100 \
    evaluation.num_posterior_samples=100 \
    evaluation.num_conditional_obs=2
```

## Project layout

```
benchmark/    4-phase pipeline: foundation, variants, evaluation, aggregation
training/     Registry-based trainer abstraction (base, registry, trainers/)
simulator/    Inference tasks (pure_gaussian, pendulum, sir, wind_tunnel, light_tunnel, js, ...)
flow_matching/Flow-matching posterior model (torch_flow.py)
baselines/    Alternative estimators (NPE, DPE, MF-NPE, ROPE) and posterior wrappers
posteriors/   Posterior estimator classes
core/         Checkpointing and shared infrastructure
utils/        Rescaling, networks, plotting, file locking
configs/      Hydra configs: config.yaml + task/, experiment/, variant/, method/
scripts/      One-off analysis / diagnostics scripts
tests/        Pytest suite
```

### Available tasks

`pure_gaussian`, `gaussian`, `adaptive_gaussian`, `high_dim_gaussian`,
`high_dim_adaptive_gaussian`, `pendulum`, `sir`, `wind_tunnel`, `light_tunnel`,
`js`, `ou_process`, plus the misspecification / independence study tasks
(`no_misspec_*`, `independence_*`).

### Methods

- **Simulation models** (trained once on simulation data): `npe`, `fmpe`.
- **Baselines** (trained on calibration data): `dpe`, `mf_npe`, `rope`.
- **Calibration variant**: `fm_pt_base` and related `fm_pt_*` variants — the
  flow-matching post-transform method this work introduces. See `configs/variant/`.

## Results layout

```
results/{experiment}/
    shared/                  # Phase 1 (shared across seeds): data + simulation models
    variants/{variant}/      # Phase 2: per-seed calibration models
    aggregated/              # Phase 3 & 4: metrics, tables, reports
```

## Development

```bash
pixi run test        # pytest tests/
pixi run lint        # ruff check .
pixi run typecheck   # basedpyright
```

## License

MIT — see [LICENSE](LICENSE).

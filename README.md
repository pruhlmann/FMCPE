# FMCPE: Flow Matching Corrected Posterior Estimation

Official implementation of Flow Matching Corrected Posterior Estimation (FMCPE), a method for posterior estimation in simulation-based inference using flow matching techniques.

## Installation

### Setting up the Environment

The recommended way to set up the environment is through conda:

```bash
# Create a new conda environment with Python 3.12
conda create -n fmcpe python=3.12

# Activate the environment
conda activate fmcpe

# Install required dependencies
pip install -r requirements.txt
```

### Requirements

The main dependencies include:
- PyTorch
- lampe (0.9.0)
- mlxp (1.0.0)
- numpyro (0.18.0)
- scikit-learn (1.6.1)
- scipy (1.12.0)
- pyro-ppl (1.9.1)
- seaborn, matplotlib, tqdm
- POT (Python Optimal Transport)

See `requirements.txt` for the complete list.

## Quick Start

### Basic Usage

The codebase provides several benchmark tasks for evaluating posterior estimation methods:
- `high_dim_gaussian`: High-dimensional Gaussian inference
- `pendulum`: Pendulum dynamics inference
- `light_tunnel`: Light tunnel simulation
- `wind_tunnel`: Wind tunnel simulation

### Running Experiments

To run a simple experiment:

```bash
# Train baseline methods
python main.py task=high_dim_gaussian \
    task.num_samples=1000 \
    task.num_cal=[10,50,100] \
    train_baselines=True \
    seed=0 \
    +mlxp.logger.parent_log_dir=logs/experiment
```

### Configuration Parameters

Key parameters you can configure:
- `task`: The inference task to use (`high_dim_gaussian`, `pendulum`, etc.)
- `task.num_samples`: Number of simulation samples for training
- `task.num_cal`: List of calibration sample sizes to evaluate
- `seed`: Random seed for reproducibility
- `train_baselines`: Whether to train baseline methods (DPE, NPE)
- `compute_reference`: Whether to compute reference posteriors with max ncal

### Analysis

After training, compute evaluation metrics:

```bash
python analysis.py \
    --logdir=logs/experiment \
    --task high_dim_gaussian \
    --metrics seeded # or 'ncal' if only one seed
    --c2st \
    --wasserstein \
    --num_test=100
```

Available metrics:
- `--c2st`: Classifier Two-Sample Test on the joint (theta,x)
- `--wasserstein`: Wasserstein distance
- `--mse`: Mean Squarred Error avregaged over observations
- `--num_test`: Number of test samples to use

### Visualization

Generate plots from your experiments:

```bash
python make_plots.py \
    --log_dir logs/experiment \
    --save_root figures/ \
    --task high_dim_gaussian
```

## Advanced Usage

### Multiple Seeds and Calibration Sizes

Run experiments across multiple configurations:

```bash
python main.py \
    task=high_dim_gaussian,pendulum \
    task.num_samples=1000 \
    task.num_cal=[10,50,100,500] \
    seed=[0,1,2,3,4] \
    +mlxp.logger.parent_log_dir=logs/multi_run
```

This will launch experiments for:
- 2 tasks × 4 calibration sizes × 5 seeds = 40 runs

**Note:** Runs can be launched in parallel. See the [mlxp documentation](https://github.com/inria-thoth/mlxp) for details.

### Configuration Files

The configuration system uses [mlxp](https://github.com/inria-thoth/mlxp) for experiment management. Config files are organized as:

```
configs/
├── config.yaml          # Base configuration
├── mlxp.yaml           # Logging and experiment tracking
└── task/               # Task-specific configs
    ├── high_dim_gaussian.yaml
    ├── light_tunnel.yaml
    ├── pendulum.yaml
    └── wind_tunnel.yaml
```

Each task configuration contains:
- **Task parameters**: `num_samples`, `num_cal`, `rescale`, `generation`
- **Simulator settings**: Parameters for the specific simulator
- **NPE configuration**: Neural Posterior Estimation hyperparameters
- **FMPE configuration**: Flow Matching parameters (time prior, probability paths, drift architecture)
- **Method-specific settings**: Training parameters for each method

### Available Methods

The codebase implements several methods:
- `fm_post_transform`: Flow Matching Posterior Transform (FMCPE)
- `dpe`: Direct Posterior Estimation (baseline)
- `mf_npe`: Mean-field NPE (baseline)

## Repository Structure

```
.
├── baselines/           # Baseline methods (NPE, DPE)
├── configs/            # Configuration files
├── flow_matching/      # Flow matching implementation
├── simulator/          # Task simulators
├── utils/              # Utility functions (metrics, plotting, etc.)
├── scripts/            # Additional experimental scripts
├── main.py            # Main training script
├── analysis.py        # Evaluation metrics computation
└── make_plots.py      # Visualization generation
```

## Example Workflow

Here's a complete workflow from training to visualization:

```bash
# 1. Train models
python main.py \
    task=high_dim_gaussian \
    task.num_samples=50000 \
    task.num_cal=[10,50,200,1000] \
    train_baselines=True \
    seed=0 \
    +mlxp.logger.parent_log_dir=logs/demo

# 2. Compute metrics
python analysis.py \
    --logdir=logs/demo \
    --metrcics seeded
    --recompute_baselines
    --recompute_method
    --c2st \
    --wasserstein \
    --num_test=2000

# 3. Generate plots
python make_plots.py \
    --log_dir logs/demo \
    --save_root figures/demo/
```

## Adding a new task

To add your own task you have to : 
- Create a `my_task.py` file in `simulator/` follwing one of the existing task as template.
- Create a `configs/task/my_task.yaml` with task specific parameters (follow of the existing configs structure)
- Add your task in `simulator/__init__.py`
- Create the necessary embedding nets in `utils/networks.py` and add it in `get_embedding_network` line 853.

**Note : Rescaling is handled internally with the nf and fm models, but it is strongly encouraged to already return standardized data in the simulator (check the light tunnel task) and set rescal=none in the configs**

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ruhlmann2025flowmatchingrobustsimulationbased,
      title={Flow Matching for Robust Simulation-Based Inference under Model Misspecification}, 
      author={Pierre-Louis Ruhlmann and Pedro L. C. Rodrigues and Michael Arbel and Florence Forbes},
      year={2025},
      eprint={2509.23385},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2509.23385}, 
}
```

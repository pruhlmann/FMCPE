import argparse
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import torch

from simulator import get_simulator
from baselines import train_dpe
from simulator.base import generate_dataset
import matplotlib.pyplot as plt
import seaborn as sns

from utils.misc import rescale, tensor_to_df

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get simulator and generate dataset
task_params = {
    "dim": 200,  # Number of time points (N)
    "low_w0": 0.0,  # Lower bound for w
    "high_w0": 3.0,  # Upper bound for w
    "low_A": 0.5,  # Lower bound for A
    "high_A": 10.0,  # Upper bound for A
    "tmax": 10.0,  # Maximum time
    "noise": 0.1,  # Noise level
}
training_params = {
    "epochs": 10_000,
    "batch_size": 256,
    "lr": 5e-4,
    "train_size": 0.8,
}
npe_params = {
    "embedding_net": {"model_path": "models", "load": False, "save": "last_", "output_dim": 10},
    "npe_params": {"embedding_dim": 10, "ntransform": 3},
}
simulator = get_simulator("pendulum", **task_params)
num_samples = 50000
theta, x, y, scales = generate_dataset(
    simulator,
    num_samples,
    rescale="standardize",
)
theta_test, x_test, y_test, scales_test = generate_dataset(
    simulator,
    5000,
    rescale="standardize",
)

dpe_model = train_dpe(
    task="pendulum",
    theta=theta,
    x=x,
    simulator=simulator,
    device=device,
    training_params=training_params,
    logger=None,
    logname="pendulum_dpe",
    load=False,
    **npe_params,
    density_estimator="maf",
)

npe_model = train_npe_estimator(
    task="pendulum",
    simulator=simulator,
    theta=theta,
    x=x,
    logger=None,
    device=device,
    log_name="pendulum_npe",
    training_params=training_params,
    load=False,
    **npe_params,
)

torch.save(dpe_model.state_dict(), "models/dpe_model_pendulum.pth")
torch.save(npe_model.state_dict(), "models/npe_model_pendulum.pth")

theta_posteriors_dpe = dpe_model._flow(y_test.to(device)).sample().cpu()
theta_posteriors_npe = npe_model.flow(y_test.to(device)).sample().cpu()
theta_posteriors_dpe = rescale(
    theta_posteriors_dpe, scales_test["theta_mean"], scales_test["theta_std"]
)
theta_posteriors_npe = rescale(
    theta_posteriors_npe, scales_test["theta_mean"], scales_test["theta_std"]
)
theta_test = rescale(theta_test, scales_test["theta_mean"], scales_test["theta_std"])

df_npe = tensor_to_df(theta_posteriors_npe, "npe", ["w_o", "A"])
df_dpe = tensor_to_df(theta_posteriors_dpe, "dpe", ["w_o", "A"])
df_test = tensor_to_df(theta_test, "true", ["w_o", "A"])
df_all = pd.concat([df_npe, df_dpe], ignore_index=True)
g = sns.JointGrid(
    data=df_all,
    x="w_o",
    y="A",
    hue="type",
)
g.plot_marginals(sns.histplot, alpha=0.5)
g.plot_joint(sns.scatterplot, alpha=0.5, s=5)
sns.histplot(
    x=df_test["w_o"], ax=g.ax_marg_x, color="gray", alpha=0.8, element="step", bins=30, fill=False
)
sns.histplot(
    y=df_test["A"], ax=g.ax_marg_y, color="gray", alpha=0.8, element="step", bins=30, fill=False
)
plt.sca(g.ax_joint)
plt.plot(
    [task_params["low_w0"], task_params["high_w0"]],
    [task_params["low_A"], task_params["low_A"]],
    "r--",
)
plt.plot(
    [task_params["low_w0"], task_params["high_w0"]],
    [task_params["high_A"], task_params["high_A"]],
    "r--",
)
plt.plot(
    [task_params["low_w0"], task_params["low_w0"]],
    [task_params["low_A"], task_params["high_A"]],
    "r--",
)
plt.plot(
    [task_params["high_w0"], task_params["high_w0"]],
    [task_params["low_A"], task_params["high_A"]],
    "r--",
)
plt.savefig("figures/pendulum_posterior.pdf")


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pendulum")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    main()

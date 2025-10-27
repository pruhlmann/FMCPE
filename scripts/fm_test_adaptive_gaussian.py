from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib import gridspec

from flow_matching.torch_flow import FlowMatching, train_flow_matching
from simulator import get_simulator
from simulator.base import generate_dataset
from utils.misc import rescale
from utils.plots import pair_plot_tensors, plot_closest_samples

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Found deouovice: {device}")
task_params = {
    "theta_dim": 2,
    "obs_dim": 2,
    "prior_var_scale": 3,
    "s": 0.8,
    "t": 1.4,
    "seed": 42,
}
# Get simulator and generate dataset
simulator = get_simulator("adaptive_gaussian", **task_params)
num_samples = 50000
thetad, xd, yd, scales = generate_dataset(
    simulator,
    num_samples,
    # rescale="standardize",
    rescale=None,
)

flow_matching = FlowMatching(
    conditional=False,
    probability_path="ot2",
    prior="power",
    base_dist="none",
    # base_dist="none",
    dim=xd.shape[1],
    prior_params={"rate": 2.0},
    drift={"architecture": "attention_mlp"},
)
flow_matching.to(device)
model_path = Path("scripts/models")
model_path.mkdir(parents=True, exist_ok=True)
noise = torch.randn_like(xd)
logname = "_adaptive_gaussian_x_no_cond"

# assert torch.allclose(noise[:100], flow_matching.sample_source(noise[:100]))
# print(f"{((noise - flow_matching.sample_source(noise)) ** 2).sum(dim=1).sqrt().mean()}")

if Path(model_path / f"flow_matching{logname}.pth").exists():
    flow_matching.load_state_dict(torch.load(model_path / f"flow_matching{logname}.pth"))

else:
    flow_matching = train_flow_matching(
        flow_matching,
        xd,
        noise,
        device,
        None,
        model_path,
        logname=logname,
        epochs=500,
        batch_size=1024,
        lr=5e-5,
    )

x_tilde = flow_matching.sample_batched(noise[:5000], 1, device, only_last=False)

# x_tilde_scaled = rescale(x_tilde[:, -1, :], scales["x_mean"], scales["x_std"])
# source_scaled = rescale(x_tilde[:, 0, :], scales["x_mean"], scales["x_std"])
# xd_scaled = rescale(xd[:1000], scales["x_mean"], scales["x_std"])
x_tilde_scaled = x_tilde[:, -1, :]
source_scaled = x_tilde[:, 0, :]
xd_scaled = xd[:5000]

savedir = Path("figures/scripts/adaptive_gaussian_baselines")
savedir.mkdir(parents=True, exist_ok=True)
# Convert tensors to DataFrames with labels
df_samples = pd.DataFrame(x_tilde_scaled.numpy(), columns=["x", "y"])
df_samples["label"] = "Samples"

df_source = pd.DataFrame(source_scaled.numpy(), columns=["x", "y"])
df_source["label"] = "Source"

df_data = pd.DataFrame(xd_scaled.numpy(), columns=["x", "y"])
df_data["label"] = "Data"

# Combine all dataframes
df_all = pd.concat([df_samples, df_source, df_data], ignore_index=True)
# Use sns.jointplot
sns.set_theme(style="whitegrid")
g = sns.jointplot(
    data=df_all,
    x="x",
    y="y",
    hue="label",
    kind="scatter",
    joint_kws={"s": 5},
    marginal_kws={"fill": False},
)
plt.savefig(savedir / "fm_on_x.pdf")

# pair_plot_tensors(xd_scaled, x_tilde_scaled, model_path, scatter_size=2, save=True)

# plot_closest_samples(xd_scaled, x_tilde_scaled, 5, 3, model_path)

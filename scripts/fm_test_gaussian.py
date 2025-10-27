from pathlib import Path

from lampe.inference import FMPE, FMPELoss
from lampe.utils import GDStep
import matplotlib
from tqdm import trange

matplotlib.use("Agg")  # use the “Agg” backend—no X server needed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.usetex": True,
    }
)
from flow_matching.torch_flow import FlowMatching, train_flow_matching
from simulator import get_simulator
from simulator.base import generate_dataset
from utils.misc import loader_from_tensor, rescale, train_val_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Found device: {device}")
task_params = {
    "theta_dim": 2,
    "obs_dim": 2,
    "prior_var_scale": 1,
    "likelihood_var_scale": 1,
    "noisy_var_scale": 1,
    "seed": 0,
}
rescaling = "whitening"
# rescaling = None
# Get simulator and generate dataset
simulator = get_simulator("pure_gaussian", **task_params)
num_samples = 25000
thetad, xd, yd, scales = generate_dataset(
    simulator,
    num_samples,
    # rescale="standardize",
    generation="transitive",
    rescale=rescaling,
)
thetad = thetad.reshape(-1, thetad.shape[-1])
theta_o, xd_o, yd_o, scales_o = generate_dataset(simulator, 2000, rescale=rescaling)

flow_matching = FlowMatching(
    conditional=False,
    probability_path="ot",
    prior="uniform",
    base_dist="gaussian",
    # base_dist="none",
    dim=xd.shape[1],
    # prior_params={"rate": 2.0},
    drift={"architecture": "resmlp", "hidden_dim": [64, 64]},
)
flow_matching.to(device)
model_path = Path("scripts/models")
model_path.mkdir(parents=True, exist_ok=True)
noise = torch.randn_like(xd)
logname = "_pure_gaussian_x_no_cond"

# assert torch.allclose(noise[:100], flow_matching.sample_source(noise[:100]))
# print(f"{((noise - flow_matching.sample_source(noise)) ** 2).sum(dim=1).sqrt().mean()}")

retrain = False
if Path(model_path / f"flow_matching{logname}.pth").exists() and not retrain:
    flow_matching.load_state_dict(torch.load(model_path / f"flow_matching{logname}.pth"))
    flow_matching.to(device)

else:
    flow_matching = train_flow_matching(
        flow_matching,
        xd,
        noise,
        device,
        None,
        model_path,
        logname=logname,
        epochs=300,
        batch_size=256,
        lr=1e-4,
    )

# x_tilde = flow_matching.sample_batched(yd_o, 1, device, only_last=False)
#
# x_tilde_scaled = rescale(x_tilde[:, -1, :], scales["x_mean"], scales["x_std"])
# source_scaled = rescale(x_tilde[:, 0, :], scales["x_mean"], scales["x_std"])
xd_o_scaled = rescale(xd_o, scales["x_mean"], scales["x_std"])
# x_tilde_scaled = x_tilde[:, -1, :]
# source_scaled = x_tilde[:, 0, :]
# xd_scaled = xd[:1000]

savedir = Path("figures/scripts/pure_gaussian_baselines")
savedir.mkdir(parents=True, exist_ok=True)
# Convert tensors to DataFrames with labels
# df_samples = pd.DataFrame(x_tilde_scaled.numpy(), columns=["x", "y"])
# df_samples["label"] = "Samples"
#
# df_source = pd.DataFrame(source_scaled.numpy(), columns=["x", "y"])
# df_source["label"] = "Source"
#
# df_data = pd.DataFrame(xd_o_scaled.numpy(), columns=["x", "y"])
# df_data["label"] = "Data"
#
# # Combine all dataframes
# df_all = pd.concat([df_samples, df_source, df_data], ignore_index=True)
# # Use sns.jointplot
# sns.set_theme(style="whitegrid")
# g = sns.jointplot(
#     data=df_all, x="x", y="y", hue="label", kind="scatter", marginal_kws={"fill": False}
# )
# g.figure.suptitle("Joint Scatter and Marginal Histograms")
# plt.savefig(savedir / "fm_on_x.pdf")


flow_matching_conditional = FlowMatching(
    conditional=True,
    probability_path="ot2",
    # prior="power",
    prior="uniform",
    base_dist="gaussian",
    # base_dist="none",
    dim=xd.shape[1],
    # prior_params={"rate": 2},
    drift={"architecture": "attention_mlp", "hidden_dim": [64, 64, 64, 64, 64]},
)
flow_matching.to(device)
model_path = Path("scripts/models")
model_path.mkdir(parents=True, exist_ok=True)
noise = torch.randn_like(xd)
logname = "_pure_gaussian_x_cond"

# assert torch.allclose(noise[:100], flow_matching.sample_source(noise[:100]))
# print(f"{((noise - flow_matching.sample_source(noise)) ** 2).sum(dim=1).sqrt().mean()}")

retrain = True
if Path(model_path / f"flow_matching{logname}.pth").exists() and not retrain:
    flow_matching_conditional.load_state_dict(
        torch.load(model_path / f"flow_matching{logname}.pth")
    )
    flow_matching_conditional.to(device)
else:
    flow_matching_conditional = train_flow_matching(
        flow_matching_conditional,
        xd,
        yd,
        device,
        None,
        model_path,
        logname=logname,
        epochs=300,
        batch_size=256,
        lr=1e-3,
    )

# train FMPE for reference
estimator = FMPE(2, 2, hidden_features=[64] * 5, activation=torch.nn.ELU)
loss = FMPELoss(estimator)
optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 128)
step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

estimator.train()
loader, _ = train_val_split(xd, yd, batch_size=256, train_size=0.8)

for epoch in (bar := trange(128, unit="epoch")):
    losses = []

    for theta, x in loader:  # 256 batches per epoch
        losses.append(step(loss(theta, x)))

    bar.set_postfix(loss=torch.stack(losses).mean().item())

with torch.no_grad():
    samples = estimator.flow(yd_o[:4].to(device)).sample((1000,)).cpu()

source = flow_matching_conditional.sample_source(yd_o, 1).squeeze(0)
x_tilde = flow_matching_conditional.sample(source, yd_o, only_last=False)

x_tilde_scaled = rescale(x_tilde[:, -1, :], scales["x_mean"], scales["x_std"])
source_scaled = rescale(x_tilde[:, 0, :], scales["x_mean"], scales["x_std"])

# Convert tensors to DataFrames with labels
df_samples = pd.DataFrame(x_tilde_scaled.numpy(), columns=["x", "y"])
df_samples["label"] = "Samples"

df_data = pd.DataFrame(xd_o_scaled.numpy(), columns=["x", "y"])
df_data["label"] = "Data"

# Combine all dataframes
df_all = pd.concat([df_samples, df_data], ignore_index=True)
# Use sns.jointplot
# sns.set_theme(style="whitegrid")
g = sns.jointplot(
    data=df_all, x="x", y="y", hue="label", kind="scatter", marginal_kws={"fill": False}
)
g.figure.suptitle("Joint Scatter and Marginal Histograms")
plt.savefig(savedir / "fm_on_x_cond.pdf")
# pair_plot_tensors(xd_scaled, x_tilde_scaled, model_path, scatter_size=2, save=True)

observation = yd_o[:4].repeat_interleave(1000, 0)
obs = flow_matching_conditional.sample(torch.randn_like(observation), observation, only_last=True)
obs_scaled = rescale(obs, scales["x_mean"], scales["x_std"])
obs_scaled = obs_scaled.reshape(4, 1000, 2).transpose(1, 0)
yd_o_scaled = rescale(yd_o, scales["y_mean"], scales["y_std"])
gt_scaled = simulator.denoise_dist(yd_o_scaled[:4]).sample((1000,)).cpu()

print("plotting obs")
obs_patch = Patch(
    facecolor=sns.color_palette("Blues", as_cmap=False)[2], alpha=0.5, label=r"$x|y_{obs}$"
)
gt_patch = Patch(
    facecolor=sns.color_palette("Reds", as_cmap=False)[2], alpha=0.5, label="Ground Truth"
)
fmcpe_patch = Patch(
    facecolor=sns.color_palette("Greens", as_cmap=False)[2], alpha=0.5, label="FMPE"
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i in range(4):
    row, col = divmod(i, 2)
    ax = axes[row][col]
    ax.set_aspect("equal")

    obs = obs_scaled[:, i, :].numpy()
    gt = gt_scaled[:, i, :].numpy()
    fmcpe = samples[:, i, :].numpy()

    # plot the two KDEs (no label needed here)
    sns.kdeplot(x=gt[:, 0], y=gt[:, 1], ax=ax, fill=True, cmap="Reds", alpha=0.8)
    sns.kdeplot(x=obs[:, 0], y=obs[:, 1], ax=ax, fill=True, cmap="Blues", alpha=0.5)
    # sns.kdeplot(x=fmcpe[:, 0], y=fmcpe[:, 1], ax=ax, fill=True, cmap="Greens", alpha=0.5)
    # sns.kdeplot(
    #     x=xd_o_scaled[:, 0].numpy(),
    #     y=xd_o_scaled[:, 1].numpy(),
    #     ax=ax,
    #     fill=False,
    #     color="gray",
    #     alpha=0.5,
    # )

    ax.set_title(f"Observation {i + 1}")
    ax.set_xlabel(r"$\mathbf{\theta}_1$")
    ax.set_ylabel(r"$\mathbf{\theta}_2$")

    # add the proxy legend
    ax.legend(handles=[obs_patch, gt_patch], loc="upper right")

plt.tight_layout()
plt.savefig(savedir / "fm_on_y_cond_obs.pdf")
# plot_closest_samples(xd_scaled, x_tilde_scaled, 5, 3, model_path)

# flow_matching_conditional_latent = FlowMatching(
#     conditional=True,
#     probability_path="ot",
#     prior="uniform",
#     base_dist="gaussian",
#     # base_dist="none",
#     dim=xd.shape[1],
#     # prior_params={"rate": 2.0},
#     drift={"architecture": "resmlp"},
# )
# flow_matching.to(device)
# model_path = Path("scripts/models")
# model_path.mkdir(parents=True, exist_ok=True)
# noise = torch.randn_like(xd)
# logname = "_pure_gaussian_x_cond_latent"
#
# # assert torch.allclose(noise[:100], flow_matching.sample_source(noise[:100]))
# # print(f"{((noise - flow_matching.sample_source(noise)) ** 2).sum(dim=1).sqrt().mean()}")
#
# retrain = True
# if Path(model_path / f"flow_matching{logname}.pth").exists() and not retrain:
#     flow_matching_conditional_latent.load_state_dict(
#         torch.load(model_path / f"flow_matching{logname}.pth")
#     )
#     flow_matching_conditional_latent.to(device)
# else:
#     flow_matching_conditional_latent = train_flow_matching(
#         flow_matching_conditional_latent,
#         xd,
#         noise,
#         device,
#         None,
#         model_path,
#         logname=logname,
#         epochs=300,
#         batch_size=256,
#         lr=1e-4,
#     )
#
# x_tilde = flow_matching_conditional_latent.sample_batched(noise[:2000], 1, device, only_last=False)
#
# x_tilde_scaled = rescale(x_tilde[:, -1, :], scales["x_mean"], scales["x_std"])
# source_scaled = rescale(x_tilde[:, 0, :], scales["x_mean"], scales["x_std"])
#
# # Convert tensors to DataFrames with labels
# df_samples = pd.DataFrame(x_tilde_scaled.numpy(), columns=["x", "y"])
# df_samples["label"] = "Samples"
#
# df_source = pd.DataFrame(source_scaled.numpy(), columns=["x", "y"])
# df_source["label"] = "Source"
#
# df_data = pd.DataFrame(xd_o_scaled.numpy(), columns=["x", "y"])
# df_data["label"] = "Data"
#
# # Combine all dataframes
# df_all = pd.concat([df_samples, df_source, df_data], ignore_index=True)
# # Use sns.jointplot
# sns.set_theme(style="whitegrid")
# g = sns.jointplot(
#     data=df_all, x="x", y="y", hue="label", kind="scatter", marginal_kws={"fill": False}
# )
# g.figure.suptitle("Joint Scatter and Marginal Histograms")
# plt.savefig(savedir / "fm_on_x_cond_latent.pdf")

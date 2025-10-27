from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flow_matching.torch_flow import FlowMatching, train_flow_matching
from simulator import get_simulator
from simulator.base import generate_dataset
from utils.misc import rescale
from utils.networks import ConditionalGaussianPrior
from utils.plots import pair_plot_tensors, plot_closest_samples

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Found device: {device}")

retrain = False

task_params = {
    "dim": 200,  # Number of time points (N)
    "low_w0": 0.0,  # Lower bound for w
    "high_w0": 3.0,  # Upper bound for w
    "low_A": 0.5,  # Lower bound for A
    "high_A": 10.0,  # Upper bound for A
    "tmax": 10.0,  # Maximum time
}
# Get simulator and generate dataset
simulator = get_simulator("pendulum", **task_params)
num_samples = 50000
thetad, xd, yd, scales = generate_dataset(
    simulator,
    num_samples,
    # rescale="standardize",
    rescale="standardize",
)
thetad = thetad.reshape(-1, thetad.shape[-1])
xd = xd.reshape(-1, xd.shape[-1])
yd = yd.reshape(-1, yd.shape[-1])
num_test = 2000
theta_o, x_o, y_o, scales_o = generate_dataset(
    simulator,
    num_test,
    rescale="standardize",
)
theta_o = theta_o.reshape(-1, theta_o.shape[-1])
x_o = x_o.reshape(-1, x_o.shape[-1])
y_o = y_o.reshape(-1, y_o.shape[-1])

flow_matching = FlowMatching(
    conditional=False,
    probability_path="ot2",
    prior="power",
    base_dist="none",
    # base_dist="none",
    dim=xd.shape[1],
    prior_params={"rate": 2.0},
    # drift={"architecture": "resmlp"},
    drift={
        "architecture": "unet",
    },
)
flow_matching.to(device)
model_path = Path("scripts/models")
model_path.mkdir(parents=True, exist_ok=True)
noise = torch.randn_like(xd)
logname = "_pendulum_x_no_cond"

# assert torch.allclose(noise[:100], flow_matching.sample_source(noise[:100]))
# print(f"{((noise - flow_matching.sample_source(noise)) ** 2).sum(dim=1).sqrt().mean()}")

if Path(model_path / f"flow_matching{logname}.pth").exists() and not retrain:
    flow_matching.load_state_dict(
        torch.load(model_path / f"flow_matching{logname}.pth", map_location=device)
    )
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
        epochs=500,
        batch_size=256,
        lr=1e-4,
    )

source = torch.randn(num_test, xd.shape[1])
x_tilde = flow_matching.sample_batched(source, 1, device, only_last=False)

x_tilde_scaled = rescale(x_tilde[:, -1, :], scales["x_mean"], scales["x_std"])
source_scaled = rescale(x_tilde[:, 0, :], scales["x_mean"], scales["x_std"])
x_o_scaled = rescale(x_o, scales["x_mean"], scales["x_std"])

savedir = Path("figures/scripts/pendulum_baselines")
savedir.mkdir(parents=True, exist_ok=True)
nrows = 5
ncols = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
axes = axes.flatten()
for i, ax in enumerate(axes):
    plt.sca(ax)
    plt.plot(x_o_scaled[i, :], label="Data", color="black")
    plt.plot(x_tilde_scaled[i, :], label="Sampled", color="red")
    plt.plot(source_scaled[i, :], label="Source", color="blue", alpha=0.5)
plt.legend()
plt.savefig(savedir / "fm_on_x_no_cond.pdf")
plt.close()

pair_plot_tensors(
    x_o_scaled, x_tilde_scaled, savedir, scatter_size=2, save=True, name="fm_on_x_no_cond_pairplot"
)

# Conditional
conditional_flow_matching = FlowMatching(
    conditional=True,
    probability_path="ot2",
    prior="power",
    base_dist="gaussian",
    # base_dist="none",
    dim=xd.shape[1],
    prior_params={"rate": 2.0},
    # drift={"architecture": "resmlp"},
    drift={
        "architecture": "unet",
    },
)

logname = "_pendulum_x_cond"
retrain = False

if Path(model_path / f"flow_matching{logname}.pth").exists() and not retrain:
    conditional_flow_matching.load_state_dict(
        torch.load(model_path / f"flow_matching{logname}.pth", map_location=device)
    )
    conditional_flow_matching.to(device)
else:
    conditional_flow_matching = train_flow_matching(
        conditional_flow_matching,
        xd,
        yd,
        device,
        None,
        model_path,
        logname=logname,
        epochs=500,
        batch_size=256,
        lr=1e-4,
    )

cond = y_o
x_tilde_cond = conditional_flow_matching.sample_batched(cond, 1, device, only_last=False)
x_tilde_cond_scaled = rescale(x_tilde_cond[:, -1, :], scales["x_mean"], scales["x_std"])
source_cond_scaled = rescale(x_tilde_cond[:, 0, :], scales["x_mean"], scales["x_std"])
cond_scaled = rescale(cond, scales["y_mean"], scales["y_std"])

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
axes = axes.flatten()
for i, ax in enumerate(axes):
    plt.sca(ax)
    plt.plot(x_o_scaled[i, :], label=r"$x$", color="black")
    plt.plot(x_tilde_cond_scaled[i, :], label=r"$x|y$", color="red")
    plt.plot(source_cond_scaled[i, :], label="Source", color="blue", alpha=0.2)
    plt.plot(cond_scaled[i, :], label=r"$y$", color="green")
plt.legend()
plt.savefig(savedir / "fm_on_x_cond.pdf")
plt.close()

pair_plot_tensors(
    x_o_scaled,
    x_tilde_cond_scaled,
    savedir,
    scatter_size=2,
    save=True,
    name="fm_on_x_cond_pairplot",
)

# Latent x

latent_flow_matching = FlowMatching(
    conditional=True,
    probability_path="ot2",
    prior="power",
    base_dist="gaussian",
    # base_dist="none",
    dim=xd.shape[1],
    prior_params={"rate": 2.0},
    # drift={"architecture": "resmlp"},
    drift={
        "architecture": "unet",
    },
)

logname = "_pendulum_x_cond_latent"
retrain = False

if Path(model_path / f"flow_matching{logname}.pth").exists() and not retrain:
    latent_flow_matching.load_state_dict(torch.load(model_path / f"flow_matching{logname}.pth"))
    latent_flow_matching.to(device)
else:
    latent_flow_matching = train_flow_matching(
        latent_flow_matching,
        xd,
        noise,
        device,
        None,
        model_path,
        logname=logname,
        epochs=500,
        batch_size=256,
        lr=1e-4,
    )
cond = torch.randn_like(y_o)
x_tilde_latent = latent_flow_matching.sample_batched(cond, 1, device, only_last=False)
x_tilde_latent_scaled = rescale(x_tilde_latent[:, -1, :], scales["x_mean"], scales["x_std"])
source_latent_scaled = rescale(x_tilde_latent[:, 0, :], scales["x_mean"], scales["x_std"])

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
axes = axes.flatten()
for i, ax in enumerate(axes):
    plt.sca(ax)
    # plt.plot(xd_scaled[i, :], label="Data", color="black")
    plt.plot(x_tilde_latent_scaled[i, :], label="Sampled", color="red")
    plt.plot(source_latent_scaled[i, :], label="Source", color="blue", alpha=0.2)
plt.legend()
plt.savefig(savedir / "fm_on_x_latent.pdf")
plt.close()
# pair plot
pair_plot_tensors(
    x_o_scaled,
    x_tilde_latent_scaled,
    savedir,
    scatter_size=2,
    save=True,
    name="fm_on_x_latent_pairplot",
)

# plots using ensemble
# cond = torch.randn(num_test, 20, y_o.shape[1])
# source = torch.randn(num_test, xd.shape[1])
# x_tilde_latent_ens = latent_flow_matching.sample_ensemble(
#     source.to(device), cond.to(device), 50, only_last=False
# ).cpu()
# x_tilde_latent_ens_scaled = rescale(x_tilde_latent_ens[:, -1, :], scales["x_mean"], scales["x_std"])
# source_latent_ens_scaled = rescale(x_tilde_latent_ens[:, 0, :], scales["x_mean"], scales["x_std"])
#
# fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     plt.sca(ax)
#     # plt.plot(xd_scaled[i, :], label="Data", color="black")
#     plt.plot(x_tilde_latent_ens_scaled[i, :], label="Sampled", color="red")
#     plt.plot(source_latent_ens_scaled[i, :], label="Source", color="blue", alpha=0.2)
# plt.legend()
# plt.savefig(savedir / "fm_on_x_latent_ens.pdf")
# plt.close()
# # pair plot
# pair_plot_tensors(
#     x_o_scaled,
#     x_tilde_latent_ens_scaled,
#     savedir,
#     scatter_size=2,
#     save=True,
#     name="fm_on_x_latent_ens_pairplot",
# )
# Fine tuning on y
num_cal = 5000
theta_cal, x_cal, y_cal, _ = generate_dataset(
    simulator,
    num_cal,
    rescale="standardize",
    # rescale="whitening",
)
theta_cal = theta_cal.reshape(-1, theta_cal.shape[-1])
x_cal = x_cal.reshape(-1, x_cal.shape[-1])
y_cal = y_cal.reshape(-1, y_cal.shape[-1])
logname = "_pendulum_x_cond_latent_cal"

calibrated_flow_matching = train_flow_matching(
    latent_flow_matching,
    x_cal,
    y_cal,
    device,
    None,
    model_path,
    logname=logname,
    epochs=500,
    batch_size=32,
    lr=1e-4,
)
calibrated_flow_matching.eval()

cond = y_o
x_tilde_cal = calibrated_flow_matching.sample_batched(cond, 1, device, only_last=False)
x_tilde_cal_scaled = rescale(x_tilde_cal[:, -1, :], scales["x_mean"], scales["x_std"])
source_cal_scaled = rescale(x_tilde_cal[:, 0, :], scales["x_mean"], scales["x_std"])
cond_scaled = rescale(cond, scales["y_mean"], scales["y_std"])

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
axes = axes.flatten()

for i, ax in enumerate(axes):
    plt.sca(ax)
    plt.plot(x_o_scaled[i, :], label="Data", color="black")
    plt.plot(x_tilde_cal_scaled[i, :], label="Sampled", color="red")
    plt.plot(source_cal_scaled[i, :], label="Source", color="blue", alpha=0.2)
    plt.plot(cond_scaled[i, :], label="Condition", color="green")
plt.legend()
plt.savefig(savedir / "fm_on_x_cond_latent_cal.pdf")
plt.close()
pair_plot_tensors(
    x_o_scaled,
    x_tilde_cal_scaled,
    savedir,
    scatter_size=2,
    save=True,
    name="fm_on_x_cond_latent_cal_pairplot",
)

# Latent Calibration
model = ConditionalGaussianPrior(y_dim=y_cal.shape[1], z_dim=x_cal.shape[1])
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
calibration_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_cal, y_cal),
    batch_size=32,
    shuffle=True,
)
flow_matching.to(device)
model.train()
for epoch in range(20):
    for x_batch, y_batch in calibration_loader:
        # project your paired x's into z-space:
        with torch.no_grad():
            z_batch = flow_matching.sample_backward(
                x_batch.to(device), torch.randn_like(x_batch).to(device), only_last=True
            )  # θ‐frozen flow encoder

        # compute −log p(z|y)
        nll = -model.log_prob(z_batch.to(device), y_batch.to(device)).mean()

        opt.zero_grad()
        nll.backward()
        opt.step()
model.eval()
flow_matching.eval()
source = model.sample(y_o.to(device))
x_y = flow_matching.sample(source, torch.randn_like(source)).cpu()
x_y_scaled = rescale(x_y[:, -1, :], scales["x_mean"], scales["x_std"]).detach().cpu()
source_scaled = rescale(x_y[:, 0, :], scales["x_mean"], scales["x_std"]).detach().cpu()
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
axes = axes.flatten()

for i, ax in enumerate(axes):
    plt.sca(ax)
    plt.plot(x_o_scaled[i, :], label="Data", color="black")
    plt.plot(x_y_scaled[i, :], label="Sampled", color="red")
    plt.plot(source_scaled[i, :], label="Source", color="blue", alpha=0.2)
    plt.plot(cond_scaled[i, :], label="Condition", color="green")
plt.legend()
plt.savefig(savedir / "fm_on_x_cond_latent_cal_gprior.pdf")
plt.close()
pair_plot_tensors(
    x_o_scaled,
    x_y_scaled,
    savedir,
    scatter_size=2,
    save=True,
    name="fm_on_x_cond_latent_cal_gprior_pairplot",
)
# plot_closest_samples(xd_scaled, x_tilde_scaled, 5, 3, model_path)

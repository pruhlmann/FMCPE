from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import tqdm.auto as tqdm
from lampe.inference import NPE, NPELoss
from lampe.utils import GDStep
from lampe.plots import corner, mark_point, nice_rc
from zuko.distributions import DiagNormal
from zuko.flows import (
    NSF,
    Flow,
    MaskedAutoregressiveTransform,
    UnconditionalDistribution,
)
from zuko.flows.neural import UMNN

from simulator.pendulum import Pendulum
from utils.networks import ConvNN1D


class UMNNFlow(Flow):
    def __init__(self, features, context, **kwargs):
        integrand_params = kwargs.get("integrand_params", {})
        conditoner_params = kwargs.get("conditioner_params", {})
        embedding_dim = kwargs.get("embedding_dim", 16)
        transforms = kwargs.get("ntransform", 3)
        neural_nets = [
            UMNN(embedding_dim, **integrand_params) for _ in range(transforms)
        ]

        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]
        orders = list(map(torch.LongTensor, orders))
        transform = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=orders[i % 2],
                univariate=neural_nets[i],
                shapes=((embedding_dim,), ()),
                **conditoner_params,
            )
            for i in range(transforms)
        ]
        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )
        super().__init__(transform, base)
        pass


class EstimatorUMNN(torch.nn.Module):
    def __init__(self, features, context, **kwargs):
        super().__init__()
        self.npe = NPE(
            features,
            context,
            build=lambda f, c: UMNNFlow(f, c, **kwargs),
        )
        self.embedding = ConvNN1D(output_dim=10)

    def forward(self, theta, x):
        return self.npe(theta, self.embedding(x))

    def flow(self, x):
        return self.npe.flow(self.embedding(x))


def two_moons(n: int, sigma: float = 1e-1, device=None):
    theta = 2 * torch.pi * torch.rand(n)
    label = (theta > torch.pi).float()

    x = torch.stack(
        [
            torch.cos(theta) + label - 1 / 2,
            torch.sin(theta) + label / 2 - 1 / 4,
        ],
        dim=-1,
    )
    return torch.normal(x, sigma).to(device), label.to(device)


def show_data(x, y):
    plt.scatter(x[y == 0, 0], x[y == 0, 1], color="blue")
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color="red")
    plt.axis("equal")
    plt.show()


def train_flows(
    data_dim,
    context_dim,
    epochs,
    train_loader,
    lr,
    **kwargs,
) -> Tuple[EstimatorUMNN, NPE, List[float], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = EstimatorUMNN(data_dim, context_dim, **kwargs).to(device)
    loss = NPELoss(estimator)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=1.0)

    losses = []
    for epoch in (
        trange := tqdm.tqdm(range(epochs), desc="Epochs", total=epochs)
    ):
        losse_at_epoch = []

        for theta, x in train_loader:
            losse_at_epoch.append(step(loss(theta, x)))
        losse_at_epoch = torch.stack(losse_at_epoch)
        losses.append(losse_at_epoch.mean().item())
        trange.set_postfix(
            loss=losse_at_epoch.mean().item(),
        )

    return estimator, estimator, losses, losses


def run_pendulum(nsamples, embedding_dim, ntransform, epochs, batch_size, lr):
    save_dir = Path("scripts/results") / "pendulum"
    save_dir.mkdir(parents=True, exist_ok=True)
    import jax

    from simulator import generate_dataset

    key = jax.random.PRNGKey(42)
    key, thkey = jax.random.split(key)
    simulator = Pendulum()
    theta, x, y, _ = generate_dataset(
        simulator, nsamples, thkey, return_type="array"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = torch.tensor(theta).to(device)
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)

    data_dim = simulator.theta_dim
    context_dim = simulator.obs_dim
    integrand_params = {
        "hidden_features": (100, 100, 100),
        "activation": torch.nn.ReLU,
    }
    conditioner_params = {
        "hidden_features": (100, 100, 100),
        "activation": torch.nn.ReLU,
    }
    params = {
        "integrand_params": integrand_params,
        "conditioner_params": conditioner_params,
        "embedding_dim": embedding_dim,
        "ntransform": ntransform,
    }

    train_dataset = data.TensorDataset(theta, x)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    estimator, estimator_nsf, losses, losses_nsf = train_flows(
        data_dim,
        10,
        epochs,
        train_loader,
        lr,
        **params,
    )
    estimator.eval()
    estimator_nsf.eval()
    key, thkey, xkey = jax.random.split(key, 3)
    theta_star = simulator.sample_prior(1, thkey)
    x_star = simulator.get_simulator(misspecified=True)(theta_star, key)
    theta_star = torch.tensor(np.array(theta_star)).to(device)
    x_star = torch.tensor(np.array(x_star)).to(device)

    with torch.no_grad():
        samples = estimator.flow(x_star).sample((10000,)).cpu().detach().numpy()
        samples_nsf = (
            estimator_nsf.flow(x_star).sample((10000,)).cpu().detach().numpy()
        )
    print(samples.shape)
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), losses, label="UMNN")
    plt.plot(range(epochs), losses_nsf, label="NSF")
    plt.legend()
    plt.savefig(save_dir / "umnn_flow_loss.pdf")

    plt.rcParams.update(nice_rc(latex=True))

    fig = corner(
        samples.squeeze(),
        smooth=2,
        domain=(
            torch.tensor([simulator.low_w0, simulator.low_A]),
            torch.tensor([simulator.high_w0, simulator.high_A]),
        ),
        labels=[r"$\omega_0$", r"$A$"],
        legend=r"$p_{\phi}(\theta|x^*)$",
        figsize=(6, 6),
    )

    mark_point(fig, theta_star.squeeze().cpu().numpy())
    plt.savefig(save_dir / "umnn_flow.pdf")

    # plt.figure(figsize=(6, 12))
    # plt.subplot(211)
    # plt.hist2d(
    #     *samples.T,
    #     bins=64,
    #     range=(
    #         (simulator.low_w0, simulator.high_w0),
    #         (simulator.low_A, simulator.high_A),
    #     ),
    # )
    # plt.title("UMNN")
    # plt.subplot(212)
    # plt.hist2d(
    #     *samples_nsf.T,
    #     bins=64,
    #     range=(
    #         (simulator.low_w0, simulator.high_w0),
    #         (simulator.low_A, simulator.high_A),
    #     ),
    # )
    # plt.title("NSF")
    # plt.suptitle(r"$\theta^*=$" + f"{theta[0]}")
    # plt.savefig(save_dir / "umnn_flow.pdf")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--task", type=str, default="two_moons")
    argparse.add_argument("--nsamples", type=int, default=20000)
    argparse.add_argument("--embedding_dim", type=int, default=16)
    argparse.add_argument("--ntransform", type=int, default=3)
    argparse.add_argument("--epochs", type=int, default=50)
    argparse.add_argument("--batch_size", type=int, default=1024)
    argparse.add_argument("--lr", type=float, default=1e-3)
    args = argparse.parse_args()

    if args.task == "two_moons":
        run_two_moons(
            nsamples=args.nsamples,
            embedding_dim=args.embedding_dim,
            ntransform=args.ntransform,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.task == "pendulum":
        run_pendulum(
            nsamples=args.nsamples,
            embedding_dim=args.embedding_dim,
            ntransform=args.ntransform,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

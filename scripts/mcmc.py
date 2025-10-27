from argparse import ArgumentParser
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_value
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set random seed for reproducibility
pyro.set_rng_seed(42)

# Task parameters
task_params = {
    "dim": 200,  # Number of time points (N)
    "low_w0": 0.0,  # Lower bound for w
    "high_w0": 3.0,  # Upper bound for w
    "low_A": 0.5,  # Lower bound for A
    "high_A": 10.0,  # Upper bound for A
    "tmax": 10.0,  # Maximum time
}

# Extract parameters
N = task_params["dim"]
max_t = task_params["tmax"]
sigma_x = 1.0  # Noise standard deviation
sigma_y = 0.1  # Noise standard deviation

# Time grid
t = torch.linspace(0, max_t, N)


def generate_data(M):
    # true_A = torch.empty(M).uniform_(task_params["low_A"], task_params["high_A"])
    # true_w = torch.empty(M).uniform_(task_params["low_w0"], task_params["high_w0"])
    true_A = torch.tensor([3.0, 5.0, 7.0])
    true_w = torch.tensor([1.0, 1.5, 2.0])
    true_phi = torch.empty(M).uniform_(0, 2 * torch.pi)
    true_alpha = torch.empty(M).uniform_(0, 1)

    x = torch.zeros(M, N)
    y = torch.zeros(M, N)
    for i in range(M):
        epsilon_x = torch.normal(0, sigma_x, size=(N,))
        epsilon_y = torch.normal(0, sigma_y, size=(N,))
        x[i] = true_A[i] * torch.cos(true_w[i] * t + true_phi[i]) + epsilon_x
        y[i] = torch.exp(-true_alpha[i] * t) * x[i] + epsilon_y
    return x, y, true_A, true_w, true_phi, true_alpha


# Define the generative model in Pyro
def model(x_observed=None, y_observed=None):
    A = pyro.sample("A", dist.Uniform(task_params["low_A"], task_params["high_A"]))
    w = pyro.sample("w", dist.Uniform(task_params["low_w0"], task_params["high_w0"]))
    phi = pyro.sample("phi", dist.Uniform(0, 2 * torch.pi))

    mean = A * torch.cos(w * t + phi)
    alpha = pyro.sample("alpha", dist.Uniform(0, 1))

    x = pyro.sample("x", dist.Normal(mean, sigma_x), obs=x_observed)
    y = pyro.sample("y", dist.Normal(x * torch.exp(-alpha * t), sigma_y), obs=y_observed)


def plot_data(M, A_list_x, w_list_x, A_list_y, w_list_y, true_A, true_w):
    # Create grid plot
    plt.figure(figsize=(5 * M, 5))  # Adjusted height for 2 rows

    # First row: 2D KDE of p(A, w | x)
    for i in range(M):
        plt.subplot(1, M, i + 1)
        sns.kdeplot(
            x=A_list_x[i].numpy().flatten(),
            y=w_list_x[i].numpy().flatten(),
            fill=False,  # Filled contours (use 'shade=True' for older Seaborn versions)
            cmap="Reds",
            thres=0.10,
        )
        sns.kdeplot(
            x=A_list_y[i].numpy().flatten(),
            y=w_list_y[i].numpy().flatten(),
            fill=False,  # Filled contours (use 'shade=True' for older Seaborn versions)
            cmap="Greens",
            thres=0.10,
        )
        plt.scatter(
            [true_A[i]], [true_w[i]], color="blue", marker="o", label="True (A, w)", s=30, zorder=2
        )
        # plt.scatter([init_A.item()], [init_w.item()], color="g", marker="o", label="Init (A, w)", s=100)
        plt.plot([], [], c="red", lw=2.0, label=r"$p(\theta \mid x)$")
        plt.plot([], [], c="green", lw=2.0, label=r"$p(\theta \mid y)$")
        plt.xlabel("A")
        plt.ylabel("w")
        plt.legend()
    plt.suptitle("2D KDE of p(A, w | x)")

    plt.tight_layout()
    plt.savefig("mcmc_example.pdf")


def main(M, num_samples=1000, warmup_steps=1000):
    # Generate synthetic data
    x_observed, y_observed, true_A, true_w, true_phi, true_alpha = generate_data(M)
    A_list_x = []
    A_list_y = []
    w_list_x = []
    w_list_y = []
    for i in range(M):
        print(f"Running MCMC for example {i + 1}")
        # init_A = torch.clamp(
        #     true_A[i] + torch.randn(1) * 0.01, task_params["low_A"], task_params["high_A"]
        # )
        # init_w = torch.clamp(
        #     true_w[i] + torch.randn(1) * 0.01, task_params["low_w0"], task_params["high_w0"]
        # )
        init_A = true_A[i]
        init_w = true_w[i]
        # Run MCMC for x
        nuts_kernel = NUTS(
            model,
            init_strategy=init_to_value(None, values={"A": init_A, "w": init_w}),
        )
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
        )
        mcmc.run(x_observed=x_observed[i], y_observed=None)

        # Extract posterior samples
        posterior_samples = mcmc.get_samples()

        A_list_x.append(posterior_samples["A"])
        w_list_x.append(posterior_samples["w"])

        # Run MCMC for y
        nuts_kernel = NUTS(
            model,
            init_strategy=init_to_value(None, values={"A": init_A, "w": init_w}),
        )
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
        )
        mcmc.run(x_observed=None, y_observed=y_observed[i])

        # Extract posterior samples
        posterior_samples = mcmc.get_samples()

        A_list_y.append(posterior_samples["A"])
        w_list_y.append(posterior_samples["w"])

    plot_data(M, A_list_x, w_list_x, A_list_y, w_list_y, true_A, true_w)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--M", type=int, default=3, help="Number of examples")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    args = parser.parse_args()
    main(args.M, args.num_samples, args.warmup_steps)

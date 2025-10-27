import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from multiprocessing import Pool
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
    "tmax": 1.0,  # Maximum time
}

# Extract parameters
N = task_params["dim"]
max_t = task_params["tmax"]
sigma = 0.1  # Noise standard deviation
M = 3  # Number of independent observations

# Time grid
t = torch.linspace(0, max_t, N)

# Simulate multiple independent observations
true_A = torch.tensor([1.5, 2.0, 1.0])  # True amplitudes
true_w = torch.tensor([2.0, 1.5, 2.5])  # True frequencies
true_phi = torch.tensor([1.2, 0.5, 2.0])  # True phases


def generate_data():
    x = torch.zeros(M, N)
    for i in range(M):
        epsilon = torch.normal(0, sigma, size=(N,))
        x[i] = true_A[i] * torch.cos(2 * torch.pi * true_w[i] * t + true_phi[i]) + epsilon
    return x


x_observed = generate_data()

# Define good starting points
init_A = torch.tensor([1.4, 1.9, 1.1])
init_w = torch.tensor([2.1, 1.6, 2.4])
init_phi = torch.tensor([1.0, 0.6, 1.8])


# Model for a single observation
def model_single(x_observed):
    A = pyro.sample("A", dist.Uniform(task_params["low_A"], task_params["high_A"]))
    w = pyro.sample("w", dist.Uniform(task_params["low_w0"], task_params["high_w0"]))
    phi = pyro.sample("phi", dist.Uniform(0, 2 * torch.pi))

    mean = A * torch.cos(2 * torch.pi * w * t + phi)

    with pyro.plate("data", N, dim=-1):
        pyro.sample("x", dist.Normal(mean, sigma), obs=x_observed)


# Function to run MCMC for one observation
def run_mcmc_for_observation(args):
    i, x_obs, init_a, init_w, init_phi = args
    pyro.set_rng_seed(42 + i)  # Different seed per process
    nuts_kernel = NUTS(model_single)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=500,
        warmup_steps=200,
        initial_params={
            "A": torch.tensor([init_a]),
            "w": torch.tensor([init_w]),
            "phi": torch.tensor([init_phi]),
        },
    )
    mcmc.run(x_obs)
    samples = mcmc.get_samples()
    return {k: v.numpy() for k, v in samples.items()}  # Convert to numpy for multiprocessing


# Parallelize MCMC across observations
if __name__ == "__main__":
    # Prepare arguments for each observation
    args_list = [(i, x_observed[i], init_A[i], init_w[i], init_phi[i]) for i in range(M)]

    # Run in parallel
    with Pool(processes=M) as pool:
        results = pool.map(run_mcmc_for_observation, args_list)

    # Extract posterior samples for all observations
    posterior_samples = {
        "A": np.stack([res["A"] for res in results], axis=1),  # Shape: (num_samples, M)
        "w": np.stack([res["w"] for res in results], axis=1),
        "phi": np.stack([res["phi"] for res in results], axis=1),
    }

    # Compute mean estimates
    A_mean = torch.tensor(posterior_samples["A"].mean(axis=0))
    w_mean = torch.tensor(posterior_samples["w"].mean(axis=0))
    phi_mean = torch.tensor(posterior_samples["phi"].mean(axis=0))

    # Generate true and estimated signals for the first observation (i=0)
    i = 0
    true_signal = true_A[i] * torch.cos(2 * torch.pi * true_w[i] * t + true_phi[i])
    estimated_signal = A_mean[i] * torch.cos(2 * torch.pi * w_mean[i] * t + phi_mean[i])

    # Create grid plot
    plt.figure(figsize=(12, 8))

    # Posterior of A for i=0
    plt.subplot2grid((2, 2), (0, 0))
    plt.hist(posterior_samples["A"][:, i], bins=30, density=True)
    plt.axvline(true_A[i], color="r", linestyle="--", label=f"True A[{i}]")
    plt.axvline(init_A[i], color="g", linestyle="--", label=f"Init A[{i}]")
    plt.title(f"Posterior of A[{i}]")
    plt.xlabel(f"A[{i}]")
    plt.legend()

    # Posterior of w for i=0
    plt.subplot2grid((2, 2), (0, 1))
    plt.hist(posterior_samples["w"][:, i], bins=30, density=True)
    plt.axvline(true_w[i], color="r", linestyle="--", label=f"True w[{i}]")
    plt.axvline(init_w[i], color="g", linestyle="--", label=f"Init w[{i}]")
    plt.title(f"Posterior of w[{i}]")
    plt.xlabel(f"w[{i}]")
    plt.legend()

    # True vs. Estimated Signal for i=0
    plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plt.plot(t.numpy(), true_signal.numpy(), label="True Signal", color="b")
    plt.plot(
        t.numpy(),
        estimated_signal.numpy(),
        label="Estimated Signal (mean)",
        color="r",
        linestyle="--",
    )
    plt.plot(t.numpy(), x_observed[i].numpy(), label="Observed Data", alpha=0.3, color="gray")
    plt.title(f"True vs. Estimated Signal (Observation {i})")
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.legend()

    plt.tight_layout()
    plt.show()

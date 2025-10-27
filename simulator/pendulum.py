from scipy.signal import correlate
import numpy as np
import torch
from pyro import distributions as dist
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist
from numpyro.infer import MCMC, NUTS
from simulator.base import Simulator
from utils.transform import LogitBoxTransform


class Pendulum(Simulator):
    def __init__(
        self,
        obs_dim: int = 200,
        noise: float = 0.1,
        low_w0: float = 0.0,
        high_w0: float = 3.0,
        low_A: float = 0.5,
        high_A: float = 10.0,
        tmax: float = 10.0,
    ):
        obs_dim = int(obs_dim)
        super().__init__(obs_dim=obs_dim, theta_dim=2, name="pendulum")
        self.obs_dim = obs_dim
        self.noise = noise
        self.low_w0 = low_w0
        self.high_w0 = high_w0
        self.low_A = low_A
        self.high_A = high_A
        self.tmax = tmax

        self.callable_simulator = True
        self.callable_dgp = True
        self.supported_generation = ["independent", "transitive"]

        # prior distribution
        self.prior_params = {
            "low": torch.Tensor([self.low_w0, self.low_A]),
            "high": torch.Tensor([self.high_w0, self.high_A]),
        }
        self.prior_dist = dist.Independent(dist.Uniform(**self.prior_params), 1)
        # self.prior_dist = dist.Normal(loc=torch.Tensor([1.5, 5.0]), scale=torch.Tensor([0.5, 2.5]))
        self.prior_dist.set_default_validate_args(False)

        # Transform for theta
        self.transform = LogitBoxTransform(a=self.prior_params["low"], b=self.prior_params["high"])

        self.like_dist = None
        self.post_dist = None
        self.denoise_dist = None

    def get_simulator(self, misspecified: bool):
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            phi = torch.rand(theta.shape[0], 1) * 2 * torch.pi
            wo, A = theta[:, 0].reshape(-1, 1), theta[:, 1].reshape(-1, 1)
            t = torch.linspace(0, self.tmax, self.obs_dim).reshape(1, -1)
            alpha = (1 - misspecified) * torch.rand(theta.shape[0], 1)
            x = torch.exp(-alpha @ t) * A * torch.cos(wo @ t + phi) + self.noise * torch.randn(
                theta.shape[0], self.obs_dim
            )
            return x

        return simulator

    def sample_reference_posterior(
        self,
        num_samples: int,
        observation: torch.Tensor,
        misspecified: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_noisy_process(self):
        def noisy_process(x: torch.Tensor) -> torch.Tensor:
            t = torch.linspace(0, self.tmax, self.obs_dim).reshape(1, -1)
            alpha = torch.rand(x.shape[0], 1)
            y = torch.exp(-alpha @ t) * x + 0.1 * self.noise * torch.randn(x.shape[0], self.obs_dim)
            return y

        return noisy_process

    def sample_denoiser(self, num_samples: int, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

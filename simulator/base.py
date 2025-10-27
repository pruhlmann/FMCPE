from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from pyro.distributions import Distribution
from torch import Tensor
from torch.distributions.transforms import Transform

from utils.transform import IdentityTransform


class Simulator:
    def __init__(
        self,
        obs_dim: int | Tuple[int, int, int],
        theta_dim: int,
        name: str,
    ):
        """Simulator base class representing a abstract simulator which can be used to generate samples using a given prior.

        Args:
            obs_dim: Dimension of the simulator output
            theta_dim: Dimension of the simulator input (prior)
            name: Name of the simulator
        """
        self.obs_dim = obs_dim
        self.theta_dim = theta_dim
        self.name = name
        self.prior_params: Dict[str, Tensor] = {}
        self.transform: Transform = IdentityTransform()

        self.prior_dist: ...
        self.like_dist: ...
        self.post_dist: ...
        self.denoise_dist: ...
        self.supported_generation: List[str]
        self.callable_simulator: bool
        self.callable_dgp: bool

    def get_prior_dist(self) -> Distribution:
        """Return the prior distribution of the simulator.
        Returns:
            Prior distribution
        """
        return self.prior_dist

    def sample_prior(self, n: int) -> Tensor:
        """Sample from the prior distribution.

        Returns:
            Samples from the prior distribution
        """
        return self.prior_dist.sample(sample_shape=(n,))

    @abstractmethod
    def get_simulator(self, misspecified: bool, **kwargs) -> Callable[[Tensor], Tensor]:
        """Return the simulator function of the simulator as a function of the parameters (prior samples).

        Args:
            misspecified: If True, return the misspecified simulator (x) else return the true data generating process (y)

        Returns:
            Simulator function
        """
        pass

    @abstractmethod
    def simulations_from_files(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the simulator output from files.
        Args:
            n: Number of samples to generate
        Returns:
            Tuple of prior samples and simulator output
        """
        pass

    @abstractmethod
    def obs_from_files(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the observations from files.
        Args:
            n: Number of samples to generate
        Returns:
            Tuple of prior samples and observations
        """
        pass

    @abstractmethod
    def dataset_from_files(self, n: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Load the dataset from files.
        Args:
            n: Number of samples to generate
        Returns:
            Tuple of prior samples, simulator output and observations
        """
        pass

    def get_like_dist(self, theta: Tensor, misspecified: bool) -> Distribution:
        """Return the likelihood distribution of the simulator.
        Returns:
            Likelihood distribution
        """
        return self.like_dist(theta, misspecified)

    def get_posterior_dist(self, x_y: Tensor, misspecified: bool) -> Distribution:
        """Return the posterior distribution of the simulator.
        Args:
            x_y: Data
            misspecified: Misspecified simulator or true data generating process
        Returns:
            Posterior distribution
        """
        return self.post_dist(x_y, misspecified)

    @abstractmethod
    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: Tensor,
        misspecified: bool,
        **kwargs,
    ) -> Tensor:
        """Sample from the true posterior distribution of the simulator given the data x.

        Args:
            num_samples: Number of samples to generate
            observations: Observations
            misspecified: Misspecified simulator or true data generating process

        Returns:
            Samples from the true posterior distribution
        """
        pass

    @abstractmethod
    def get_noisy_process(self) -> Callable[[Tensor], Tensor]:
        """Return the observation from the simulation.
        Args:
        Returns:
            noisy process
        """
        pass

    def get_denoise_dist(self, y: Tensor) -> Distribution:
        """Return the denoising distribution of the simulator.
        Returns:
            Denoising distribution
        """
        return self.denoise_dist(y)

    @abstractmethod
    def sample_denoiser(self, num_samples: int, y: Tensor) -> Tensor:
        """Sample from the denoising distribution of the simulator given the data y.
        Args:
            num_samples: Number of samples to generate
            y: Observations
        Returns:
            Samples from the denoising distribution
        """
        pass


def rescale(theta, x, y):
    theta_mean, theta_std = theta.mean(), theta.std()
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()

    scales = {
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    theta = (theta - theta_mean) / theta_std
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std

    return theta, x, y, scales


def generate_simulation_dataset(
    simulator: Simulator,
    n: int,
) -> Tuple[Tensor, Tensor]:
    """Generate a dataset using a given simulator. The dataset consists of
    samples from the prior, the simulator output for the samples from the prior

    Args:
        simulator: Simulator object
        n: Number of samples to generate
        rescale: Whether to rescale the data
        generation: Generation method, either "independent" or "transitive"
        augment: Whether to augment the data

    Returns:
        Tuple of prior samples, simulator output and real data and a dictionary of scales
    """

    if simulator.callable_simulator:
        # If the simulator is callable, we can directly call it with the prior samples
        theta = simulator.sample_prior(n)
        x = simulator.get_simulator(misspecified=True)(theta)
    else:
        theta, x = simulator.simulations_from_files(n)
    theta = simulator.transform(theta)  # E.g. from [a,b] to R
    return theta, x


def generate_calibration_dataset(
    simulator: Simulator,
    n: int,
    generation: Optional[str] = "independent",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate a dataset using a given simulator. The dataset consists of
    samples from the prior, the simulator output for the samples from the prior and the real data.

    Args:
        simulator: Simulator object
        n: Number of samples to generate
        rescale: Whether to rescale the data
        generation: Generation method, either "independent" or "transitive"
        augment: Whether to augment the data

    Returns:
        Tuple of prior samples, simulator output and real data and a dictionary of scales
    """

    assert generation in simulator.supported_generation, (
        "Unsupported generation method: ",
        generation,
        ". Supported methods: ",
        simulator.supported_generation,
    )
    if generation == "independent":
        if simulator.callable_simulator and simulator.callable_dgp:
            theta = simulator.sample_prior(n)
            x = simulator.get_simulator(misspecified=True)(theta)
            y = simulator.get_simulator(misspecified=False)(theta)
        elif simulator.callable_simulator and not simulator.callable_dgp:
            theta, y = simulator.obs_from_files(n)
            x = simulator.get_simulator(misspecified=True)(theta)
        elif not simulator.callable_simulator and simulator.callable_dgp:
            theta, x = simulator.simulations_from_files(n)
            y = simulator.get_simulator(misspecified=False)(theta)
        elif not simulator.callable_simulator and not simulator.callable_dgp:
            theta, x, y = simulator.obs_from_files(n)
    elif generation == "transitive":
        assert simulator.callable_simulator, "Transitive generation requires a callable simulator."
        theta = simulator.sample_prior(n)
        if simulator.name == "gaussian":
            x, x_raw = simulator.get_simulator(misspecified=True, return_raw=True)(theta)
            y = simulator.get_noisy_process()(x_raw)
        else:
            x = simulator.get_simulator(misspecified=True)(theta)
            y = simulator.get_noisy_process()(x)

    theta: torch.Tensor = simulator.transform(theta)  # E.g. from [a,b] to R
    return theta, x, y

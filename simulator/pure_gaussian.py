import pyro.distributions as dist
import torch
from torch import Tensor

from simulator.base import Simulator


class PureGaussian(Simulator):
    def __init__(
        self,
        theta_dim: int = 2,
        obs_dim: int = 2,
        prior_var_scale: float = 5.0,
        likelihood_var_scale: float = 2.0,
        noisy_var_scale: float = 5.0,
        seed: int = 0,
    ):
        obs_dim = int(obs_dim)
        theta_dim = int(theta_dim)
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="pure_gaussian")
        self.callable_simulator = True
        self.callable_dgp = True
        self.supported_generation = ["independent", "transitive"]

        # Prior parameters
        torch.manual_seed(seed)
        prior_loc = torch.rand((theta_dim,)) * 10 - 5
        cov_theta_sqrt = torch.normal(0.0, prior_var_scale, (theta_dim, theta_dim))
        prior_cov = cov_theta_sqrt @ cov_theta_sqrt.T + torch.eye(theta_dim)

        self.prior_params = {"loc": prior_loc, "covariance_matrix": prior_cov}

        self.prior_dist = dist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

        # Likelihood parameters
        cov_likelihood_sqrt = torch.normal(0.0, likelihood_var_scale, (obs_dim, obs_dim))
        A = torch.normal(0.0, 1.0, (obs_dim, theta_dim))
        b = torch.normal(0.0, 1.0, (obs_dim,))
        likelihood_cov = cov_likelihood_sqrt @ cov_likelihood_sqrt.T

        def mean_likelihood(theta):
            return theta @ A.T + b

        self.mean_likelihood = mean_likelihood
        self.simulator_params = {
            "coef": A,
            "bias": b,
            "covariance_matrix": likelihood_cov,
        }

        self.likelihood_dist = lambda theta: dist.MultivariateNormal(
            loc=mean_likelihood(theta), covariance_matrix=likelihood_cov
        )

        # Noisy process parameters
        cov_noisy_sqrt = torch.normal(0.0, noisy_var_scale, (obs_dim, obs_dim))
        C = torch.normal(1.0, 1.0, (obs_dim, obs_dim))
        d = torch.rand((obs_dim,)) * 5 + 5
        noise_cov = cov_noisy_sqrt @ cov_noisy_sqrt.T

        def mean_noise(x):
            return x @ C.T + d

        self.mean_noise = mean_noise
        self.noise_params = {
            "coef": C,
            "bias": d,
            "covariance_matrix": noise_cov,
        }

    def get_simulator(self, misspecified: bool):
        def simulator(
            theta: Tensor,
        ) -> Tensor:
            parameters = {
                "loc": self.mean_likelihood(theta),
                "covariance_matrix": self.simulator_params["covariance_matrix"],
            }
            if not misspecified:
                x = dist.MultivariateNormal(**parameters).sample()
                noise_parameters = {
                    "loc": self.mean_noise(x),
                    "covariance_matrix": self.noise_params["covariance_matrix"],
                }
                noise = dist.MultivariateNormal(**noise_parameters).sample()
                return noise
            else:
                return dist.MultivariateNormal(**parameters).sample()

        return simulator

    def denoise_dist(self, y: Tensor) -> dist.MultivariateNormal:
        """
        Given a batch of noisy observations y of shape (batch_size, obs_dim),
        return the batch of conditional distributions p(x | y).
        """
        # Prior over theta
        mu_theta = self.prior_params["loc"]
        Sigma_theta = self.prior_params["covariance_matrix"]

        # Likelihood parameters
        A = self.simulator_params["coef"]
        b = self.simulator_params["bias"]
        Sigma_lik = self.simulator_params["covariance_matrix"]

        # Noise parameters
        C = self.noise_params["coef"]
        d = self.noise_params["bias"]
        Sigma_noise = self.noise_params["covariance_matrix"]

        # Mean and covariance of x
        mu_x = A @ mu_theta + b
        Sigma_x = A @ Sigma_theta @ A.T + Sigma_lik

        # Covariance between x and y
        Sigma_xy = Sigma_x @ C.T

        # Covariance of y
        Sigma_y = C @ Sigma_x @ C.T + Sigma_noise
        Sigma_y_inv = torch.linalg.inv(Sigma_y)

        # Mean of y
        mu_y = C @ mu_x + d

        # Centered y: shape (batch_size, obs_dim)
        y_centered = y - mu_y

        # Compute conditional mean: μ_x|y = μ_x + Σ_xy Σ_yy^{-1} (y - μ_y)
        correction = y_centered @ Sigma_y_inv.T @ Sigma_xy.T
        mu_x_given_y = mu_x + correction  # (batch_size, x_dim)

        # Conditional covariance
        Sigma_x_given_y = Sigma_x - Sigma_xy @ Sigma_y_inv @ Sigma_xy.T

        return dist.MultivariateNormal(
            loc=mu_x_given_y, covariance_matrix=Sigma_x_given_y.expand(y.shape[0], -1, -1)
        )

    def posterior_theta_given_x(self, x: Tensor) -> dist.MultivariateNormal:
        """
        Given a batch of clean observations x of shape (batch_size, obs_dim),
        return the batch of posterior distributions p(theta | x).
        """
        # Prior
        mu_theta = self.prior_params["loc"]  # (theta_dim,)
        Sigma_theta = self.prior_params["covariance_matrix"]  # (theta_dim, theta_dim)

        # Likelihood
        A = self.simulator_params["coef"]  # (obs_dim, theta_dim)
        b = self.simulator_params["bias"]  # (obs_dim,)
        Sigma_lik = self.simulator_params["covariance_matrix"]  # (obs_dim, obs_dim)

        # Covariance between theta and x
        Sigma_thetay = Sigma_theta @ A.T

        # Covariance of x
        Sigma_x = A @ Sigma_theta @ A.T + Sigma_lik
        Sigma_x_inv = torch.linalg.inv(Sigma_x)

        # Mean of x
        mu_x = A @ mu_theta + b

        # Centered y: shape (batch_size, obs_dim)
        x_centered = x - mu_x

        # Compute conditional mean: μ_x|y = μ_x + Σ_xy Σ_yy^{-1} (y - μ_y)
        correction = x_centered @ Sigma_x_inv.T @ Sigma_thetay.T
        mu_theta_given_x = mu_theta + correction  # (batch_size, x_dim)

        # Conditional covariance
        Sigma_theta_given_x = Sigma_theta - Sigma_thetay @ Sigma_x_inv @ Sigma_thetay.T

        return dist.MultivariateNormal(
            loc=mu_theta_given_x, covariance_matrix=Sigma_theta_given_x.expand(x.shape[0], -1, -1)
        )

    def posterior_theta_given_y(self, y: Tensor) -> dist.MultivariateNormal:
        """
        Given a batch of noisy observations y of shape (batch_size, obs_dim),
        return the batch of posterior distributions p(theta | y).
        """
        # Prior
        mu_theta = self.prior_params["loc"]
        Sigma_theta = self.prior_params["covariance_matrix"]

        # Likelihood
        A = self.simulator_params["coef"]
        b = self.simulator_params["bias"]
        Sigma_lik = self.simulator_params["covariance_matrix"]

        # Noise
        C = self.noise_params["coef"]
        d = self.noise_params["bias"]
        Sigma_noise = self.noise_params["covariance_matrix"]

        # Linear mapping from theta to y
        F = C @ A
        c = C @ b + d
        Sigma_y = C @ Sigma_lik @ C.T + Sigma_noise

        # Inverses
        Sigma_y_inv = torch.linalg.inv(Sigma_y)
        Sigma_theta_inv = torch.linalg.inv(Sigma_theta)

        # Posterior covariance
        Sigma_post = torch.linalg.inv(Sigma_theta_inv + F.T @ Sigma_y_inv @ F)

        # Center y
        y_centered = y - c
        mean_post = (
            Sigma_post @ (Sigma_theta_inv @ mu_theta + y_centered @ (Sigma_y_inv @ F)).T
        ).T  # shape (batch_size, theta_dim)

        return dist.MultivariateNormal(
            loc=mean_post, covariance_matrix=Sigma_post.expand(y.shape[0], -1, -1)
        )

    def post_dist(self, x_y: Tensor, misspecified: bool) -> dist.MultivariateNormal:
        if misspecified:
            return self.posterior_theta_given_x(x_y)
        else:
            return self.posterior_theta_given_y(x_y)

    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: Tensor,
        misspecified: bool,
    ) -> Tensor:
        return self.post_dist(observations, misspecified).sample((num_samples,))

    def get_noisy_process(self):
        def noisy_process(x: Tensor) -> Tensor:
            parameters = {
                "loc": self.mean_noise(x),
                "covariance_matrix": self.noise_params["covariance_matrix"],
            }
            return dist.MultivariateNormal(**parameters).sample()

        return noisy_process

    def sample_denoiser(self, num_samples: int, y: Tensor) -> Tensor:
        raise NotImplementedError

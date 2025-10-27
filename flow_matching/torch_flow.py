from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
from numpy.testing import assert_
import torch
import torch.nn as nn
from mlxp.logger import Logger
from torch import Tensor, mean
from torch.distributions import (
    Beta,
    Distribution,
    MultivariateNormal,
    Normal,
    Uniform,
)
from tqdm import trange

from utils.misc import loader_from_tensor, train_val_split
from utils.networks import ResMLP, create_cf

SUPPORTED_ARCHITECTURES = ["resmlp", "cfnet"]


class FlowMatching(nn.Module):
    """Flow matching model."""

    def __init__(
        self,
        conditional: bool,
        probability_path: str,
        prior: str,
        base_dist: str,
        dim: torch.Size,
        cond_dim: Optional[torch.Size] = None,
        **kwargs,
    ) -> None:
        """
        Initialize FlowMatching model.

        Args:
            conditional (bool): Whether the model is conditional.
            probability_path (str): Probability path used to interpolate between source and target.
            prior (str): Prior distribution for time.
            base_dist (str): Base distribution for the model.
            dim (int): Dimension of the target space.
            **kwargs: Additional arguments for the model.
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim if cond_dim else dim
        self._check_args(conditional, probability_path, prior, base_dist, **kwargs)
        self.conditional = conditional
        self.num_steps: int = kwargs.get("num_steps", 50)  # WARN : BREAKING CHANGE 100 ==> 50
        path_kwargs = kwargs.get("probability_path_params", {})
        self.alpha, self.beta, self.gamma = self._build_interpolant(probability_path, **path_kwargs)
        drift_kwargs = kwargs.get("drift", {})
        self.drift = self._build_drift_model(conditional, self.dim, self.cond_dim, **drift_kwargs)
        time_kwargs = kwargs.get("prior_params", {})
        self.time_prior = self._build_time_prior(prior, **time_kwargs)
        base_dist_kwargs = kwargs.get("base_dist_params", {})
        self.base_dist = self._build_base_dist(
            base_dist, self.dim, self.cond_dim, conditional, **base_dist_kwargs
        )
        self.from_data = base_dist == "data_eps"

        # Attributes definition
        self.register_buffer("data_mean", torch.zeros(self.dim))
        self.register_buffer("data_cov", torch.zeros(self.dim))
        self.register_buffer("data_std", torch.ones(self.dim))
        self.register_buffer("data_whiten_matrix", torch.zeros(self.dim))
        self.register_buffer("data_whiten_inv", torch.zeros(self.dim))
        self.register_buffer("cond_mean", torch.zeros(self.cond_dim))
        self.register_buffer("cond_cov", torch.zeros(self.cond_dim))
        self.register_buffer("cond_std", torch.ones(self.cond_dim))
        self.register_buffer("cond_whiten_matrix", torch.zeros(self.cond_dim))
        self.register_buffer("cond_whiten_inv", torch.zeros(self.cond_dim))

    def set_scales(self, data: Tensor, cond: Tensor, rescale_name: str):
        """Set scales for the model."""
        self.rescale_name = rescale_name
        if rescale_name == "whiten":
            self.cond_cov = cond.cov()
            self.data_cov = data.cov()
            # data
            eigvalues, eigvecs = torch.linalg.eig(self.data_cov)
            whiten_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvalues)) @ eigvecs.T
            self.data_whiten_matrix = whiten_matrix
            whiten_inv = eigvecs @ torch.diag(torch.sqrt(eigvalues)) @ eigvecs.T
            self.data_whiten_inv = whiten_inv
            # cond
            eigvalues, eigvecs = torch.linalg.eig(self.cond_cov)
            cond_whiten_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvalues)) @ eigvecs.T
            self.cond_whiten_matrix = cond_whiten_matrix
            cond_whiten_inv = eigvecs @ torch.diag(torch.sqrt(eigvalues)) @ eigvecs.T
            self.cond_whiten_inv = cond_whiten_inv
        elif rescale_name == "z_score":
            self.data_mean = data.mean(dim=0)
            self.cond_mean = cond.mean(dim=0)
            self.data_std = data.std(dim=0)
            self.cond_std = cond.std(dim=0)
        elif rescale_name == "none":
            pass
        else:
            raise ValueError("Invalid rescale method.")

    def rescale(self, data: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Rescale data and condition."""
        device = data.device
        if self.rescale_name == "whiten":
            data = (data - self.data_mean.to(device)) @ self.data_whiten_matrix.to(device)
            cond = (cond - self.cond_mean.to(device)) @ self.cond_whiten_matrix.to(device)
        elif self.rescale_name == "z_score":
            data = (data - self.data_mean.to(device)) / self.data_std.to(device)
            cond = (cond - self.cond_mean.to(device)) / self.cond_std.to(device)
        elif self.rescale_name == "none":
            pass
        else:
            raise ValueError("Invalid rescale method.")
        return data, cond

    def scale(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Scale data and condition."""
        device = x.device
        if self.rescale_name == "whiten":
            x = x @ self.data_whiten_inv.to(device) + self.data_mean.to(device)
            cond = cond @ self.cond_whiten_inv.to(device) + self.cond_mean.to(device)
        elif self.rescale_name == "z_score":
            x = x * self.data_std.to(device) + self.data_mean.to(device)
            cond = cond * self.cond_std.to(device) + self.cond_mean.to(device)
        elif self.rescale_name == "none":
            pass
        return x, cond

    def sample_time(self, shape: Tuple | int) -> Tensor:
        """Sample time from prior."""
        if isinstance(shape, int):
            shape = (shape,)
        size = torch.Size(shape)
        return self.time_prior.sample(size)

    def sample_source(self, cond: Tensor, nsamples: Optional[int] = None) -> Tensor:
        """
        Sample source from base distribution.
        If the model in unconditional, return the input.
        Else, sample from the base distribution conditioned on the input.
        """
        if self.base_dist is None:
            assert nsamples is None, "Cannot specify nsamples when base_dist is None."
            return cond
        else:
            if nsamples:
                size = torch.Size((nsamples, cond.shape[0]))
            else:
                size = torch.Size((cond.shape[0],))
        if self.from_data:
            noise = self.base_dist.sample(size).to(cond.device)
            if nsamples:
                return cond[None, :] + noise
            return cond + noise
        sources = self.base_dist.sample(size)
        return sources.to(cond.device)

    def interpolant(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Interpolant between x0 and x1 at time t."""
        rsize = (-1,) + (1,) * (x0.dim() - 1)  # Reshape for broadcasting
        xt = (
            self.alpha(t.view(rsize)) * x0
            + self.beta(t.view(rsize)) * x1
            + self.gamma(t.view(rsize)) * torch.randn_like(x0)
        )
        return xt

    def target(self, x0: Tensor, x1: Tensor):
        """Target for the velocity field."""
        return x1 - x0

    def forward(self, xt: Tensor, cond: Tensor, t: Tensor):
        """Compute the velocity field at time t."""
        t = t.view(-1, 1)  # Ensure t is of shape (batch_size, 1)
        if self.conditional:
            drift_input = (xt, t, cond)
        else:
            drift_input = (xt, t)
        v = self.drift(drift_input)
        return v

    def sample(
        self,
        x0: Tensor,
        cond: Tensor,
        device: torch.device,
        num_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        only_last: bool = False,
        disable_tqdm: bool = False,
    ) -> Tensor:
        """Sample trajectory from x0 with condition cond."""
        if not num_steps:
            num_steps = self.num_steps
        dt = 1.0 / num_steps
        xt = x0.cpu()
        self.to("cpu")
        _, cond = self.rescale(xt.cpu(), cond.cpu())
        self.to(device)
        if not only_last:
            traj = [xt.cpu()]

        tbar = trange(num_steps, desc="Sampling", leave=False, disable=disable_tqdm)
        for step in tbar:
            with torch.no_grad():
                if batch_size is not None:
                    bs = min(batch_size, x0.shape[0])
                    N = x0.shape[0]
                    for start in range(0, N, bs):
                        end = min(start + bs, N)
                        xt_batch = xt[start:end]
                        cond_batch = cond[start:end]
                        t_batch = torch.ones((xt_batch.shape[0],)) * (step + 1) * dt
                        v = self.forward(
                            xt_batch.to(device), cond_batch.to(device), t_batch.to(device)
                        ).cpu()
                        xt[start:end] = xt_batch + v * dt
                else:
                    t = torch.ones((x0.shape[0],)) * (step + 1) * dt
                    v = self.forward(xt.to(device), cond.to(device), t.to(device))
                    xt = xt.cpu() + v.cpu() * dt
                if not only_last:
                    traj.append(xt)
        if only_last:
            self.to("cpu")
            out = self.scale(xt.cpu(), cond.cpu())[0]
            self.to(device)
            return out  # shape (batch_size, dim)
        # rescale xt in traj
        traj = [self.scale(xt, cond)[0] for xt in traj]
        return torch.stack(traj, dim=1)  # shape (batch_size, num_steps, dim)

    def sample_backward(
        self,
        x1: Tensor,
        cond: Tensor,
        num_steps: int = 100,
        only_last: bool = False,
        disable_tqdm: bool = False,
    ) -> Tensor:
        """Sample trajectory from x1 with condition cond."""
        dt = 1.0 / num_steps
        xt = x1
        if not only_last:
            traj = [xt]
        tbar = trange(num_steps, desc="Sampling", leave=False, disable=disable_tqdm)
        for step in tbar:
            with torch.no_grad():
                t = torch.ones(x1.shape[0], 1) * (num_steps - step) * dt
                v = self.forward(xt, cond, t.to(x1.device))
                xt = xt - v * dt
                if not only_last:
                    traj.append(xt)
        if only_last:
            return xt
        return torch.stack(traj, dim=1)

    def sample_ensemble(
        self,
        x0: Tensor,
        cond: Tensor,
        num_steps: int = 100,
        only_last: bool = False,
        disable_tqdm: bool = False,
    ) -> Tensor:
        """Sample using average velocity field."""
        dt = 1.0 / num_steps
        xt = x0
        if not only_last:
            traj = [xt]

        # Cond should of shape (batch_size, n_ensemble, dim)
        assert len(cond.shape) == 3, "Condition should be of shape (batch_size, n_ensemble, dim)"
        tbar = trange(num_steps, desc="Sampling", leave=False, disable=disable_tqdm)
        for step in tbar:
            with torch.no_grad():
                t = torch.ones(x0.shape[0], 1) * (step + 1) * dt
                v = torch.zeros_like(xt)
                for i in range(cond.shape[1]):
                    v += self.forward(xt, cond[:, i, :], t.to(x0.device))
                v = v / cond.shape[1]

                xt = xt + v * dt
                if not only_last:
                    traj.append(xt)
        if only_last:
            return xt
        return torch.stack(traj, dim=1)

    def sample_batched(
        self,
        source: Tensor,
        cond: Tensor,
        device: torch.device,
        num_steps: int = 100,
        only_last: bool = False,
        batch_size: int = 100,
        disable_tqdm: bool = False,
    ) -> Tensor:
        """
        Sample batch of trajectories

        Args:
            cond (Tensor): Condition for the trajectories
            n_traj (int): Number of trajectories to sample for each source point
            device (torch.device): Device to use
            num_steps (int): Number of steps in the trajectory
            only_last (bool): Whether to return only the last point
            batch_size (int): Batch size for the sampling
            disable_tqdm (bool): Disable tqdm

        Returns:
            Tensor: Sampled trajectories un shape (n_traj * batch_size, num_steps, dim)
            or (batch_size, dim) if only_last is True
        """
        self.eval()

        x_tilde = []
        _, cond = self.rescale(source, cond)
        loader_source, loader_cond = loader_from_tensor(source, cond, batch_size)
        for (source_batch,), (y_batch,) in zip(loader_source, loader_cond):
            x_batch = self.sample(
                source_batch.to(device),
                y_batch.to(device),
                num_steps,
                only_last=only_last,
                disable_tqdm=disable_tqdm,
            ).cpu()
            x_tilde.append(self.scale(x_batch, y_batch)[0])
        x_tilde = torch.cat(x_tilde, dim=0)
        return x_tilde

    @staticmethod
    def _check_args(
        conditional: bool,
        probability_path: str,
        prior: str,
        base_dist: str,
        **kwargs,
    ) -> None:
        """Check arguments."""
        try:
            assert probability_path in [
                "ot",
                "ot2",
                "vf",
            ], "Invalid probability path."
            assert prior in ["power", "uniform"], "Invalid prior."
            assert base_dist in ["gaussian", "conditional", "none", "data_eps"], (
                "Invalid base distribution."
            )
        except AssertionError as e:
            print("Invalid arguments for FlowMatching model :")
            raise ValueError(e)
        try:
            assert kwargs.get("drift", {})["architecture"] in SUPPORTED_ARCHITECTURES
        except AssertionError as e:
            print("Invalid parameters :")
            raise ValueError(e)

    @staticmethod
    def _build_interpolant(
        probability_path: str, **kwargs
    ) -> Tuple[
        Callable[[Tensor], Tensor],
        Callable[[Tensor], Tensor],
        Callable[[Tensor], Tensor],
    ]:
        """Build interpolant."""
        if probability_path == "ot":

            def alpha(t: Tensor) -> Tensor:
                return 1.0 - t

            def beta(t: Tensor) -> Tensor:
                return t

            def gamma(t: Tensor) -> Tensor:
                return torch.sqrt(2 * t * (1.0 - t))
        elif probability_path == "ot2":
            sigma_min = kwargs.get("sigma_min", 1e-4)

            def alpha(t: Tensor) -> Tensor:
                return 1.0 - (1 - sigma_min) * t

            def beta(t: Tensor) -> Tensor:
                return t

            def gamma(t: Tensor) -> Tensor:
                return torch.Tensor([0.0]).to(t.device)
        elif probability_path == "vf":
            raise NotImplementedError("Variance Preserving flow not implemented yet.")
        return alpha, beta, gamma

    @staticmethod
    def _build_drift_model(
        conditional, dim: torch.Size, cond_dim: torch.Size, **kwargs
    ) -> nn.Module:
        """Build drift model."""
        architecture = kwargs.get("architecture", "resmlp")
        if architecture == "resmlp":
            hidden_features = kwargs.get("hidden_dim", (64, 64))
            mlp_kwargs = kwargs.get("mlp_params", {"activation": nn.ELU})
            if conditional:
                assert len(dim) == 1, "Conditional ResMLP must have 1-dimensional target."
                assert len(cond_dim) == 1, "Conditional ResMLP must have 1-dimensional condition."
                return ResMLP(dim[0] + cond_dim[0] + 1, dim[0], hidden_features, **mlp_kwargs)
            else:
                raise NotImplementedError()
        elif architecture == "cfnet":
            assert conditional, "Non Conditional CFNet is not implemented."
            return create_cf(
                kwargs["posterior_kwargs"],
                kwargs.get("embedding_kwargs", {}),
                kwargs.get("theta_embedding_kwargs", {}),
            )
        else:
            raise ValueError("Invalid drift model architecture.")

    @staticmethod
    def _build_time_prior(prior: str, **kwargs) -> Distribution:
        """Build time prior."""
        if prior == "uniform":
            return Uniform(0.0, 1.0)
        if prior == "power":
            rate = kwargs.get("rate", 1.5)
            return Beta(rate, 1.0)
        else:
            raise ValueError("Invalid time prior distribution.")

    @staticmethod
    def _build_base_dist(
        name: str, dim: torch.Size, cond_dim: torch.Size, cond: bool, **kwargs
    ) -> Distribution | None:
        """Build base distribution."""
        if name == "data":
            return None
        if name == "data_eps":
            eps = kwargs.get("eps", 0.01)
            return Normal(loc=torch.zeros(dim), scale=eps * torch.ones(dim))
        if name == "gaussian":
            return Normal(
                loc=torch.zeros(dim),
                scale=torch.ones(dim),
            )
        else:
            raise ValueError("Invalid base distribution.")

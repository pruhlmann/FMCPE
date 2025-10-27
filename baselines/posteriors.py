from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import ot
import torch
import torch.nn as nn
from lampe.inference import NPE
from pyro.distributions import Distribution
from torch import Tensor
from zuko.distributions import DiagNormal
from zuko.flows import MAF, NSF, Flow, UnconditionalDistribution
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.flows.neural import UMNN

from flow_matching.torch_flow import FlowMatching
from simulator.base import Simulator
from utils.metrics import (
    MMD,
    classifier_two_samples_test,
    classifier_two_samples_test_torch,
    mse,
    stein_discrepancy,
)
from utils.networks import get_embedding_network


class Posterior(nn.Module, ABC):
    def __init__(self, name, theta_dim, obs_dim):
        super().__init__()
        self.name = name
        self._theta_dim = theta_dim
        self._obs_dim = obs_dim

    @property
    def theta_dim(self):
        return self._theta_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    @abstractmethod
    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the log probability of theta given x."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, x: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        """
        Sample from the posterior given x.
        Args:
            x: Observed data, shape (batch_size, dim).
            nsamples: Number of samples to draw.

        Returns:
            samples: Samples from the posterior, shape (nsamples, batch_size, dim).
        """
        raise NotImplementedError

    def evaluate_metrics(
        self,
        y: torch.Tensor,
        theta: torch.Tensor,
        device: torch.device,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> dict:
        """
        Evaluate the metrics for the posterior.

        Args:
            y: Observed data, shape (nobs, dim).
            theta: GT Parameters, shape (nobs, dim) ie one sample per observation.
            save_path: Path to save the results.
        """
        ntest = theta.shape[0]
        compute_c2st = kwargs.get("compute_c2st", True)
        compute_wasserstein = kwargs.get("compute_wasserstein", True)
        compute_mse = kwargs.get("compute_mse", False)
        simulator: Optional[Simulator] = kwargs.get("simulator", None)
        with torch.no_grad():
            y_emb = y.flatten(start_dim=1)
        if simulator is not None and simulator.name in ["wind_tunnel", "pendulum"]:
            out_dim = 10
            emb_net = "conv1d_v2" if simulator.name == "wind_tunnel" else "conv1d"
            model_kwargs = {
                "emb_net": emb_net,
                "theta_dim": theta.shape[1],
                "x_dim": y_emb.shape[1:],
                "out_dim": out_dim,
            }
            training_kwargs = {
                "epochs": 150,
                "device": "cuda",
            }
        elif simulator is not None and simulator.name == "light_tunnel":
            out_dim = 10
            emb_net = "conv2d"
            image_size = simulator.obs_dim
            model_kwargs = {
                "emb_net": emb_net,
                "theta_dim": theta.shape[1],
                "x_dim": image_size,
                "out_dim": out_dim,
                "image_size": image_size,
            }
            training_kwargs = {
                "device": "cuda",
            }
        else:
            model_kwargs = None
            training_kwargs = {
                "device": "cuda",
            }
        m = kwargs.get("m", 100)  # number of samples for acauc
        metrics = {}
        if save_path is not None:
            try:
                theta_pred = torch.load(save_path / f"theta_pred_{ntest}.pt")
            except FileNotFoundError:
                theta_pred = self.sample(y, 1, device, **kwargs).squeeze(0)
                torch.save(theta_pred, save_path / f"theta_pred_{ntest}.pt")
        else:
            theta_pred = self.sample(y, 1, device, **kwargs).squeeze(0)
        # assert theta_pred.shape == theta.shape, "theta_pred and theta_true must have the same shape"
        if compute_c2st:
            print("\tComputing Joint C2ST")
            theta_y_pred = torch.cat([theta_pred, y_emb], dim=1)
            theta_true_y = torch.cat([theta, y_emb], dim=1)
            metrics["joint_c2st"] = classifier_two_samples_test_torch(
                theta_true_y,
                theta_y_pred,
                z_score=True,
                n_folds=3,
                scoring="accuracy",
                model="mlp",
                model_kwargs=model_kwargs,
                training_kwargs=training_kwargs,
            )
        if compute_wasserstein:
            print("\tComputing Wasserstein")
            a, b = (
                torch.ones((theta.shape[0],)) / theta.shape[0],
                torch.ones((theta_pred.shape[0],)) / theta_pred.shape[0],
            )
            M = ot.dist(theta, theta_pred)
            metrics["wasserstein"] = torch.sqrt(ot.emd2(a, b, M)).item()

            print("\tComputing joint Wasserstein")
            theta_y_pred = torch.cat([theta_pred, y_emb], dim=1)
            theta_true_y = torch.cat([theta, y_emb], dim=1)
            a, b = (
                torch.ones((theta_true_y.shape[0],)) / theta_true_y.shape[0],
                torch.ones((theta_y_pred.shape[0],)) / theta_y_pred.shape[0],
            )
            M = ot.dist(theta_true_y, theta_y_pred)
            metrics["joint_wasserstein"] = torch.sqrt(ot.emd2(a, b, M)).item()

        if compute_mse:
            print("\tComputing MSE")
            if save_path is not None:
                try:
                    theta_pred_m = torch.load(save_path / "theta_pred_m.pt")
                except FileNotFoundError:
                    theta_pred_m = self.sample(y, m, device, **kwargs).to("cpu")
                    torch.save(theta_pred_m, save_path / "theta_pred_m.pt")
            # rescale samples for light tunnel
            if simulator is not None and simulator.name == "light_tunnel":
                a = simulator.prior_params["low"]
                b = simulator.prior_params["high"]
                theta_pred_m = (theta_pred_m - a[None, None, :]) / (
                    b[None, None, :] - a[None, None, :]
                )
            metrics["mse"] = mse(theta, theta_pred_m)

        return metrics

    def evaluate_conditional_metrics(
        self,
        y: torch.Tensor,
        theta: torch.Tensor,
        device: torch.device,
        true_dist: Optional[Callable[[Tensor], Distribution]] = None,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> dict:
        """Evaluate the conditional metrics for the posterior.
        Args:
            y: Observed data, shape (nobs,dim).
            theta: GT Parameters, shape (nsamples, nobs ,dim).
            metrics_to_compute: List of metrics to compute.
            simulator: Simulator object to compute ground_truth pdf.
        returns:
            metrics: Dictionary of computed metrics.
        """
        compute_lpp = kwargs.get("compute_lpp", True)
        compute_c2st = kwargs.get("compute_c2st", True)
        compute_stein_discrepancy = kwargs.get("compute_stein_discrepancy", True)
        compute_wasserstein = kwargs.get("compute_wasserstein", True)

        metrics = {}
        ntest = theta.shape[0]
        nobs = theta.shape[1]
        dim = theta.shape[2]

        if save_path is not None:
            try:
                theta_pred = torch.load(save_path / "theta_pred.pt")
            except FileNotFoundError:
                theta_pred = self.sample(y, ntest, device)
                torch.save(theta_pred, save_path / "theta_pred.pt")
        else:
            theta_pred = self.sample(y, ntest, device)

        if compute_c2st:
            print("\tComputing C2ST")
            c2st = [
                classifier_two_samples_test(
                    theta[:, i, :], theta_pred[:, i, :], n_folds=3, scoring="accuracy"
                )
                for i in range(nobs)
            ]
            metrics["c2st"] = c2st

        if compute_wasserstein:
            print("\tComputing Wasserstein")
            ws = []
            for i in range(nobs):
                theta_pred_i = theta_pred[:, i, :]
                gt_theta_i = theta[:, i, :]
                a, b = (
                    torch.ones((ntest,)) / ntest,
                    torch.ones((ntest,)) / ntest,
                )  # uniform distribution on samples
                M = ot.dist(theta_pred_i, gt_theta_i)
                ws.append(torch.sqrt(ot.emd2(a, b, M)).item())
            metrics["wasserstein"] = ws

        if compute_stein_discrepancy:
            print("\tComputing Stein discrepancy")
            sds = []
            if true_dist is None:
                # skip stein discrepancy computation
                print("true_dist must be provided for Stein discrepancy computation")
                return metrics
            for i in range(nobs):
                theta_pred_i = theta_pred[:, i, :]
                y_i = y[i, :].reshape(1, -1).repeat_interleave(ntest, 0)

                def score_fn(theta):
                    log_prob = true_dist(y_i).log_prob(theta)  # log p(θ | y)
                    log_prob.sum().backward()  # Compute gradients
                    theta_grad = theta.grad  # ∇θ log p(θ | y)
                    return theta_grad

                sd = stein_discrepancy(theta_pred_i, score_fn)
                sds.append(sd)
            metrics["stein_discrepancy"] = sds

        return metrics


class UMNNFlow(Flow):
    def __init__(self, features, context, **kwargs):
        integrand_params = kwargs.get("integrand_params", {})
        conditoner_params = kwargs.get("conditioner_params", {})
        embedding_dim = kwargs.get("embedding_dim", None)
        transforms = kwargs.get("ntransform", None)
        neural_nets = [UMNN(embedding_dim, **integrand_params) for _ in range(transforms)]

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


class Estimator(nn.Module):
    def __init__(self, task_name: str, dim: torch.Size, cond_dim: torch.Size, **kwargs):
        super(Estimator, self).__init__()
        density_estimator = kwargs.get("density_estimator", "nsf")
        embedding_net = kwargs.get("embedding_net", {})
        npe_params = kwargs.get("npe_params", {})
        if len(dim) != 1:
            raise ValueError("dim must be a 1D tensor representing the parameter dimension.")
        self.theta_dim = dim
        self.cond_dim = cond_dim
        self.obs_dim = embedding_net.get("output_dim", cond_dim[0])
        self.npe = NPE(
            dim[0],
            self.obs_dim,
            build=get_build_fn(task_name, density_estimator, **npe_params),
        )
        self.embedding_net = get_embedding_network(task_name, **embedding_net)

        # Attributes definition
        self.register_buffer("data_mean", torch.zeros(dim))
        self.register_buffer("data_cov", torch.zeros(dim))
        self.register_buffer("data_std", torch.ones(dim))
        self.register_buffer("data_whiten_matrix", torch.zeros(dim))
        self.register_buffer("data_whiten_inv", torch.zeros(dim))
        self.register_buffer("cond_mean", torch.zeros(cond_dim))
        self.register_buffer("cond_cov", torch.zeros(cond_dim))
        self.register_buffer("cond_std", torch.ones(cond_dim))
        self.register_buffer("cond_whiten_matrix", torch.zeros(cond_dim))
        self.register_buffer("cond_whiten_inv", torch.zeros(cond_dim))

    def set_scales(self, data: Tensor, cond: Tensor, rescale_name: str):
        """Set scales for the model."""
        self.rescale_name = rescale_name
        self.data_mean = data.mean(dim=0)
        self.cond_mean = cond.mean(dim=0)
        if rescale_name == "whiten":
            self.data_cov = data.cov()
            self.cond_cov = cond.cov()
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

    def embedding(self, x: Tensor) -> Tensor:
        """Embed the input data."""
        if x.shape[1] == self.obs_dim:
            return x
        else:
            _, x = self.rescale(torch.zeros(1, self.theta_dim), x)
            return self.embedding_net(x)

    def forward(self, theta, x):
        theta, x = self.rescale(theta, x)
        return self.npe(theta, self.embedding_net(x))

    def _flow(self, x):
        return self.npe.flow(self.embedding_net(x))

    def sample(self, x, nsamples, device, **kwargs):
        self.to(device)  # Move to device
        _, x = self.rescale(torch.zeros(1, *self.theta_dim).to(x.device), x)
        space = kwargs.get("space", "data")
        if kwargs.get("batch_size", None) is not None:
            samples = []
            bs = kwargs["batch_size"]
            N = x.shape[0]  # number of observations
            for start in range(0, N, bs):
                with torch.no_grad():
                    end = min(start + bs, N)
                    x_batch = x[start:end].to(device)
                    if space == "latent":
                        samples_batch = self.npe.flow(x_batch).sample((nsamples,))
                    elif space == "data":
                        samples_batch = self._flow(x_batch).sample(
                            (nsamples,)
                        )  # shape (nsamples, end-start, *self.theta_dim)
                    samples.append(samples_batch.cpu())
            samples = torch.cat(samples, dim=1)  # shape (nsamples, N ,*self.theta_dim)
        else:
            with torch.no_grad():
                # Sample from the flow
                # x is shape (N, *self.obs_dim)
                # nsamples is the number of samples to draw
                # samples will be of shape (nsamples, N, *self.theta_dim)
                if space == "latent":
                    samples = self.npe.flow(x.to(device)).sample((nsamples,)).to("cpu")
                elif space == "data":
                    samples = (
                        self._flow(x.to(device)).sample((nsamples,)).to("cpu")
                    )  # shape (nsamples, N, *self.theta_dim)
        samples, _ = self.scale(samples.to(x.device), x)
        # print("Samples shape :", samples.shape)
        return samples

    def log_prob(self, theta, x, **kwargs):
        theta, x = self.rescale(theta, x)
        return self.npe.flow(self.embedding_net(x)).log_prob(theta)


class DirectPosteriorEstimator(Posterior):
    def __init__(self, name: str, posterior: Estimator):
        super().__init__(name, posterior.theta_dim, posterior.obs_dim)
        self.posterior = posterior

    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        return self.posterior.sample(y, nsamples, device, batch_size=kwargs.get("batch_size", None))

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.posterior.log_prob(theta, y)


class DirectFlowMatchingPosterior(Posterior):
    def __init__(
        self,
        name: str,
        flow: FlowMatching,
    ):
        super().__init__(name, flow.dim, flow.cond_dim)
        self.flow = flow

    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        batch_size = kwargs.get("batch_size", None)
        self.flow.to(device)  # Move to device
        source = self.flow.sample_source(y, nsamples)  # shape (nsamples, N, *obs_dim)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)
        samples = self.flow.sample(
            source.reshape(-1, *source.shape[2:]),  # shape (nsamples * N, *obs_dim)
            cond.reshape(-1, *cond.shape[2:]),  # shape (nsamples * N, *obs_dim)
            device,
            only_last=True,
            num_steps=20,
            batch_size=batch_size,
        )
        return samples.reshape(nsamples, y.shape[0], *self.theta_dim)

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("log_prob not implemented for DirectFlowMatchingPosterior")


class FlowMatchingPosterior(Posterior):
    def __init__(
        self,
        name: str,
        posterior: Union[Estimator, DirectFlowMatchingPosterior],
        flow: FlowMatching,
        theta_dim,
        obs_dim,
    ):
        super().__init__(name, theta_dim, obs_dim)
        self.denoiser = flow
        self.npe = posterior

    def sample(
        self, y: torch.Tensor, nsamples: int, device: torch.device, **kwargs
    ) -> torch.Tensor:
        batch_size = kwargs.get("batch_size", None)
        self.denoiser.to(device)  # Move to device
        source = self.denoiser.sample_source(y, nsamples)  # shape (nsamples, N, obs_dim)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)
        x_tilde = self.denoiser.sample(
            source.reshape(-1, *source.shape[2:]),  # shape (nsamples * N, *obs_dim)
            cond.reshape(-1, *cond.shape[2:]),  # shape (nsamples * N, *obs_dim)
            device,
            only_last=True,
            batch_size=batch_size,
            num_steps=20,
        )
        return (
            self.npe.sample(x_tilde, 1, device, batch_size=batch_size)
            .squeeze(0)
            .reshape(nsamples, y.shape[0], *self.theta_dim)
        )

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        ntraj = kwargs.get("ntraj", 1)
        batch_size = kwargs.get("batch_size", 100)
        device = y.device
        source = (
            self.denoiser.sample_source(y, ntraj).transpose(0, 1).to(device)
        )  # shape (ntraj, N ,obs_dim)
        source = source.reshape(-1, source.shape[-1])  # shape (ntraj * N, obs_dim)
        cond = y.repeat_interleave(ntraj, dim=0)  # shape (ntraj * N, obs_dim)
        x_tilde = self.denoiser.sample_batched(
            source, cond, device, batch_size=batch_size, only_last=True
        )
        theta = theta.repeat_interleave(ntraj, 0)
        self.npe.to("cpu")  # To avoid GPU memory issues
        pdf = torch.exp(self.npe.log_prob(theta.cpu(), x_tilde))
        self.npe.to(device)  # Move back to the original device
        # resize to (ntraj, N)
        lpp = torch.log(pdf.reshape(ntraj, -1).mean(dim=0))
        return lpp


class DualFlowPosteriorEstimator(Posterior):
    def __init__(
        self,
        name: str,
        base_dist: Estimator | DirectFlowMatchingPosterior,
        flow_theta: FlowMatching,
        flow_x: Optional[FlowMatching] = None,
        embedding_net: Optional[nn.Module] = None,
    ):
        super().__init__(name, flow_theta.dim, flow_theta.cond_dim)
        self.posterior_transform = flow_theta
        self.proposal = base_dist
        self.denoiser = flow_x
        self.embedding_net = embedding_net
        self.space = "data" if embedding_net is None else "latent"

    def y_to_x(
        self, y: torch.Tensor, nsamples: int, device, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        # Convert y to x using the flow matching model
        if self.denoiser is not None:
            source = self.denoiser.sample_source(y, nsamples)
            broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
            cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)
            x_tilde = self.denoiser.sample(
                source.reshape(-1, *source.shape[2:]),  # shape (nsamples * N, *obs_dim)
                cond.reshape(-1, *cond.shape[2:]),  # shape (nsamples * N, *obs_dim)
                device,
                only_last=True,
                batch_size=batch_size,
                num_steps=20,
            )
            return x_tilde.reshape(nsamples, y.shape[0], *self.denoiser.dim)
        else:
            broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
            cond = (
                y.unsqueeze(0).expand(*broadcast_shape).to(device)
            )  # shape (nsamples, N, *obs_dim)
            return cond

    def sample(self, y: torch.Tensor, nsamples: int, device, **kwargs) -> torch.Tensor:
        batch_size = kwargs.get("batch_size", None)
        x = self.y_to_x(y, nsamples, device, batch_size=batch_size)
        x = x.reshape(-1, *x.shape[2:])
        source = self.proposal.sample(
            x, 1, device, batch_size=batch_size, space=self.space
        ).squeeze(0)  # shape (nsamples* N, obs_dim)
        source, _ = self.posterior_transform.rescale(source, y)
        broadcast_shape = (nsamples, y.shape[0], *([-1] * (y.dim() - 1)))
        cond = y.unsqueeze(0).expand(*broadcast_shape)  # shape (nsamples, N, *obs_dim)
        samples = self.posterior_transform.sample(
            source,  # shape (nsamples * N, *obs_dim)
            cond.reshape(-1, *cond.shape[2:]),  # shape (nsamples * N, *obs_dim)
            device,
            only_last=True,
            batch_size=batch_size,
            num_steps=20,
        )
        return samples.reshape(nsamples, y.shape[0], *self.theta_dim)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        print("log_prob not implemented for DirectFlowMatchingPosterior")
        return torch.zeros(theta.shape[0], device=x.device)


def get_build_fn(task: str, density_estimator: str, **kwargs) -> Callable[[int, int], Flow]:
    if task in ["pendulum", "light_tunnel", "wind_tunnel"]:
        embedding_dim = kwargs.get("embedding_dim", None)
        transforms = kwargs.get("ntransform", None)
        if embedding_dim is None or transforms is None:
            raise ValueError("embedding_dim and ntransform must be provided")
        return lambda f, c: UMNNFlow(f, c, **kwargs)
    elif task == "gaussian":
        return lambda f, c: MAF(f, c, **kwargs)
    else:
        if density_estimator == "nsf":
            return lambda f, c: NSF(f, c, **kwargs)
        elif density_estimator == "maf":
            return lambda f, c: MAF(f, c, **kwargs)


torch.serialization.add_safe_globals([Estimator, UMNNFlow])

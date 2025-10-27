from copy import deepcopy
import copy
import json
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple

from jax._src import state
import torch
from torch import Tensor
from lampe.inference import NPELoss
from lampe.utils import GDStep
from mlxp.logger import Logger
from tqdm.auto import trange

from flow_matching.torch_flow import FlowMatching
from baselines.posteriors import (
    DirectFlowMatchingPosterior,
    DirectPosteriorEstimator,
    DualFlowPosteriorEstimator,
    Estimator,
    FlowMatchingPosterior,
    Posterior,
)
from simulator import get_simulator
from simulator.base import Simulator
from utils.misc import train_val_split, train_val_split_n
from utils.networks import ContinuousFlow
from utils.transform import LogitBoxTransform


class Trainer:
    def __init__(self, name: str, logger: Logger, model_path: Path, task: Dict):
        self.name = name
        self.logger = logger
        self.path = model_path
        self.config = task[name]["config"]

    def train(
        self,
        data_sim: Tuple[Tensor, Tensor],
        data_cal: Tuple[Tensor, Tensor, Tensor],
        cal_values: List[int],
        naugment: int,
        device: torch.device,
        load: bool = False,
        compute_reference: bool = False,
        **kwargs,
    ) -> Dict[str, Posterior]:
        """Train the model on the provided data."""
        train_fn = globals().get(f"train_{self.name}")
        if train_fn is None:
            raise ValueError(f"Training function for {self.name} not found.")
        models = {}
        # Pre-train if necessary
        pre_train_fn = globals().get(f"pretrain_{self.name}")
        print("##############################")
        print(f"Training method {self.name} on {cal_values} calibration data points")
        print("##############################")
        if pre_train_fn is not None:
            print(f"Pre-training {self.name} method")
            pre_train_fn(
                data_sim,
                self.config,
                device,
                logger=self.logger,
                logname=self.name + "_pretrained",
                model_path=self.path,
                **kwargs,
            )
            print("Pre-training done")
        for ncal in cal_values:
            if ncal == cal_values[-1] and not compute_reference:
                print(
                    f"Skipping reference model for {self.name} with {ncal} calibration data points"
                )
                continue
            print(f"Training {self.name} with {ncal} calibration data points")
            key = str(ncal) if (ncal != cal_values[-1] or not compute_reference) else "ref"
            theta_cal, x_cal, y_cal = (
                data_cal[0][: ncal * naugment],
                data_cal[1][: ncal * naugment],
                data_cal[2][: ncal * naugment],
            )
            posterior = train_fn(
                (theta_cal, x_cal, y_cal),
                self.config,
                device,
                logger=self.logger,
                logname=f"{self.name}_{key}",
                model_path=self.path,
                load=load,
                **kwargs,
            )
            models[key] = posterior
        return models


def train_npe(
    task: Dict,
    theta: Tensor,
    x: Tensor,
    device: torch.device,
    model_path: Path,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=True,
    use_pretrained: bool = False,
) -> Estimator:
    """Neural Posterior estimation.

    Args:
        task: Configuration for the task.
        theta: Samples from the prior.
        x: Samples from the simulator or the true data.
        simulator: Simulator object.
        device: Device for computation.
        training_params: Training parameters.
        **kwargs: Additional arguments.
    Returns:
        Estimator: Estimator object.
    """
    # Training parameters
    training_params = task["npe"]["training"]
    epochs = training_params.get("epochs", 1000)
    lr = training_params.get("lr", 5e-4)
    batch_size = training_params.get("batch_size", 256)
    train_size = training_params.get("train_size", 0.9)
    max_patience = training_params.get("max_patience", 20)
    rescale = training_params.get("rescale", "z_score")

    estimator = Estimator(
        task["name"],
        theta.shape[1:],
        x.shape[1:],
        **task["npe"]["params"],
    )

    if load:
        # Load
        try:
            estimator.load_state_dict(
                torch.load(
                    model_path / (logname + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
            estimator.rescale_name = task[
                "rescale"
            ]  # TODO : save rescale mith statedict to avoid conflict
            return estimator
        except FileNotFoundError:
            print("Model not found, training from scratch.")
    if use_pretrained:
        try:
            print("\tUsing pretrained NPE model")
            estimator.load_state_dict(
                torch.load(
                    model_path / ("npe.pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained model {'npe.pth'} not found. ")

    estimator.set_scales(theta, x, rescale)
    estimator.to(device)

    loss = NPELoss(estimator)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=5.0)
    train_loader, val_loader = train_val_split(theta, x, train_size, batch_size)

    best_loss = float("inf")
    best_state_dict = estimator.state_dict()
    patience = 0
    for epoch in (bar := trange(epochs, desc="Training NPE estimator", total=epochs)):
        estimator.train()
        loss_list_train = []
        loss_list_val = []
        for thb, xb in train_loader:
            loss_list_train.append(step(loss(thb.to(device), xb.to(device))))
        estimator.eval()
        with torch.no_grad():
            for thb, xb in val_loader:
                loss_list_val.append(loss(thb.to(device), xb.to(device)).item())
        loss_value_train = torch.stack(loss_list_train).mean().item()
        loss_value_val = torch.tensor(loss_list_val).mean().item()
        bar.set_postfix(npe_loss=loss_value_train, npe_val_loss=loss_value_val)

        # Logging
        if logger is not None:
            logger.log_metrics(
                {
                    "train_loss": loss_value_train,
                    "val_loss": loss_value_val,
                    "epoch": epoch,
                },
                log_name=logname,
            )

        patience += 1

        if loss_value_val < best_loss:
            best_loss = loss_value_val
            patience = 0
            best_state_dict = deepcopy(estimator.state_dict())

        if patience >= max_patience:
            bar.write("\tEarly stopping, best model saved.")
            bar.write(f"\tBest validation loss: {best_loss}")
            torch.save(estimator, model_path / f"{logname}.pkl")
            break
    estimator.load_state_dict(
        best_state_dict,
    )
    # Save
    if save:
        torch.save(estimator.state_dict(), model_path / f"{logname}.pth")
    return estimator


def train_flow_matching(
    target: Tensor,
    cond: Tensor,
    config: Dict,
    device: torch.device,
    logger: Optional[Logger],
    logname: str = "",
    load: bool = False,
    save: bool = True,
    model_path: Path = Path("models"),
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 128,
    train_size: float = 0.8,
    max_patience: int = 20,
    rescale: str = "z_score",
    use_pretrained: bool = False,
):
    flow_matching = FlowMatching(
        config["conditional"],
        config["probability_path"],
        config["prior"],
        config["base_dist"],
        target.shape[1:],
        cond_dim=cond.shape[1:] if config["conditional"] else None,
        **config["params"],
    )
    if use_pretrained:
        print("\tUsing pretrained flow matching model")
        try:
            flow_matching.load_state_dict(
                torch.load(
                    model_path / (logname.rsplit("_", 1)[0] + "_pretrained" + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pretrained model {logname + '_pretrained' + '.pth'} not found. "
            )
    if load:
        print("\tLoading model from", model_path / (logname + ".pth"))
        try:
            flow_matching.load_state_dict(
                torch.load(
                    model_path / (logname + ".pth"),
                    weights_only=True,
                    map_location=device,
                )
            )
            return flow_matching
        except FileNotFoundError:
            print("Model not found, training from scratch.")
    flow_matching.to(device)
    flow_matching.set_scales(target, cond, rescale)
    if not flow_matching.conditional:
        # Shuffle cond if unconditional
        # Not shuffling induces a coupling in the data
        cond = cond[torch.randperm(len(cond))]
    train_loader, val_loader = train_val_split(target, cond, train_size, batch_size)
    optimizer = torch.optim.Adam(flow_matching.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float("inf")
    best_state_dict = flow_matching.state_dict()
    bar = trange(
        epochs,
        desc=f"Training Flow Matching, conditional : {flow_matching.conditional}",
        total=epochs,
    )

    for epoch in bar:
        # Training
        flow_matching.train()
        loss_list_train = []
        loss_list_val = []
        for xbatch, ybatch in train_loader:
            with torch.no_grad():
                # Rescale the data
                xbatch, ybatch = flow_matching.rescale(xbatch.to(device), ybatch.to(device))
            optimizer.zero_grad()
            source = flow_matching.sample_source(ybatch).to(device)
            t = flow_matching.sample_time(xbatch.shape[0]).to(device)
            xt = flow_matching.interpolant(t, source, xbatch.to(device))
            v = flow_matching.target(source, xbatch.to(device))
            loss = (flow_matching(xt, ybatch.to(device), t) - v).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_matching.parameters(), 1.0)
            optimizer.step()
            loss_list_train.append(loss.detach())
        scheduler.step()

        # Validation
        flow_matching.eval()
        with torch.no_grad():
            for xbatch, ybatch in val_loader:
                xbatch, ybatch = flow_matching.rescale(xbatch.to(device), ybatch.to(device))
                optimizer.zero_grad()
                source = flow_matching.sample_source(ybatch).to(device)
                t = flow_matching.sample_time(xbatch.shape[0]).to(device)
                xt = flow_matching.interpolant(t, source, xbatch.to(device))
                v = flow_matching.target(source, xbatch.to(device))
                loss = (flow_matching(xt, ybatch.to(device), t) - v).pow(2).mean()
                loss_list_val.append(loss.detach())
        loss_value_train = torch.stack(loss_list_train).mean().item()
        loss_value_val = torch.stack(loss_list_val).mean().item()
        bar.set_postfix(train=loss_value_train, val=loss_value_val)

        # Logging
        if logger is not None:
            logger.log_metrics(
                {
                    "train_loss": loss_value_train,
                    "val_loss": loss_value_val,
                    "epoch": epoch,
                },
                log_name=logname,
            )
    torch.save(flow_matching.state_dict(), model_path / (logname + ".pth"))
    return flow_matching


def train_fm_post_transform(
    data: Tuple[Tensor, Tensor, Tensor],
    config: Dict,
    device: torch.device,
    logger: Optional[Logger] = None,
    logname: str = "",
    load=False,
    save=False,
    model_path: Path = Path("models"),
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 128,
    train_size: float = 0.8,
    max_patience: int = 20,
    rescale: str = "z_score",
) -> DualFlowPosteriorEstimator:
    theta, x, y = data

    # ---- Load simulator ---- #
    with open(model_path / "simulator.json", "r") as f:
        config_simulator = json.load(f)
        params = config_simulator["params"]
        name = config_simulator["name"]
        simulator = get_simulator(name, **params)
    try:
        simulate = simulator.get_simulator(misspecified=True)
    except NotImplementedError:
        simulate = None

    flow_matching_theta = FlowMatching(
        config["flow_theta"]["conditional"],
        config["flow_theta"]["probability_path"],
        config["flow_theta"]["prior"],
        config["flow_theta"]["base_dist"],
        theta.shape[1:],
        cond_dim=y.shape[1:],
        **config["flow_theta"]["params"],
    )
    flow_matching_x = FlowMatching(
        config["flow_x"]["conditional"],
        config["flow_x"]["probability_path"],
        config["flow_x"]["prior"],
        config["flow_x"]["base_dist"],
        x.shape[1:],
        cond_dim=y.shape[1:],
        **config["flow_x"]["params"],
    )

    # ---- Load npe ---- #
    if config["npe"] == "npe":
        with open(model_path / "npe.pkl", "rb") as f:
            npe: Estimator = pickle.load(f)
        # embedding_net_x = copy.deepcopy(npe.embedding_net)
        npe = DirectPosteriorEstimator("npe", npe)

    elif config["npe"] == "fmpe":
        with open(model_path / "npe_fmpe.json", "rb") as f:
            config_npe = json.load(f)
        flow_matching = FlowMatching(
            config_npe["conditional"],
            config_npe["probability_path"],
            config_npe["prior"],
            config_npe["base_dist"],
            theta.shape[1:],
            cond_dim=x.shape[1:] if config_npe["conditional"] else None,
            **config_npe["params"],
        )
        flow_matching.rescale_name = rescale
        npe = DirectFlowMatchingPosterior("npe_fmpe", flow_matching)
        drift: ContinuousFlow = flow_matching.drift
        # embedding_net_x = copy.deepcopy(drift.context_embedding_net)
    else:
        raise ValueError(f"Unknown npe type {config['npe']}, should be 'npe' or 'fmpe'")
    npe.to(device)

    # Training parameters
    if load:
        print(f"\tLoading model from {model_path / f'{logname}.pth'}")
        try:
            flow_matching_theta.load_state_dict(
                torch.load(
                    model_path / f"{logname}_theta.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
            flow_matching_x.load_state_dict(
                torch.load(
                    model_path / f"{logname}_x.pth",
                    weights_only=True,
                    map_location=device,
                )
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model path does not exist, set load_models=False in config.yaml"
            )
        posterior = DualFlowPosteriorEstimator(
            "fm_post_transform",
            base_dist=npe,
            flow_theta=flow_matching_theta,
            flow_x=flow_matching_x,
        )
        return posterior

    flow_matching_theta.to(device)
    flow_matching_theta.set_scales(theta, y, rescale)
    flow_matching_x.to(device)
    flow_matching_x.set_scales(x, y, rescale)

    lambda_ = 0.5  # Weight for loss_theta; (1 - lambda_) will weight loss_x

    # Combine parameters of both models
    optimizer = torch.optim.Adam(
        list(flow_matching_theta.parameters()) + list(flow_matching_x.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Prepare data loaders
    train_loader, val_loader = train_val_split_n(
        theta, x, y, train_size=train_size, batch_size=batch_size
    )

    for epoch in (bar := trange(epochs, desc="Training flow matching models", total=epochs)):
        flow_matching_theta.train()
        flow_matching_x.train()

        loss_list_train = []
        loss_list_val = []

        # -------- TRAINING --------
        for thb, xb, yb in train_loader:
            if simulate is not None:
                xb = simulate(thb)
            xb = xb.to(device)
            thb = thb.to(device)
            yb = yb.to(device)

            # Sample from x|y
            with torch.no_grad():
                source = flow_matching_x.sample_source(yb).to(device)
                x_pred = flow_matching_x.sample(
                    source, yb, device, disable_tqdm=True, only_last=True
                ).to(device)

                # Sample from base distribution
                source_theta = (
                    npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0).to(device)
                )
                source_theta, _ = flow_matching_theta.rescale(source_theta, yb)
                source_x = flow_matching_x.sample_source(yb).to(device)
                thb, yb = flow_matching_theta.rescale(thb, yb)
                xb, _ = flow_matching_x.rescale(xb, yb)

            time_theta = flow_matching_theta.sample_time(thb.shape[0]).to(device)
            theta_t = flow_matching_theta.interpolant(time_theta, source_theta, thb)
            v_theta = flow_matching_theta.target(source_theta, thb)

            time_x = flow_matching_x.sample_time(xb.shape[0]).to(device)
            x_t = flow_matching_x.interpolant(time_x, source_x, xb)
            v_x = flow_matching_x.target(source_x, xb)

            # Forward passes and loss
            pred_theta = flow_matching_theta(theta_t, yb, time_theta)
            pred_x = flow_matching_x(x_t, yb, time_x)

            loss_theta = (pred_theta - v_theta).pow(2).mean()
            loss_x = (pred_x - v_x).pow(2).mean()
            total_loss = lambda_ * loss_theta + (1 - lambda_) * loss_x

            # Backward + step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_matching_x.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(flow_matching_theta.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_list_train.append(total_loss.detach())

        scheduler.step()
        # -------- VALIDATION --------
        flow_matching_theta.eval()
        flow_matching_x.eval()

        with torch.no_grad():
            for thb, xb, yb in val_loader:
                if simulate is not None:
                    xb = simulate(thb)
                xb = xb.to(device)
                thb = thb.to(device)
                yb = yb.to(device)

                # Sample from x|y
                source = flow_matching_x.sample_source(yb).to(device)
                x_pred = flow_matching_x.sample(
                    source, yb, device, disable_tqdm=True, only_last=True
                ).to(device)

                source_theta = (
                    npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0).to(device)
                )
                source_theta, _ = flow_matching_theta.rescale(source_theta, yb)
                source_x = flow_matching_x.sample_source(yb).to(device)

                time_theta = flow_matching_theta.sample_time(thb.shape[0]).to(device)
                thb, yb = flow_matching_theta.rescale(thb, yb)
                theta_t = flow_matching_theta.interpolant(time_theta, source_theta, thb)
                v_theta = flow_matching_theta.target(source_theta, thb)

                time_x = flow_matching_x.sample_time(xb.shape[0]).to(device)
                xb, _ = flow_matching_x.rescale(xb, yb)
                x_t = flow_matching_x.interpolant(time_x, source_x, xb)
                v_x = flow_matching_x.target(source_x, xb)

                pred_theta = flow_matching_theta(theta_t, yb, time_theta)
                pred_x = flow_matching_x(x_t, yb, time_x)

                loss_theta = (pred_theta - v_theta).pow(2).mean()
                loss_x = (pred_x - v_x).pow(2).mean()
                total_loss = lambda_ * loss_theta + (1 - lambda_) * loss_x

                loss_list_val.append(total_loss.detach())

        # Aggregate losses
        loss_value_train = torch.stack(loss_list_train).mean().item()
        loss_value_val = torch.stack(loss_list_val).mean().item()
        bar.set_postfix(train=loss_value_train, val=loss_value_val)

        # Logging
        if logger is not None:
            logger.log_metrics(
                {
                    "train_loss": loss_value_train,
                    "val_loss": loss_value_val,
                    "epoch": epoch,
                },
                log_name=logname,
            )
    torch.save(flow_matching_theta.state_dict(), model_path / f"{logname}_theta.pth")
    torch.save(flow_matching_x.state_dict(), model_path / f"{logname}_x.pth")

    posterior = DualFlowPosteriorEstimator(
        "fm_post_transform",
        base_dist=npe,
        flow_theta=flow_matching_theta,
        flow_x=flow_matching_x,
    )
    return posterior


def list_train_functions():
    return [name for name in globals() if name.startswith("train_")]

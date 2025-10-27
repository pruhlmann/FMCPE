import copy
from pathlib import Path
from typing import Dict, Union

from mlxp.logger import Logger
from torch import Tensor, device

from baselines.posteriors import (
    DirectPosteriorEstimator,
    Estimator,
    Posterior,
)
from baselines.trainers import train_npe
from simulator.base import Simulator


def compute_baselines(
    task: Dict,
    theta_cal: Tensor,
    x_cal: Tensor,
    y_cal: Tensor,
    theta_sim: Tensor,
    x_sim: Tensor,
    simulator: Simulator,
    device: device,
    model_path: Path,
    logger: Logger,
    npe: Estimator,
    max_cal: int,
) -> Dict[str, Posterior]:
    """
    Compute calibration-related baselines.

    Args:
        baselines: List of baselines to compute.
        task: Task to compute the baselines for.
        theta_cal: Calibration parameters.
        x_cal: Calibration outputs (misspecified).
        y_cal: Calibration outputs (observation).
        simulator: Simulator to use for the baselines.
        device: Device to use for the baselines.
        npe_training_params: Training parameters for the NPE.
        npe_kwargs: Additional keyword arguments for the NPE.
        fmconfig: Configuration for the flow matching.

    Returns:
        Dict of the computed calibration baselines.
    """
    posterior_dict = {}
    cal_size = theta_cal.shape[0] if (theta_cal.shape[0] != max_cal) else "ref"

    print("\t##############################")
    print("\tTraining DPE on calibration data")
    estimator_dpe_cal = train_npe(
        task=task,
        theta=theta_cal,
        x=y_cal,
        device=device,
        model_path=model_path,
        logger=logger,
        logname=f"dpe_{cal_size}",
    )
    estimator_dpe_cal.cpu()
    dpe_posterior = DirectPosteriorEstimator("dpe", estimator_dpe_cal)
    posterior_dict["dpe"] = dpe_posterior
    print("\t##############################\n")

    print("\t##############################")
    print("\tTraining MF-NPE")
    mf_npe = train_npe(
        task=task,
        theta=theta_cal,
        x=y_cal,
        device=device,
        model_path=model_path,
        logger=logger,
        logname=f"mf_npe_{cal_size}",
        load=False,
        save=True,
        use_pretrained=True,
    )
    mf_npe.cpu()
    mf_npe_posterior = DirectPosteriorEstimator("dpe", mf_npe)
    posterior_dict["mf_npe"] = mf_npe_posterior
    print("\t##############################\n")

    return posterior_dict

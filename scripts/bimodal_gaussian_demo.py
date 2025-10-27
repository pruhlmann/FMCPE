from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple

from flow_matching.torch_flow import FlowMatching, train_flow_matching


def bimodal_gaussian(
    num_samples: int,
    mean_1: np.ndarray = np.array([0.0, 0.0]),
    mean_2: np.ndarray = np.array([3.0, 0.0]),
    sigma_1: float = 0.2,
    sigma_2: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bimodal gaussian data

    Args:
        num_samples: Number of samples to generate
        mean_1: Mean of the first gaussian
        mean_2: Mean of the second gaussian
        sigma_1: Std of the first gaussian
        sigma_2: Std of the second gaussian

    Returns:
        Data and labels
    """
    x = np.concatenate(
        [
            np.random.randn(num_samples // 2, 2) * sigma_1 + mean_1,
            np.random.randn(num_samples // 2, 2) * sigma_2 + mean_2,
        ],
        axis=0,
    )
    labels = np.concatenate(
        [np.zeros(num_samples // 2), np.ones(num_samples // 2)]
    )
    perm = np.random.permutation(len(x))
    return x[perm], labels[perm]


if __name__ == "__main__":
    shift_y = np.array([0.0, 10.0])
    shift_x = np.array([5.0, 0.0])
    mean_x1 = np.array([0.0, 0.0])
    mean_x2 = mean_x1 + shift_y
    mean_y1 = mean_x1 + shift_x
    mean_y2 = mean_x1 + shift_x + shift_y
    x, _ = bimodal_gaussian(5000, mean_x1, mean_x2)
    y, _ = bimodal_gaussian(5000, mean_y1, mean_y2)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow_matching = FlowMatching(
        conditional=False,
        probability_path="ot2",
        prior="uniform",
        base_dist="none",
        dim=x.shape[1],
        drift={
            "architecture": "resmlp",
            "hidden_dim": [64, 512, 1024, 512, 64],
        },
    )
    flow_matching.to(device)
    model_path = Path(".")

    flow_matching = train_flow_matching(
        flow_matching,
        x,
        y,
        device,
        None,
        model_path,
        save_model=False,
        calibration=False,
        batch_size=256,
        epochs=300,
    )

    traj = (
        flow_matching.sample(
            y.to(device), y.to(device), only_last=False, num_steps=500
        )
        .cpu()
        .detach()
        .numpy()
    )
    print(traj.shape)

    trajx = np.vstack([traj[:500, j, 0] for j in range(traj.shape[1])])
    trajy = np.vstack([traj[:500, j, 1] for j in range(traj.shape[1])])
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(trajx, trajy, c="olive", alpha=0.1)
    plt.scatter(
        traj[:, -1, 0],
        traj[:, -1, 1],
        s=1,
        color="black",
        alpha=0.3,
        label=r"$T(y)$",
    )
    plt.scatter(x[:, 0], x[:, 1], s=1, color="blue", label="x")
    plt.scatter(y[:, 0], y[:, 1], s=1, color="red", label="y")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.hist(traj[:, -1, 0], bins=50, color="black")
    plt.hist(x[:, 0], bins=50, alpha=0.5, color="blue")
    plt.subplot(1, 3, 3)
    plt.hist(traj[:, -1, 1], bins=50, color="black")
    plt.hist(x[:, 1], bins=50, alpha=0.5, color="blue")
    plt.savefig("bimodal_gaussian_demo.pdf")

import json
import numpy as np
import pickle
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import mlxp
import torch

from simulator import get_simulator
from simulator.base import generate_calibration_dataset
from utils.misc import rescale
from utils.plots import (
    pairplot_sbi_diag,
    plot_boxplots,
    plot_losses_baselines,
    scatter_plots_by_cal_size,
)

NAMES = {
    "dpe": r"NPE",
    "fm_post_transform": r"Ours",
    "npe": r"$q(\mathbf{\theta} | \mathbf{x})$",
    "mf_npe": r"MF-NPE",
    "npe_fmpe": r"FMPE",
}

TASK_NAMES = {
    "high_dim_gaussian": "Gaussian",
    "pendulum": "Pendulum",
    "wind_tunnel": "Wind Tunnel",
    "light_tunnel": "Light Tunnel",
}

METRICS_NAMES = {
    "c2st": r"C2ST - $\theta$",
    "joint_c2st": r"$j$C2ST",
    "wasserstein": r"$W_2$ - $\theta$",
    "joint_wasserstein": r"$W_2$",
    "mse": "MSE",
}


def plot_all(save_root: Path, log_dir: Path):
    """
    Plot all the results for the given log directory.

    Args:
        save_root (Path): The root directory to save the plots.
        log_dir (Path): The directory containing the mlxp logs (the database.json file).
        task_to_plot (List[str]): List of tasks to plot.
        dataset (str): The dataset to use for plotting (test or calib).
    """
    save_dir = save_root / log_dir.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    reader = mlxp.Reader(log_dir, refresh=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # model loss
    loss_path = log_dir / "losses"
    save_path = save_dir / "losses"
    save_path.mkdir(parents=True, exist_ok=True)
    try:
        for loss_file in loss_path.iterdir():
            if loss_file.suffix != ".npz":
                continue
            loss_dict = np.load(loss_file, allow_pickle=True)
            task, method, ncal = loss_file.stem.rsplit("_", 2)
            if method != "combined":
                continue
            save_dir_task = save_path / task
            save_dir_task.mkdir(parents=True, exist_ok=True)
            plot_losses_baselines(loss_dict, NAMES, save_dir_task, name=f"loss_{method}_{ncal}.pdf")
    except FileNotFoundError:
        print("No losses found, skipping plot")
    # lpp by ntraj
    # plot_losses_ntraj(log_dir / "ntraj", BASELINE_NAMES, TASK_NAMES, save_dir)
    # loss by metrics

    # Seeded plots
    save_path = save_dir / "seeded"
    save_path.mkdir(parents=True, exist_ok=True)
    task_data = {}
    for task in TASK_NAMES.keys():
        if task in ["js", "light_tunnel", "pure_gaussian", "adaptive_gaussian"]:
            continue
        try:
            d = json.load(open(log_dir / "seeded" / f"metrics_{task}.json", "r"))
            key = next(iter(d.keys()))
            task_data[key] = d[key]
        except FileNotFoundError:
            pass
    try:
        seeded_metrics = json.load(open(log_dir / "seeded" / "metrics.json", "r"))
        plot_boxplots(
            seeded_metrics,
            NAMES,
            METRICS_NAMES,
            TASK_NAMES,
            save_path,
        )
    except FileNotFoundError:
        print("No metrics found for seeded, skipping plot")

    # theta scatter plots
    for task in TASK_NAMES.keys():
        if task not in ["pendulum"]:
            continue
        save_path = save_dir / task
        save_path.mkdir(parents=True, exist_ok=True)
        make_pair_plots(task, reader, save_path, log_dir)


def make_pair_plots(task: str, reader: mlxp.Reader, save_path: Path, logdir: Path):
    print("Making pair plots for task:", task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load or generate GT data and sample from posterior
    samples_path = logdir / "seeded" / "samples" / task
    samples_path.mkdir(parents=True, exist_ok=True)
    # if directory is empty return
    if not any(samples_path.iterdir()):
        print(f"No samples found for task {task}, skipping pair plots")
        return
    theta_dict = {}
    true_data = None

    # --- Load simulator ---
    try:
        with open(logdir / "models" / f"{task}" / "simulator.json", "r") as f:
            config_simulator = json.load(f)
        simulator = get_simulator(task, **config_simulator["params"])
    except FileNotFoundError:
        print(f"No simulator found for task {task}, skipping pair plots")
        return None

    # iterate over seeds
    for seed_dir in [p for p in samples_path.iterdir() if p.is_dir()]:
        if seed_dir.stem != "seed_8":
            continue
        print(f"\tProcessing seed directory: {seed_dir}")
        temps_path = logdir / "plots" / "samples" / task / f"{seed_dir.stem}"
        temps_path.mkdir(parents=True, exist_ok=True)
        res = collect_theta_predictions(
            reader,
            logdir,
            task,
            theta_dict,
            device,
            temps_path / "theta_data.pkl",
            seed=int(seed_dir.stem.split("_")[-1]),
        )
        if res is None:
            print(f"No results found for task {task}, skipping pair plots")
            return
        test_data, theta_dict = res
        # Generate ground truth if possible
        # try:
        #     true_data = simulator.sample_reference_posterior(
        #         2000, test_data["y"], misspecified=False
        #     )
        # except NotImplementedError:
        #     print(f"Simulator for task {task} does not implement reference posterior sampling")

        # --- Load simulator ---
        with open(logdir / "models" / f"{task}" / "simulator.json", "r") as f:
            config_simulator = json.load(f)
        simulator = get_simulator(task, **config_simulator["params"])

        seed_save_path = save_path / seed_dir.stem
        seed_save_path.mkdir(parents=True, exist_ok=True)
        # Make pair plots
        if task != "wind_tunnel":
            scatter_plots_by_cal_size(
                test_data,
                theta_dict,
                simulator,
                NAMES=NAMES,
                save_path=seed_save_path,
                max_samples=2000,
                max_obs=3,
                title=f"{TASK_NAMES.get(task, task)}",
                type_plot="kde",
            )
        if task == "wind_tunnel":
            pairplot_sbi_diag(
                test_data,
                theta_dict,
                simulator,
                NAMES=NAMES,
                save_path=seed_save_path,
                max_samples=1000,
                max_obs=3,
                title=f"{TASK_NAMES.get(task, task)}",
            )


def plot_debug(results_simulations, results_calibration, log_dir, save_dir):
    """
    Debug function to plot the results of the simulation and calibration runs.
    """
    for key_task in results_simulations.group_vals:
        if key_task[0] != "pendulum":
            continue
        save_path = save_dir / key_task[0]
        save_path.mkdir(parents=True, exist_ok=True)

        res_task_sim = results_simulations[key_task][-1]
        test_data = torch.load(log_dir / "ncal" / "samples" / key_task[0] / "test_data.pt")
        models = res_task_sim["artifact.pickle."]["models"].load()
        posterior = models["fm_on_x_y_sim"]
        source = posterior.flow_matching.sample_source(test_data["y"][:6], 1).squeeze(0)
        x_samples = posterior.flow_matching.sample(source, test_data["y"][:6], only_last=True)
        source = rescale(source, test_data["scales"]["x_mean"], test_data["scales"]["x_std"])
        x = rescale(x_samples, test_data["scales"]["x_mean"], test_data["scales"]["x_std"])
        y = rescale(test_data["y"][:6], test_data["scales"]["y_mean"], test_data["scales"]["y_std"])
        x_true = rescale(
            test_data["x"][:6], test_data["scales"]["x_mean"], test_data["scales"]["x_std"]
        )
        nrows = 2
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.plot(x_true[i, :], label=r"$x$", color="black")
            plt.plot(x[i, :], label=r"$x|y$", color="red")
            plt.plot(source[i, :], label="Source", color="blue", alpha=0.2)
            plt.plot(y[i, :], label=r"$y$", color="green")
        plt.legend()
        plt.savefig(save_path / "fm_on_x_cond.pdf")
        plt.close()

        theta_pred = posterior.sample(test_data["y"][:6], 100).reshape(6 * 100, -1)
        theta_true = test_data["theta"][:6]
        theta_pred = rescale(
            theta_pred, test_data["scales"]["theta_mean"], test_data["scales"]["theta_std"]
        )
        theta_true = rescale(
            theta_true, test_data["scales"]["theta_mean"], test_data["scales"]["theta_std"]
        )
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.scatter(theta_true[i, 0], theta_true[i, 1], label=r"$\theta$", color="black", s=10)
            plt.scatter(
                theta_pred[i : i + 100, 0],
                theta_pred[i : i + 100, 1],
                label=r"$\theta|y$",
                color="red",
                s=2,
            )
        plt.legend()
        plt.savefig(save_path / "fm_on_theta_cond.pdf")
        plt.close()


def collect_theta_predictions(
    reader, logdir, task, theta_dict, device, temp_path, nsamples=2000, nobs=10, seed=0
):
    """
    Collect posterior samples for methods and baselines for a given task.
    Saves/loads intermediate results to avoid recomputation.

    Args:
        reader: An object with `.filter()` and `.groupby()` methods for querying results.
        logdir (Path): Path to the logging directory containing models.
        task (str): Task name to extract results for.
        theta_dict (dict): Dictionary where theta predictions will be stored.
        device (torch.device): Torch device for sampling.
        temp_path (Path): Path to cache file for saving/loading results.

    Returns:
        tuple | None: (test_data, theta_dict) if successful, None if task results are missing.
                      test_data = {"theta": th, "x": x, "y": y}
    """
    # --- Try loading from cache ---
    if temp_path.exists():
        with open(temp_path, "rb") as f:
            cached = pickle.load(f)
        return cached  # (test_data, theta_dict)

    # --- Collecting runs where methods were trained ---
    query = f"info.status == 'COMPLETE' & config.seed == {seed}"
    results = reader.filter(query_string=query)
    results_methods = results.groupby("config.task.name").toPandas(lazy=False)

    # --- Collecting runs where baselines were trained ---
    query = f"info.status == 'COMPLETE' & config.train_baselines == True & config.seed == {seed}"
    results_baselines = (
        reader.filter(query_string=query).groupby("config.task.name").toPandas(lazy=False)
    )

    try:
        # Select latest run for methods
        last_run_id = max([index for index in results_methods.loc[task].index])
        methods_df = results_methods.loc[(task, last_run_id)]

        # Select latest run for baselines
        last_run_id = max([index for index in results_baselines.loc[task].index])
        baselines_df = results_baselines.loc[(task, last_run_id)]
    except KeyError:
        return None

    # --- Load simulator ---
    try:
        with open(logdir / "models" / f"{task}" / "simulator.json", "r") as f:
            config_simulator = json.load(f)
        simulator = get_simulator(task, **config_simulator["params"])
    except FileNotFoundError:
        print(f"No simulator found for task {task}, skipping pair plots")
        return None

    torch.manual_seed(33575)  # Fixed seed for reproducibility
    th, x, y = generate_calibration_dataset(
        simulator, nobs, generation=baselines_df["config.task.generation"]
    )
    test_data = {"theta": th, "x": x, "y": y}

    # --- Load methods and baselines ---
    methods = methods_df["artifact.pickle."]["methods"].load()
    baselines = baselines_df["artifact.pickle."]["baselines"].load()

    # --- Process methods ---
    for name, method in methods.items():
        for ncal, posterior in method.items():
            theta_pred = posterior.sample(y, nsamples, device, batch_size=500)
            theta_dict.setdefault(ncal, {})[name] = theta_pred.cpu()

    # --- Process baselines ---
    for name, baseline in baselines.items():
        if name == "fmpe":
            continue
        for ncal, posterior in baseline.items():
            theta_pred = posterior.sample(y, nsamples, device, batch_size=500)
            theta_dict.setdefault(ncal, {})[name] = theta_pred.cpu()

    # --- Save to cache ---
    with open(temp_path, "wb") as f:
        pickle.dump((test_data, theta_dict), f)

    return test_data, theta_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_root", type=str, default="plots")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    plot_all(Path(args.save_root), Path(args.log_dir))

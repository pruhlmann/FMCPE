import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import mlxp
import numpy as np
import torch
from pandas import DataFrame
from xarray.core.duck_array_ops import first

from baselines.posteriors import Posterior
from simulator import get_simulator
from simulator.base import generate_calibration_dataset
from utils.log import Timer
from utils.misc import convert_tensors_to_float, get_last_logid, get_params, save_or_merge_metrics

METHOD_NAMES = {
    "fm_post_transform": r"$u_t(\mathbf{\theta}_t, \mathbf{y}) + v_t(\mathbf{x}_t, \mathbf{y})$",
}

BASELINE_NAMES = {
    "dpe": r"DPE $q(\mathbf{\theta} | \mathbf{y})$",
    "mf_npe": r"MF-NPE (fine-tuned)",
}


def compute_all_metrics(
    reader: mlxp.Reader,
    query_args: str,
    log_dir: Path,
    task_to_plot,
    metrics_to_plot,
    num_samples_test=2000,
    compute_c2st: bool = False,
    compute_stein_discrepancy: bool = False,
    compute_wasserstein: bool = False,
    compute_mse: bool = False,
    recompute_baselines: bool = False,
    recompute_methods: bool = False,
    batch_size: int = 500,
):
    # Collecting runs where methods were trained
    if query_args != "":
        query = "info.status == 'COMPLETE'" + " & " + query_args
    else:
        query = "info.status == 'COMPLETE'"
    print("Querying runs with:", query)
    try:
        reader_methods = reader.filter(query_string=query)
        results_methods = reader_methods.groupby("config.task.name").toPandas(lazy=False)
    except mlxp.errors.InvalidKeyError as e:
        raise KeyError(
            f"No runs found with query {query}. Please check the query arguments."
        ) from e

    # Collecting runs where baselines were trained
    if query_args != "":
        query = "info.status == 'COMPLETE' & config.train_baselines == True" + " & " + query_args
    else:
        query = "info.status == 'COMPLETE' & config.train_baselines == True"
    reader_baselines = reader.filter(query_string=query)
    results_baselines = reader_baselines.groupby("config.task.name").toPandas(lazy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Loss for each ncal
    if "ncal" in metrics_to_plot:
        print(
            f"Computing ncal metrics for task {task_to_plot}, c2st: {compute_c2st}, wasserstein: {compute_wasserstein}, mse: {compute_mse}"
        )
        compute_metrics_ncal(
            results_methods,
            results_baselines,
            task_to_plot,
            device,
            log_dir,
            num_samples_test,
            compute_c2st,
            compute_wasserstein,
            compute_mse,
            recompute_baselines=recompute_baselines,
            recompute_methods=recompute_methods,
            batch_size=batch_size,
        )

    if "conditional" in metrics_to_plot:
        print("############################################")
        print("Plotting conditional metrics for each model (when available)")
        conditional_metrics(
            results_methods,
            results_baselines,  # type: ignore[assignment]
            task_to_plot,
            device,
            log_dir,
            num_samples_test,
            n_obs=10,
            compute_c2st=compute_c2st,
            compute_stein_discrepancy=compute_stein_discrepancy,
            compute_wasserstein=compute_wasserstein,
            recompute_baselines=recompute_baselines,
            recompute_methods=recompute_methods,
        )

    # Losses for each model
    if "loss" in metrics_to_plot:
        print("############################################")
        print("Plotting losses for each model")
        save_path = log_dir / "losses"
        save_path.mkdir(parents=True, exist_ok=True)
        for task in results_baselines.index.get_level_values(0).unique():
            if task not in task_to_plot:
                continue
            last_run_id = max([index for index in results_baselines.loc[task].index])
            print("baseline", task)
            res_task = results_baselines.loc[(task, last_run_id)]
            for ncal in res_task["config.task.num_cal"] + ["ref"]:
                process_and_save_losses(
                    res_task, BASELINE_NAMES, save_path, task + f"_baselines_{ncal}", ncal
                )

        for task in results_methods.index.get_level_values(0).unique():
            if task not in task_to_plot:
                continue
            last_run_id = max([index for index in results_methods.loc[task].index])
            print(last_run_id, "methods", task)
            res_task = results_methods.loc[(task, last_run_id)]
            for ncal in res_task["config.task.num_cal"] + ["ref"]:
                process_and_save_losses(
                    res_task, METHOD_NAMES, save_path, task + f"_methods_{ncal}", ncal
                )
        # Combine losses between baselines and methods
        for loss_file in save_path.glob("*.npz"):
            if "baselines" in loss_file.name:
                # combine with methods
                method_file = loss_file.parent / loss_file.name.replace("baselines", "methods")
                if method_file.exists():
                    print("Combining losses", loss_file.name, "and", method_file.name)
                    baselines = np.load(loss_file)
                    methods = np.load(method_file)
                    combined = {
                        **baselines,
                        **methods,
                    }
                    combined_file = save_path / loss_file.name.replace("baselines", "combined")
                    np.savez(combined_file, **combined)
                    print("Saved combined losses to", combined_file)
    if "seeded" in metrics_to_plot:
        print("############################################")
        print("Plotting seeded metrics for each model")
        compute_metrics_seeded(
            reader_methods.groupby(["config.task.name", "config.seed"]).toPandas(lazy=False),
            reader_baselines.groupby(["config.task.name", "config.seed"]).toPandas(lazy=False),
            task_to_plot,
            device,
            log_dir,
            num_samples_test,
            compute_c2st=compute_c2st,
            compute_wasserstein=compute_wasserstein,
            compute_mse=compute_mse,
            recompute_baselines=recompute_baselines,
            recompute_methods=recompute_methods,
        )


def conditional_metrics(
    methods_df: DataFrame,
    baselines_df: DataFrame,
    task_to_plot: List[str],
    device: torch.device,
    log_dir: Path,
    num_samples_test: int = 2000,
    n_obs: int = 10,
    compute_c2st: bool = True,
    compute_stein_discrepancy: bool = True,
    compute_wasserstein: bool = True,
    recompute_baselines: bool = False,
    recompute_methods: bool = False,
):
    """
    Compute conditional metrics for each ncal and save them.
    """
    conditional_metrics = {}
    root_dir = log_dir / "conditional"
    root_dir.mkdir(parents=True, exist_ok=True)
    save_path = root_dir / "metrics.json"
    for task in methods_df.index.get_level_values(0).unique():
        if task not in task_to_plot:
            continue
        print("############################################")
        print("Task:", task)
        last_run_id = max([index for index in methods_df.loc[task].index])
        if last_run_id > 1:
            print(f"Multiple runs found for task {task}, using last run:", last_run_id)

        if task not in [
            "high_dim_gaussian",
        ]:
            print("Skipping conditional metrics for task", task)
            continue

        # Load task related results
        res_task = methods_df.loc[(task, last_run_id)]
        last_run_id = max([index for index in baselines_df.loc[task].index])
        if last_run_id > 1:
            print(f"Multiple runs found for task {task}, using last run:", last_run_id)
        res_baselines_task = baselines_df.loc[(task, last_run_id)]
        assert res_task["config.seed"] == res_baselines_task["config.seed"], (
            "Seeds do not match between methods and baselines"
        )

        samples_path = root_dir / "samples" / f"{task}"
        samples_path.mkdir(parents=True, exist_ok=True)
        conditional_metrics_task = evaluate_conditional_models(
            res_baselines_task,
            res_task,
            device,
            samples_path,
            num_samples_test=num_samples_test,
            n_obs=n_obs,
            compute_c2st=compute_c2st,
            compute_stein_discrepancy=compute_stein_discrepancy,
            compute_wasserstein=compute_wasserstein,
            recompute_baselines=recompute_baselines,
            recompute_methods=recompute_methods,
        )
        conditional_metrics[task] = conditional_metrics_task
    save_or_merge_metrics(conditional_metrics, save_path)
    print("Saved conditional metrics for task in", save_path)


def compute_metrics_ncal(
    methods_df: DataFrame,
    baselines_df: DataFrame,
    task_to_plot: List[str],
    device: torch.device,
    log_dir: Path,
    num_samples_test: int = 2000,
    compute_c2st: bool = True,
    compute_wasserstein: bool = True,
    compute_mse: bool = False,
    recompute_baselines: bool = False,
    recompute_methods: bool = False,
    batch_size: int = 500,
):
    """
    Compute metrics for each ncal and save them.

    Args:
        res_task_ncal (GroupedDataFrame): Grouped data (trained en calibration) for each task.
        res_baselines (GroupedDataFrame): Grouped data (trained on simulation) for each task.
        task_to_plot (List[str]): List of tasks to plot.
        dataset (str): Dataset to evaluate the models on (test or cal).
        device (torch.device): Device to use for the evaluation.
        batch_size (int): Batch size for the evaluation.
        n_traj (int): Number of trajectories to use for the evaluation.
        log_dir (Path): Directory where the metrics will be saved.
    """
    metrics = {}
    root_dir = log_dir / "ncal"
    root_dir.mkdir(parents=True, exist_ok=True)
    save_path = root_dir / "metrics.json"
    for task in methods_df.index.get_level_values(0).unique():
        if task not in task_to_plot:
            continue
        print("############################################")
        print("Task:", task)
        last_run_id = max([index for index in methods_df.loc[task].index])
        if last_run_id > 1:
            print(f"Multiple runs found for task {task}, using last run:", last_run_id)

        samples_path = root_dir / "samples" / f"{task}"
        samples_path.mkdir(parents=True, exist_ok=True)

        # Load task related results
        res_task = methods_df.loc[(task, last_run_id)]
        last_run_id = max([index for index in baselines_df.loc[task].index])
        if last_run_id > 1:
            print(f"Multiple runs found for task {task}, using last run:", last_run_id)
        res_baselines_task = baselines_df.loc[(task, last_run_id)]
        assert res_task["config.seed"] == res_baselines_task["config.seed"], (
            "Seeds do not match between methods and baselines"
        )

        metrics_task = evaluate_models(
            res_baselines_task,
            res_task,
            device,
            samples_path,
            compute_c2st=compute_c2st,
            compute_wasserstein=compute_wasserstein,
            compute_mse=compute_mse,
            num_samples_test=num_samples_test,
            recompute_baselines=recompute_baselines,
            recompute_methods=recompute_methods,
            batch_size=batch_size,
        )
        metrics[task] = convert_tensors_to_float(metrics_task)

    # Save metrics
    save_or_merge_metrics(metrics, save_path)
    print("Saved metrics in", save_path)


def compute_metrics_seeded(
    methods_df: DataFrame,
    baselines_df: DataFrame,
    task_to_plot: List[str],
    device: torch.device,
    log_dir: Path,
    num_samples_test: int = 2000,
    compute_c2st: bool = True,
    compute_wasserstein: bool = True,
    compute_mse: bool = False,
    recompute_baselines: bool = False,
    recompute_methods: bool = False,
):
    metrics = {}
    metrics_agg = {}
    root_dir = log_dir / "seeded"
    root_dir.mkdir(parents=True, exist_ok=True)
    save_path = root_dir / "metrics.json"
    for task in methods_df.index.get_level_values(0).unique():
        if task not in task_to_plot:
            continue
        print("############################################")
        print("Task:", task)

        samples_path = root_dir / "samples" / f"{task}"
        samples_path.mkdir(parents=True, exist_ok=True)

        metrics[task] = {}
        for seed in methods_df.index.get_level_values(1).unique():
            try:
                # Load task related results (methods)
                last_run_id = max([index for index in methods_df.loc[task, seed].index])
            except KeyError:
                print(f"No runs found for task {task} and seed {seed}, skipping")
                continue
            if last_run_id > 1:
                print(
                    f"Multiple runs found for task {task} and seed {seed}, using last run:",
                    last_run_id,
                )
            res_task = methods_df.loc[(task, seed, last_run_id)]

            # Load task related results (baselines)
            last_run_id = max([index for index in baselines_df.loc[task, seed].index])
            if last_run_id > 1:
                print(
                    f"Multiple runs found for task {task} and seed {seed}, using last run:",
                    last_run_id,
                )
            res_baselines_task = baselines_df.loc[(task, seed, last_run_id)]
            print("Task", task, "seed", seed)

            seeded_path = samples_path / f"seed_{seed}"
            seeded_path.mkdir(parents=True, exist_ok=True)

            metrics_task = evaluate_models(
                res_baselines_task,
                res_task,
                device,
                seeded_path,
                compute_c2st=compute_c2st,
                compute_wasserstein=compute_wasserstein,
                compute_mse=compute_mse,
                num_samples_test=num_samples_test,
                recompute_baselines=recompute_baselines,
                recompute_methods=recompute_methods,
            )
            metrics[task][seed] = convert_tensors_to_float(metrics_task)
            metrics_task = convert_tensors_to_float(metrics_task)
            save_or_merge_metrics(metrics_task, root_dir / f"metrics_{task}.json")
            with (root_dir / f"metrics_{task}_{seed}.json").open("w") as f:
                json.dump(metrics_task, f, indent=2)

        # Save intermediate metrics
        save_or_merge_metrics(metrics, root_dir / f"metrics_{task}.json")

        # Aggregate metrics across seeds
        first_seed = next(iter(metrics[task].keys()))
        metrics_agg[task] = {}
        metrics_agg[task]["methods"] = {}
        for name in metrics[task][first_seed]["methods"].keys():
            metrics_agg[task]["methods"][name] = {}
            for ncal in metrics[task][first_seed]["methods"][name].keys():
                metrics_agg[task]["methods"][name][ncal] = {}
                for metric_name in metrics[task][first_seed]["methods"][name][ncal].keys():
                    values = [
                        metrics[task][seed]["methods"][name][ncal][metric_name]
                        for seed in metrics[task].keys()
                    ]
                    metrics_agg[task]["methods"][name][ncal][metric_name] = values
        metrics_agg[task]["baselines"] = {}
        for name in metrics[task][first_seed]["baselines"].keys():
            metrics_agg[task]["baselines"][name] = {}
            for ncal in metrics[task][first_seed]["baselines"][name].keys():
                metrics_agg[task]["baselines"][name][ncal] = {}
                for metric_name in metrics[task][first_seed]["baselines"][name][ncal].keys():
                    values = [
                        metrics[task][seed]["baselines"][name][ncal][metric_name]
                        for seed in metrics[task].keys()
                    ]
                    metrics_agg[task]["baselines"][name][ncal][metric_name] = values

    # Save metrics
    save_or_merge_metrics(metrics_agg, save_path)
    print("Saved metrics in", save_path)


def evaluate_models(
    baselines_df,
    methods_df,
    device,
    sample_path: Path,
    compute_c2st: bool = True,
    compute_wasserstein: bool = True,
    compute_mse: bool = False,
    num_samples_test: int = 1000,
    recompute_baselines: bool = False,
    recompute_methods: bool = False,
    batch_size: int = 100,
) -> dict:
    # Get simulator parameters
    simulator_params = get_params(baselines_df, "config.task.simulator.params")
    print("Simulator parameters:", simulator_params)
    simulator = get_simulator(baselines_df["config.task.name"], **simulator_params)

    # Generate test data
    torch.manual_seed(351054)
    theta_o, x_o, y_o = generate_calibration_dataset(
        simulator,
        n=num_samples_test,
        generation=baselines_df["config.task.generation"],
    )
    test_data = {
        "theta": theta_o[:num_samples_test, :],
        "x": x_o[:num_samples_test, :],
        "y": y_o[:num_samples_test, :],
    }
    torch.save(
        test_data,
        sample_path / "test_data.pt",
    )
    timer = Timer()

    # Methods evaluation
    methods = methods_df["artifact.pickle."]["methods"].load()
    metrics_methods = {}

    if recompute_methods:
        print("Evaluating methods")
        for name, method in methods.items():
            metrics_methods[name] = {}
            for ncal, posterior in method.items():
                print(f"Evaluating method {name} with ncal={ncal}")
                with timer:
                    save_path = sample_path / name / str(ncal)
                    save_path.mkdir(parents=True, exist_ok=True)
                    metrics = posterior.evaluate_metrics(
                        test_data["y"],
                        test_data["theta"],
                        device,
                        simulator=simulator,
                        save_path=save_path,  # Uncomment to save/load samples
                        compute_c2st=compute_c2st,
                        compute_wasserstein=compute_wasserstein,
                        compute_mse=compute_mse,
                        batch_size=batch_size,
                    )
                    posterior.to("cpu")  # Ensure CPU
                    metrics_methods[name][ncal] = metrics

    # load model for baselines
    baselines = baselines_df["artifact.pickle."]["baselines"].load()
    metrics_baselines = {}

    if recompute_baselines:
        print("Evaluating baselines")
        for name, baseline in baselines.items():
            if name == "fmpe":
                continue
            metrics_baselines[name] = {}
            for ncal, posterior in baseline.items():
                print(f"Evaluating baseline {name} with ncal={ncal}")
                with timer:
                    save_path = sample_path / name / str(ncal)
                    save_path.mkdir(parents=True, exist_ok=True)
                    metrics = posterior.evaluate_metrics(
                        test_data["y"],
                        test_data["theta"],
                        device,
                        simulator=simulator,
                        save_path=save_path,  # Uncomment to save/load samples
                        compute_c2st=compute_c2st,
                        compute_wasserstein=compute_wasserstein,
                        compute_mse=compute_mse,
                        batch_size=batch_size,
                    )
                    posterior.to("cpu")
                    metrics_baselines[name][ncal] = metrics

    # Save metrics
    combined_metrics = {
        "methods": metrics_methods,
        "baselines": metrics_baselines,
    }
    return combined_metrics


def evaluate_conditional_models(
    baselines_df,
    methods_df,
    device,
    sample_path: Path,
    num_samples_test: int = 1000,
    n_obs: int = 10,
    compute_c2st: bool = True,
    compute_stein_discrepancy: bool = False,
    compute_wasserstein: bool = False,
    recompute_baselines: bool = False,
    recompute_methods: bool = False,
):
    # Get simulator parameters
    simulator_params = get_params(baselines_df, "config.task.simulator.params")
    print("Simulator parameters:", simulator_params)
    simulator = get_simulator(baselines_df["config.task.name"], **simulator_params)

    # Generate test data
    torch.manual_seed(33575)
    theta_o, x_o, y_o = generate_calibration_dataset(
        simulator,
        n=n_obs,
        generation=baselines_df["config.task.generation"],
    )
    test_data = {
        "theta": theta_o,
        "x": x_o,
        "y": y_o,
    }
    torch.save(
        test_data,
        sample_path / "test_data.pt",
    )

    yobs = test_data["y"]  # Shape (n_obs, obs_dim)
    xobs = test_data["x"]  # Shape (n_obs, obs_dim)

    # Generating reference data
    pdf_available = True
    try:
        p_theta_y = simulator.get_posterior_dist(yobs, misspecified=False)
        p_theta_x = simulator.get_posterior_dist(xobs, misspecified=True)
        # p_x_y = simulator.get_denoise_dist(yobs)
        theta_true_y = p_theta_y.sample((num_samples_test,))  # shape (num_samples_test, n_obs, dim)
        theta_true_x = p_theta_x.sample((num_samples_test,))  # shape (num_samples_test, n_obs, dim)
    except TypeError:
        # set to none
        pdf_available = False
        p_theta_y = None
        p_theta_x = None
        theta_true_y = simulator.sample_reference_posterior(
            num_samples_test,
            yobs,
            misspecified=False,
        )
        theta_true_x = None
        x_true_y = None
    true_data = {
        "theta_true_y": theta_true_y,
        "theta_true_x": theta_true_x,
        "x_true_y": x_true_y,
    }
    torch.save(true_data, sample_path / "true_data.pt")

    timer = Timer()

    # Methods evaluation
    methods = methods_df["artifact.pickle."]["methods"].load()
    metrics_methods = {}

    if recompute_methods:
        print("Evaluating methods")
        for name, method in methods.items():
            metrics_methods[name] = {}
            for ncal, posterior in method.items():
                print(f"Evaluating method {name} with ncal={ncal}")
                with timer:
                    save_path = sample_path / name / str(ncal)
                    save_path.mkdir(parents=True, exist_ok=True)
                    metrics = posterior.evaluate_conditional_metrics(
                        yobs,
                        theta_true_y,
                        device,
                        (lambda y: simulator.get_posterior_dist(y, misspecified=False))
                        if pdf_available
                        else None,
                        save_path=save_path,
                        compute_c2st=compute_c2st,
                        compute_stein_discrepancy=compute_stein_discrepancy,
                        compute_wasserstein=compute_wasserstein,
                    )
                    posterior.to("cpu")  # Move back to CPU after evaluation
                    metrics_methods[name][ncal] = metrics

    # Baselines evaluation
    baselines = baselines_df["artifact.pickle."]["baselines"].load()
    metrics_baselines = {}

    if recompute_baselines:
        print("Evaluating baselines")
        for name, baseline in baselines.items():
            metrics_baselines[name] = {}
            for ncal, posterior in baseline.items():
                print(f"Evaluating baseline {name} with ncal={ncal}")
                with timer:
                    save_path = sample_path / name / str(ncal)
                    save_path.mkdir(parents=True, exist_ok=True)
                    metrics = posterior.evaluate_conditional_metrics(
                        yobs,
                        theta_true_y,
                        device,
                        (lambda y: simulator.get_posterior_dist(y, misspecified=False))
                        if pdf_available
                        else None,
                        save_path=save_path,
                        compute_c2st=compute_c2st,
                        compute_stein_discrepancy=compute_stein_discrepancy,
                        compute_wasserstein=compute_wasserstein,
                    )
                    posterior.to("cpu")
                    metrics_baselines[name][ncal] = metrics

    metrics = {
        "methods": metrics_methods,
        "baselines": metrics_baselines,
    }
    return metrics


def process_and_save_losses(group_by_task, baseline_names, save_dir, name, ncal: int):
    """
    Process losses for each baseline and save them.

    Args:
        group_by_task (dict): Dictionary containing task data grouped by key.
        baseline_names (dict): Dictionary of baseline names.
        save_dir (Path): Directory where the loss dictionary will be saved.
        name (str): Name of the file for saving the loss dictionary.

    Returns:
        None
    """
    output_file = save_dir / f"{name}.npz"
    try:
        loss_dict = dict(np.load(output_file))
    except FileNotFoundError:
        # Initialize the loss dictionary
        loss_dict = {k: np.zeros(1) for k in baseline_names.keys()}

    # Process each baseline
    for baseline in baseline_names.keys():
        try:
            # Extract epochs, training loss, and validation loss
            epochs = group_by_task[f"{baseline}_{ncal}.epoch"]
            train_loss = group_by_task[f"{baseline}_{ncal}.train_loss"]
            val_loss = group_by_task[f"{baseline}_{ncal}.val_loss"]

            # Store losses in the dictionary
            loss_dict[baseline] = np.array((epochs, train_loss, val_loss))
        except KeyError:
            # Handle missing data for the baseline
            print(f"No losses found for {baseline}")
        try:
            # Extract epochs, training loss, and validation loss
            epochs = group_by_task[f"{baseline}_{ncal}_x.epoch"]
            train_loss = group_by_task[f"{baseline}_{ncal}_x.train_loss"]
            val_loss = group_by_task[f"{baseline}_{ncal}_x.val_loss"]

            # Store losses in the dictionary
            loss_dict[baseline + "_x"] = np.array((epochs, train_loss, val_loss))
        except KeyError:
            # Handle missing data for the baseline
            print(f"No losses found for {baseline}")
        try:
            # Extract epochs, training loss, and validation loss
            epochs = group_by_task[f"pretrain_{baseline}.epoch"]
            train_loss = group_by_task[f"pretrain_{baseline}_{ncal}.train_loss"]
            val_loss = group_by_task[f"pretrain_{baseline}_{ncal}.val_loss"]

            # Store losses in the dictionary
            loss_dict[baseline + "_pretrained"] = np.array((epochs, train_loss, val_loss))
        except KeyError:
            # Handle missing data for the baseline
            print(f"No losses found for {baseline}")

    try:
        # Extract epochs, training loss, and validation loss
        epochs = group_by_task["npe.epoch"]
        train_loss = group_by_task[f"npe.train_loss"]
        val_loss = group_by_task[f"npe.val_loss"]

        # Store losses in the dictionary
        loss_dict["npe"] = np.array((epochs, train_loss, val_loss))
    except KeyError:
        # Handle missing data for the baseline
        print("No losses found for npe")

    try:
        # Extract epochs, training loss, and validation loss
        epochs = group_by_task["npe_fmpe.epoch"]
        train_loss = group_by_task["npe_fmpe.train_loss"]
        val_loss = group_by_task["npe_fmpe.val_loss"]

        # Store losses in the dictionary
        loss_dict["npe_fmpe"] = np.array((epochs, train_loss, val_loss))
    except KeyError:
        # Handle missing data for the baseline
        print("No losses found for npe_fmpe")

    # Save the loss dictionary
    output_file = save_dir / f"{name}.npz"
    np.savez(output_file, **loss_dict)
    print(f"Loss dictionary saved to {output_file}")


def main(reader, logdir, args):
    compute_all_metrics(
        reader,
        args.query_args,
        logdir,
        args.task,
        args.metrics,
        args.num_test,
        args.c2st,
        args.sd,
        args.wasserstein,
        args.mse,
        args.recompute_baselines,
        args.recompute_methods,
        args.batch_size,
    )


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--logdir", type=str)
    args.add_argument("--batch_size", type=int, default=500)
    args.add_argument("--metrics", type=str, nargs="+", default=["ncal", "loss", "ntraj"])
    args.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=["adaptive_gaussian", "pure_gaussian", "pendulum", "gaussian"],
    )
    args.add_argument("--num_test", type=int, default=2000)
    args.add_argument("--c2st", action="store_true")
    args.add_argument("--sd", action="store_true")
    args.add_argument("--wasserstein", action="store_true")
    args.add_argument("--mse", action="store_true")
    args.add_argument("--recompute_baselines", action="store_true")
    args.add_argument("--recompute_methods", action="store_true")
    args.add_argument("--query_args", type=str, default="")
    params = args.parse_args()
    print(params.query_args)

    parent_dir = Path(params.logdir)
    reader = mlxp.Reader(parent_dir, refresh=True)

    main(reader, parent_dir, params)

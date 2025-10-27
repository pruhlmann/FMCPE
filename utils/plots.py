import json
import matplotlib.lines as mlines
import math
import warnings
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,  # , message=".*Pass `\\(name,\\)` instead of `name`.*"
)
# plt.style.use(["science", "grid"])
# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.serif": ["Computer Modern"],
#         "axes.titlesize": 10,
#         # "axes.labelsize": 12,
#         # "xtick.labelsize": 10,
#         # "ytick.labelsize": 10,
#         "legend.fontsize": 7,
#         "lines.linewidth": 1,
#         "lines.markersize": 4,
#     }
# )

matplotlib.use("pgf")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
)
group_colors = {
    "fmcpe_nf": "tab:blue",
    "fmcpe_fm": "tab:blue",
    "fm_post_transform": "tab:blue",
    "fm_post_transform_only_ut": "tab:green",
    "fm_post_transform_plug_y": "tab:green",
    "fm_on_theta_cond": "tab:purple",
    "flow_matching_calibrated": "black",
    "dpe": "tab:orange",
    "npe": "tab:green",
    "fmpe": "tab:red",
    "mf_fmpe": "tab:gray",
    "mf_npe": "tab:green",
}
group_markers = {
    "flow_matching_calibrated": "o",
    "fmcpe_fm": "o",
    "fmcpe_nf": "s",
    "fm_post_transform": "s",
    "fm_post_transform_only_ut": "s",
    "fm_on_theta_cond": "o",
    "npe": "s",
    "dpe": "s",
    "fmpe": "o",
    "mf_fmpe": "o",
    "mf_npe": "s",
}

linestyle_methods = "-"
linestyle_baselines = "-"

# plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

TEXT_WIDTH = 6.53  # inches


def plot_losses_baselines(
    loss_dict: Dict[str, np.ndarray],
    baseline_names: Dict[str, str],
    save_root: Path,
    fig_title: str = "Baseline Losses",
    name: str = "losses_baselines.pdf",
):
    """
    Plots training and validation losses for different baselines in a grid layout.

    Args:
        loss_dict (Dict[str, np.ndarray]): Dictionary where keys are baseline names and values are loss arrays.
        baseline_names (Dict[str, str]): Dictionary mapping baseline keys to their display names.
        save_root (Path): Directory to save the resulting plot.
        fig_title (str, optional): Title for the entire figure. Defaults to "Baseline Losses".
    """
    loss_dict = {k: v for k, v in loss_dict.items() if v is not None and v.ndim > 1}
    num_plots = len(loss_dict)
    if num_plots == 0:
        return
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)  # Dynamically calculate the number of rows
    scale = 1.0
    ratio = 2 / 3
    width = TEXT_WIDTH * scale
    height = width * ratio

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(width, height), constrained_layout=True)
    axs = axs.flatten()  # Flatten axes for easy iteration
    # print(len(axs), num_rows, num_cols, num_plots, name)

    for idx, (baseline, loss) in enumerate(loss_dict.items()):
        ax = axs[idx]
        epochs = loss[0]
        train_loss = loss[1]
        val_loss = loss[2]
        ax.plot(epochs, train_loss, label="Train")
        ax.plot(epochs, val_loss, label="Val")
        ax.set_title(f"{baseline_names[baseline]}")

    ax.legend(loc="upper right")
    # Hide any extra subplots
    # for idx in range(len(loss_dict), len(axs)):
    #     fig.delaxes(axs[idx])

    # Add a title to the entire figure
    fig.suptitle(fig_title)

    # Save and close the figure
    plt.tight_layout()
    plt.savefig(save_root / name)
    plt.close(fig)


# Plotting function
def plot_boxplots(data, names, metrics_names, task_names, save_path):
    methods_list = ["fm_post_transform", "dpe", "mf_npe"]
    ncal_list = [10, 50, 200, 1000]
    # metrics = ["joint_wasserstein", "joint_c2st"]
    metrics = ["mse"]
    nrows = len(metrics)
    tasks = ["high_dim_gaussian", "pendulum", "wind_tunnel", "light_tunnel"]
    n_tasks = len(tasks)

    ratio = 0.28 * nrows
    scale = 1.0
    width = TEXT_WIDTH * scale
    height = width * ratio
    fig, axes = plt.subplots(
        nrows, n_tasks, figsize=(width, height), sharey=False, constrained_layout=True
    )

    if n_tasks == 1:
        axes = [axes]
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i, metric in enumerate(metrics):
        for idx, task in enumerate(tasks):
            ax = axes[i, idx]
            width = 0.8 / len(methods_list)  # distribute boxes within group

            for gi, ncal in enumerate(ncal_list):
                base_x = gi
                vals = None
                for mi, method in enumerate(methods_list):
                    for category in ["methods", "baselines"]:
                        if method in data[task].get(category, {}):
                            if str(ncal) in data[task][category][method]:
                                if metric in data[task][category][method][str(ncal)]:
                                    vals = np.array(data[task][category][method][str(ncal)][metric])
                    if vals is not None:
                        pos = base_x + mi * width
                        bp = ax.boxplot(
                            vals,
                            positions=[pos],
                            widths=width,
                            patch_artist=True,
                            showfliers=False,
                            boxprops=dict(linewidth=0.5, color="black"),  # thinner box edge
                            whiskerprops=dict(linewidth=0.5, color="black"),  # thinner whiskers
                            capprops=dict(linewidth=0.5, color="black"),
                            medianprops=dict(color="black", lw=0.3),
                        )
                        for patch in bp["boxes"]:
                            patch.set_facecolor(f"C{mi}")
                        if gi == 0:  # only add handle once per method
                            ax.plot([], c=f"C{mi}", label=names.get(method, method))

            ax.set_xticks([g + (len(methods_list) - 1) * width / 2 for g in range(len(ncal_list))])
            ax.set_xticklabels([str(nc) for nc in ncal_list])
            ax.set_title(task_names.get(task, task))
            ax.set_xlabel("Calibration size")
            if i == 0 and nrows > 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if i == 1 and nrows > 1:
                ax.set_title("")
            if idx == 0:
                ax.set_ylabel(metrics_names.get(metric, metric))
        axes[0, -1].legend()
        # fig.supxlabel("Calibration size")
    metrics_str = "_".join(metrics)
    plt.savefig(save_path / f"{metrics_str}_boxplots.pdf")


def plot_metrics_with_errorbars(data, metrics_to_plot, names, metrics_names, task_names, save_path):
    tasks = ["high_dim_gaussian", "high_dim_adaptive_gaussian", "pendulum", "wind_tunnel"]
    n_tasks = len(tasks)
    n_cols = 4
    n_rows = 1
    scale = 1.0
    width = TEXT_WIDTH * scale
    scale = 1.0
    ratio = 1 / 4

    for metric in metrics_to_plot:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, width * ratio * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        axes = [ax for row in axes for ax in row]  # flatten

        for idx, task in enumerate(tasks):
            col, row = idx % n_cols, idx // n_cols
            ax = axes[idx]
            results = {}

            for seed, seed_data in data[task].items():
                for category in ["methods", "baselines"]:
                    for method, calibs in seed_data.get(category, {}).items():
                        for calib, values in calibs.items():
                            if calib == "ref":
                                continue
                            if metric in values:
                                results.setdefault(method, {})
                                results[method].setdefault(int(calib), [])
                                results[method][int(calib)].append(values[metric])
            for method, calib_dict in results.items():
                if method not in ["fm_post_transform", "dpe", "mf_npe"]:
                    continue
                marker = group_markers.get(method, "o")
                color = group_colors.get(method, "tab:gray")
                calib_sizes = sorted(calib_dict.keys())
                means = [np.mean(calib_dict[c]) for c in calib_sizes]
                stds = [np.std(calib_dict[c]) for c in calib_sizes]
                display_name = names.get(method, method)
                ax.errorbar(
                    calib_sizes,
                    means,
                    yerr=stds,
                    marker=marker,
                    ms=2,
                    capsize=3,
                    label=display_name,
                    linestyle=linestyle_methods,
                    linewidth=0.5,
                    color=color,
                )

            ax.set_xscale("log")
            ax.set_title(task_names.get(task, task))
            if col == 0:
                ax.set_ylabel(metrics_names.get(metric, metric))
            ax.set_xlabel("Calibration size")

        # Remove unused subplots if tasks < n_rows * n_cols
        for j in range(len(tasks), n_rows * n_cols):
            fig.delaxes(axes[j])

        handles, labels = ax.get_legend_handles_labels()

        # Extract the line from ErrorbarContainer
        line_handles = []
        for h in handles:
            if isinstance(h, tuple) or hasattr(h, "lines"):  # ErrorbarContainer
                line = h[0] if isinstance(h, tuple) else h.lines[0]
                line_handles.append(
                    Line2D([0], [0], color=line.get_color(), lw=line.get_linewidth())
                )
            else:  # fallback for normal Line2D
                line_handles.append(Line2D([0], [0], color=h.get_color(), lw=h.get_linewidth()))

        axes[-1].legend(line_handles, labels, ncol=1)
        plt.savefig(save_path / f"{metric}_with_errorbars.pdf")


def plot_loss_grid_std_task(
    metrics: Dict,
    BASELINE_NAMES,
    TASK_NAMES,
    METRICS_NAMES,
    task: str,
    save_dir=None,
    title=None,
):
    losses = [
        "joint_c2st",
        "joint_wasserstein",
        "mse",
        # "acauc_v2",
    ]  # TODO : Move to global args
    # order alphabetically
    scale = 1.0
    ratio = 1 / 3
    width = TEXT_WIDTH * scale
    height = width * ratio
    num_cols = len(losses)
    fig, axes = plt.subplots(
        1,
        num_cols,
        figsize=(width, height),
        sharex="col",
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(1, num_cols)

    task_display_name = TASK_NAMES.get(task, task)
    baselines = metrics[task]["baselines"]
    methods = metrics[task]["methods"]
    for col, loss_name in enumerate(losses):
        ax = axes[0, col]
        ax.set_xscale("log")
        # Set tick labels at 10, 50, 200, 1000
        ax.set_xticks([10, 50, 200, 1000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if "c2st" not in loss_name:
            pass
        else:
            # set y lim between 0.48 and 1
            ax.set_ylim(0.48, 1.0)
        method_handles = []
        method_labels = []

        baseline_handles = []
        baseline_labels = []

        for method_name, method_metrics in methods.items():
            if method_name not in ["fm_post_transform"]:
                continue
            x = sorted([int(k) for k in methods[method_name].keys() if k.isdigit()])
            marker = group_markers.get(method_name, "o")
            color = group_colors.get(method_name, "tab:gray")
            display_name = BASELINE_NAMES.get(method_name, method_name)

            # get first key of method_metrics
            first_key = next(iter(method_metrics.keys()))
            if "ref" not in method_metrics:
                y_ref = None
            else:
                if loss_name not in method_metrics["ref"]:
                    y_ref = None
                else:
                    y_ref = method_metrics["ref"][loss_name]

            if loss_name not in method_metrics[first_key]:
                continue

            if isinstance(method_metrics[first_key][loss_name], list):
                # For conditional metrics, each entry is a list of values
                # for each obs
                y_ref = np.mean(y_ref) if y_ref is not None else None
                y_mean = np.mean(
                    np.asarray([method_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                )
                y_std = np.std(
                    np.asarray([method_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                )
                (line, _, _) = ax.errorbar(
                    x,
                    y_mean,
                    yerr=y_std,
                    label=display_name,
                    marker=marker,
                    linestyle=linestyle_methods,
                    color=color,
                    alpha=1.0,
                    capsize=3,
                )
                ax.set_xlim(left=0.9 * x[0])

            else:
                y = [method_metrics[str(ncal)][loss_name] for ncal in x]
                if loss_name in ["c2st", "joint_c2st"]:
                    # return 1 - y if y <= 0.5 else y
                    y = [1.0 - v if v <= 0.5 else v for v in y]
                (line,) = ax.plot(
                    x,
                    y,
                    label=display_name,
                    marker=marker,
                    linestyle=linestyle_methods,
                    color=color,
                    alpha=1.0,
                )
            method_handles.append(line)
            method_labels.append(display_name)

            if y_ref is not None:
                # Draw a horizontal line at the reference value
                ax.axhline(
                    y_ref,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=display_name,
                    alpha=0.3,
                )
        for baseline_name, base_metrics in baselines.items():
            if baseline_name not in ["dpe", "mf_npe"]:
                continue
            x = sorted([int(k) for k in baselines[baseline_name].keys() if k.isdigit()])
            marker = group_markers.get(baseline_name, "s")
            color = group_colors.get(baseline_name, "tab:gray")
            display_name = BASELINE_NAMES.get(baseline_name, baseline_name)

            first_key = next(iter(base_metrics.keys()))
            if "ref" not in base_metrics:
                y_ref = None
            else:
                if loss_name not in base_metrics["ref"]:
                    y_ref = None
                else:
                    y_ref = base_metrics["ref"][loss_name]

            if loss_name not in method_metrics[first_key]:
                continue

            if isinstance(base_metrics[first_key][loss_name], list):
                # For conditional metrics, each entry is a list of values
                # for each obs
                y_ref = np.mean(y_ref) if y_ref is not None else None
                y_mean = np.mean(
                    np.asarray([base_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                )
                y_std = np.std(
                    np.asarray([base_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                )
                (line, _, _) = ax.errorbar(
                    x,
                    y_mean,
                    yerr=y_std,
                    label=display_name,
                    marker=marker,
                    linestyle=linestyle_baselines,
                    color=color,
                    alpha=1.0,
                    capsize=3,
                )
                ax.set_xlim(left=0.9 * x[0])
            else:
                y = [base_metrics[str(ncal)][loss_name] for ncal in x]
                if loss_name in ["c2st", "joint_c2st"]:
                    # return 1 - y if y <= 0.5 else y
                    y = [1.0 - v if v <= 0.5 else v for v in y]

                (line,) = ax.plot(
                    x,
                    y,
                    label=display_name,
                    marker=marker,
                    linestyle=linestyle_baselines,
                    color=color,
                    alpha=1.0,
                )
            baseline_handles.append(line)
            baseline_labels.append(display_name)

            if y_ref is not None:
                # Draw a horizontal line at the reference value
                ax.axhline(
                    y_ref,
                    color=color,
                    linestyle="--",
                    label=display_name,
                    alpha=0.3,
                )
        ax.set_xlabel("Calibration set size")
        # if row == 0 and col == num_cols - 1:
        # ax.legend(title="Baseline", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_title(METRICS_NAMES[loss_name])
        # ax.tick_params(axis="x")
        # if col == 2 and task == "pendulum":
        #     ax.set_ylim(bottom=0, top=6.5)

    ax.legend(loc="best", frameon=False)
    # Remove duplicate labels
    # method_legend = dict(zip(method_labels, method_handles))
    # baseline_legend = dict(zip(baseline_labels, baseline_handles))
    # # plt.subplots_adjust(bottom=0.3)  # Adjust as needed
    # fig.legend(
    #     method_legend.values(),
    #     method_legend.keys(),
    #     ncol=2,
    #     title="Methods",
    #     bbox_to_anchor=(0.25, -0.02),
    #     bbox_transform=fig.transFigure,
    #     loc="upper center",
    # )
    #
    # fig.legend(
    #     baseline_legend.values(),
    #     baseline_legend.keys(),
    #     ncol=2,
    #     title="Baselines",
    #     bbox_to_anchor=(0.75, -0.02),
    #     bbox_transform=fig.transFigure,
    #     loc="upper center",
    #     # loc="center left",
    # )
    if save_dir is not None:
        if title is not None:
            save_path = Path(save_dir) / f"loss_grid_cal_std_{title}.pdf"
        else:
            save_path = Path(save_dir) / f"loss_grid_cal_std_{task}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")


def plot_loss_grid_std(
    metrics: Dict,
    BASELINE_NAMES,
    TASK_NAMES,
    METRICS_NAMES,
    save_dir=None,
):
    tasks = list(metrics.keys())
    # order alphabetically
    tasks = sorted(tasks)

    losses = [name for name in METRICS_NAMES.keys()]

    num_rows, num_cols = len(losses), len(tasks)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(max(10, 7 * num_cols), 4 * num_rows),
        sharex="col",
        squeeze=False,
    )
    fig.subplots_adjust(right=0.7)  # Adjust as needed
    axes = np.array(axes).reshape(num_rows, num_cols)

    for col, task in enumerate(tasks):
        task_display_name = TASK_NAMES.get(task, task)
        baselines = metrics[task]["baselines"]
        methods = metrics[task]["methods"]
        for row, loss_name in enumerate(losses):
            ax = axes[row, col]
            ax.set_xscale("log")
            method_handles = []
            method_labels = []

            baseline_handles = []
            baseline_labels = []

            for method_name, method_metrics in methods.items():
                if method_name not in [
                    "fm_post_transform",
                ]:
                    continue
                x = sorted([int(k) for k in methods[method_name].keys() if k.isdigit()])
                marker = group_markers.get(method_name, "o")
                color = group_colors.get(method_name, "tab:gray")
                display_name = BASELINE_NAMES.get(method_name, method_name)

                # get first key of method_metrics
                first_key = next(iter(method_metrics.keys()))
                if "ref" not in method_metrics:
                    y_ref = None
                else:
                    if loss_name not in method_metrics["ref"]:
                        y_ref = None
                    else:
                        y_ref = method_metrics["ref"][loss_name]

                if loss_name not in method_metrics[first_key]:
                    continue

                if isinstance(method_metrics[first_key][loss_name], list):
                    # For conditional metrics, each entry is a list of values
                    # for each obs
                    y_ref = np.mean(y_ref) if y_ref is not None else None
                    y_mean = np.mean(
                        np.asarray([method_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                    )
                    y_std = np.std(
                        np.asarray([method_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                    )
                    (line, _, _) = ax.errorbar(
                        x,
                        y_mean,
                        yerr=y_std,
                        label=display_name,
                        marker=marker,
                        linewidth=1,
                        linestyle=linestyle_methods,
                        color=color,
                        alpha=1.0,
                        capsize=3,
                    )
                    ax.set_xlim(left=0.9 * x[0])

                else:
                    y = [method_metrics[str(ncal)][loss_name] for ncal in x]
                    if loss_name in ["c2st", "joint_c2st"]:
                        # return 1 - y if y <= 0.5 else y
                        y = [1.0 - v if v <= 0.5 else v for v in y]
                    (line,) = ax.plot(
                        x,
                        y,
                        label=display_name,
                        marker=marker,
                        markersize=6,
                        linewidth=1,
                        linestyle=linestyle_methods,
                        color=color,
                        alpha=1.0,
                    )
                method_handles.append(line)
                method_labels.append(display_name)

                if y_ref is not None:
                    # Draw a horizontal line at the reference value
                    ax.axhline(
                        y_ref,
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        label=display_name,
                        alpha=0.3,
                    )
            for baseline_name, base_metrics in baselines.items():
                if baseline_name in ["rope", "fmpe"]:
                    continue
                x = sorted([int(k) for k in baselines[baseline_name].keys() if k.isdigit()])
                marker = group_markers.get(baseline_name, "s")
                color = group_colors.get(baseline_name, "tab:gray")
                display_name = BASELINE_NAMES.get(baseline_name, baseline_name)

                # print(f"Plotting baseline {baseline_name} for task {task} with loss {loss_name}")
                first_key = next(iter(base_metrics.keys()))
                if "ref" not in base_metrics:
                    y_ref = None
                else:
                    if loss_name not in base_metrics["ref"]:
                        y_ref = None
                    else:
                        y_ref = base_metrics["ref"][loss_name]

                if loss_name not in method_metrics[first_key]:
                    continue

                if isinstance(base_metrics[first_key][loss_name], list):
                    # For conditional metrics, each entry is a list of values
                    # for each obs
                    y_ref = np.mean(y_ref) if y_ref is not None else None
                    y_mean = np.mean(
                        np.asarray([base_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                    )
                    y_std = np.std(
                        np.asarray([base_metrics[str(ncal)][loss_name] for ncal in x]), axis=1
                    )
                    (line, _, _) = ax.errorbar(
                        x,
                        y_mean,
                        yerr=y_std,
                        label=display_name,
                        marker=marker,
                        linewidth=1,
                        linestyle=linestyle_baselines,
                        color=color,
                        alpha=1.0,
                        capsize=3,
                    )
                    ax.set_xlim(left=0.9 * x[0])
                else:
                    y = [base_metrics[str(ncal)][loss_name] for ncal in x]
                    if loss_name in ["c2st", "joint_c2st"]:
                        # return 1 - y if y <= 0.5 else y
                        y = [1.0 - v if v <= 0.5 else v for v in y]

                    (line,) = ax.plot(
                        x,
                        y,
                        label=display_name,
                        marker=marker,
                        markersize=6,
                        linewidth=1,
                        linestyle=linestyle_baselines,
                        color=color,
                        alpha=1.0,
                    )
                baseline_handles.append(line)
                baseline_labels.append(display_name)

                if y_ref is not None:
                    # Draw a horizontal line at the reference value
                    ax.axhline(
                        y_ref,
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        label=display_name,
                        alpha=0.3,
                    )
            # if row == 0 and col == num_cols - 1:
            # ax.legend(title="Baseline", bbox_to_anchor=(1.05, 1), loc="upper left")
            if row == 0:
                ax.set_title(task_display_name, fontsize=14)
            if col == 0:
                ax.set_ylabel(METRICS_NAMES[loss_name], fontsize=12)
            if row == num_rows - 1:
                ax.set_xlabel("Number of Calibration Samples", fontsize=12)
                ax.tick_params(axis="x", labelsize=10)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    # Remove duplicate labels
    method_legend = dict(zip(method_labels, method_handles))
    baseline_legend = dict(zip(baseline_labels, baseline_handles))
    fig.legend(
        method_legend.values(),
        method_legend.keys(),
        title="Methods",
        bbox_to_anchor=(0.73, 0.65),
        loc="center left",
        title_fontsize=14,
        fontsize=14,
    )

    fig.legend(
        baseline_legend.values(),
        baseline_legend.keys(),
        title="Baselines",
        bbox_to_anchor=(0.73, 0.5),
        loc="center left",
        title_fontsize=14,
        fontsize=14,
    )

    if save_dir is not None:
        save_path = Path(save_dir) / "loss_grid_cal_std.pdf"
        plt.savefig(save_path, format="pdf")
        print(f"Figure saved to {save_path}")


def pairplot_sbi(
    theta_true,
    theta_dict,
    simulator,
    NAMES,
    save_path=None,
    max_samples=2000,
    max_obs=None,
    dpi=300,
    title=None,
):
    """
    Create a corner-style pair plot for each observation.
    Scatter plots are rasterized to avoid LaTeX memory issues.

    Args:
        theta_true (torch.Tensor): (nobs, D) true parameters.
        theta_dict (dict): {method_name: torch.Tensor (nsamples, nobs, D)}.
        prior (torch.distributions.Distribution): prior distribution p(theta).
        save_path (str | Path | None): folder to save plots. If None, return figs.
        n_prior_samples (int): number of prior samples.
        max_samples (int): max number of samples per method (for clarity).
        max_obs (int | None): maximum number of observations to plot.
        dpi (int): resolution for rasterized scatter points.
    """
    nobs, D = theta_true.shape

    # Limit number of observations
    if max_obs is not None:
        nobs = min(nobs, max_obs)

    save_path = Path(save_path) if save_path is not None else None

    # Prepare LaTeX column names
    theta_labels = [f"$\\theta_{{{d}}}$" for d in range(D)]

    # Prior samples (shared across obs)
    # prior_samples = prior.sample((n_prior_samples,)).cpu().numpy()

    figs = []
    for obs in range(nobs):
        dfs = []

        # Methods
        for method, samples in theta_dict.items():
            samples = samples[:, obs, :]
            samples = simulator.transform.inv(samples).detach().cpu().numpy()
            if samples.shape[0] > max_samples:
                idx = np.random.choice(samples.shape[0], max_samples, replace=False)
                samples = samples[idx]
            df = pd.DataFrame(samples, columns=theta_labels)
            df["Methods"] = NAMES.get(method, method)
            dfs.append(df)

        # Prior
        # df_prior = pd.DataFrame(prior_samples, columns=theta_labels)
        # df_prior["Methods"] = "Prior"
        # dfs.append(df_prior)

        # True theta
        if true_data is not None:
            true_samples = true_data[:, obs, :].detach().cpu().numpy()
            df_true = pd.DataFrame(true_samples, columns=theta_labels)
            df_true["Methods"] = "True data"
            dfs.append(df_true)

        # Combine
        df_all = pd.concat(dfs, ignore_index=True)

        # Pairplot
        g = sns.pairplot(
            df_all,
            hue="Methods",
            diag_kind="kde",
            plot_kws=dict(alpha=0.5, s=5),
            diag_kws=dict(fill=True, alpha=0.5),
        )

        # Overlay true theta
        t = simulator.transform.inv(theta_true).detach().cpu().numpy()
        for i in range(D):
            g.axes[i, i].axvline(t[obs, i], color="red", linestyle="--", lw=1)
            for j in range(D):
                if i != j:
                    g.axes[i, j].scatter(
                        t[obs, j],
                        t[obs, i],
                        color="red",
                        marker="x",
                        s=40,
                        label="True",
                    )

        # Rasterize scatter plots
        for i in range(D):
            for j in range(D):
                if i != j and g.axes[i, j] is not None:
                    for coll in g.axes[i, j].collections:
                        coll.set_rasterized(True)

        # Fix legend
        # handles, labels = g.axes[0, -1].get_legend_handles_labels()
        # g.fig.legend(handles, labels, loc="upper center", ncol=len(labels))
        for ax in g.axes.flatten():
            if ax is not None and ax.get_legend() is not None:
                ax.get_legend().remove()

        # p # Set custom figure title
        if title is not None:
            g.fig.suptitle(title, fontsize=16)
            g.fig.subplots_adjust(top=0.92)  # leave space for titlelt.tight_layout()

        # Save or return
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            out_path = save_path / f"pairplot_{obs}.pdf"
            g.figure.savefig(out_path, bbox_inches="tight", dpi=dpi)
            plt.close(g.figure)
        else:
            figs.append((g.fig, g.axes))

    if save_path is None:
        return figs


def pairplot_sbi_diag(
    test_data,
    theta_dict,
    simulator,
    NAMES,
    save_path=None,
    max_samples=2000,
    max_obs=None,
    dpi=150,
    title=None,
):
    """
    Create diagonal-only pair plots (1D marginals) for each observation.
    Useful for summarizing marginal distributions without scatter plots.

    Args:
        theta_true (torch.Tensor): (nobs, D) true parameters.
        theta_dict (dict): {method_name: torch.Tensor (nsamples, nobs, D)}.
        prior (torch.distributions.Distribution): prior distribution p(theta).
        save_path (str | Path | None): folder to save plots. If None, return figs.
        n_prior_samples (int): number of prior samples.
        max_samples (int): max number of samples per method (for clarity).
        max_obs (int | None): maximum number of observations to plot.
        dpi (int): resolution.
        title (str | None): custom title for figures.
        scale, ratio, text_width: control figure size like in LaTeX.
    """
    theta_true = test_data["theta"]
    nobs, D = theta_true.shape
    assert D == 1, "This function only supports D=1."

    theta_true = simulator.transform.inv(theta_true).detach().cpu()
    nobs, dim_theta = theta_true.shape
    obs_indices = range(min(nobs, max_obs)) if max_obs is not None else range(nobs)
    palette = dict(
        zip(
            [NAMES["fm_post_transform"], NAMES["dpe"], NAMES["mf_npe"]],
            sns.color_palette(n_colors=3),
        )
    )
    if max_obs is not None:
        nobs = min(nobs, max_obs)

    scale = 1.0
    ratio = 1 / 3
    width = TEXT_WIDTH * scale
    height = width * ratio
    n_cal_to_display = 3
    save_path = Path(save_path) if save_path is not None else None
    theta_labels = [r"$H$"]

    figs = []
    for obs in range(nobs):
        dfs = []
        # Figure sizing (like in LaTeX)
        fig, axes = plt.subplots(
            1, n_cal_to_display, figsize=(width, height), dpi=dpi, sharey=False
        )

        # Methods
        for i, ncal in enumerate(
            sorted(theta_dict.keys(), key=lambda x: int(x))[:n_cal_to_display]
        ):
            dfs = []
            fig_solo, ax_solo = plt.subplots(
                1, 1, figsize=(width / n_cal_to_display, height), dpi=dpi
            )
            ax = axes[i]
            for method in ["fm_post_transform", "mf_npe", "dpe"]:
                samples = theta_dict[ncal].get(method, None)
                theta_inv = simulator.transform.inv(samples)
                arr = theta_inv[:, obs, 0].reshape(-1, 1).detach().cpu().numpy()
                df = pd.DataFrame(arr, columns=theta_labels)
                df["method"] = NAMES.get(method, method)
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            others = df_all[df_all["method"] != NAMES["fm_post_transform"]]
            fm_df = df_all[df_all["method"] == NAMES["fm_post_transform"]]

            # Plot marginals
            sns.kdeplot(
                data=df_all,
                x=theta_labels[0],
                hue="method",
                ax=ax,
                common_norm=False,
                fill=True,
                linewidth=1.5,
            )
            sns.kdeplot(
                data=df_all,
                x=theta_labels[0],
                hue="method",
                ax=ax_solo,
                common_norm=False,
                fill=True,
                linewidth=1.5,
            )
            # Overlay true theta
            t = theta_true[obs, 0]
            ax.axvline(t, color="black", linestyle="--", lw=0.5, label=r"$\theta^*$")
            ax.set_xlabel(theta_labels[0])
            ax.set_xlim(left=0, right=50)
            # remove tick labels for y axis if not first plot
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylabel("")

            ax_solo.axvline(t, color="black", linestyle="--", lw=0.5, label=r"$\theta^*$")
            ax_solo.set_xlabel(theta_labels[0])
            ax_solo.set_xlim(left=0, right=50)
            # remove tick labels for y axis if not first plot
            if i != 0:
                ax_solo.set_yticklabels([])
                ax_solo.set_yticks([])
                ax_solo.set_ylabel("")
            if i == 0:
                ax.set_ylabel("Density")
                handles, labels = ax_solo.get_legend_handles_labels()

                # Add custom handle for True θ (black dashed cross)
                # true_handle = mlines.Line2D(
                #     [], [], color="black", linestyle="--", lw=1, label=r"True $\theta$"
                # )
                for meth, c in palette.items():
                    true_handle = mlines.Line2D([], [], color=c, linestyle="-", lw=1, label=meth)
                    handles.append(true_handle)
                    labels.append(meth)

                # Put legend on the last joint axis
                ax_solo.legend(handles, labels, frameon=False, loc="upper left")
            else:
                ax_solo.get_legend().remove()

            # Only last axis gets the legend
            if i != n_cal_to_display - 1:
                ax.get_legend().remove()
            if save_path is not None:
                seed = save_path.stem.split("_")[-1]
                out_file_solo = save_path / f"pair_plot_diag_obs{obs}_seed{seed}_ncal{ncal}.pdf"
                fig_solo.savefig(out_file_solo, bbox_inches="tight")
                plt.close(fig_solo)
        # Title
        # if title is not None:
        #     fig.suptitle(title)
        #     fig.subplots_adjust(top=0.85)

        # plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for title + legend

        # Save or return
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            seed = save_path.stem.split("_")[-1]
            out_path = save_path / f"pairplot_diag_{obs}_seed{seed}.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
        else:
            figs.append((fig, axes))

    if save_path is None:
        return figs


def scatter_plots_by_cal_size(
    test_data,
    theta_dict,
    simulator,
    NAMES,
    save_path=None,
    rasterized=True,
    max_obs=None,
    max_samples=2000,
    title=None,
    type_plot="scatter",
):
    """
    Create 2x2 grid of scatter + marginal plots for calibration sizes.
    Each figure corresponds to one observation index.

    Args:
        test_data: dict with key "theta" of shape (nobs, dim_theta)
        theta_dict: dict[ncal][method] = tensor of shape (nsamples, nobs, dim_theta)
        simulator: object with .transform.inv method
        NAMES: dict mapping method -> display name
        save_path: optional Path to save PDFs
        rasterized: rasterize scatter plots
        max_obs: maximum number of observations to plot
    """

    theta_true = test_data["theta"]
    theta_true = simulator.transform.inv(theta_true).detach().cpu()
    nobs, dim_theta = theta_true.shape
    obs_indices = range(min(nobs, max_obs)) if max_obs is not None else range(nobs)
    palette = dict(
        zip(
            [NAMES["fm_post_transform"], NAMES["dpe"], NAMES["mf_npe"]],
            sns.color_palette(n_colors=3),
        )
    )
    if simulator.name == "pendulum":
        theta_labels = [r"$\omega_0$", r"$A$"]
        idx0, idx1 = 0, 1
    elif simulator.name == "light_tunnel":
        theta_labels = [r"$R$", r"$G$"]
        idx0, idx1 = 0, 1
    else:
        idx0, idx1 = 0, 1
        theta_labels_all = [f"$\\theta_{{{d + 1}}}$" for d in range(theta_true.shape[1])]
        theta_labels = [theta_labels_all[idx0], theta_labels_all[idx1]]
    # global x/y limits across all obs + methods
    if "low" in simulator.prior_params:
        prior_low = simulator.prior_params["low"][[idx0, idx1]]
        prior_high = simulator.prior_params["high"][[idx0, idx1]]
        xlim = (prior_low[0].item(), prior_high[0].item())
        ylim = (prior_low[1].item(), prior_high[1].item())
    else:
        all_x, all_y = [], []
        for ncal, methods in theta_dict.items():
            for method, samples in methods.items():
                if method not in ["dpe", "mf_npe", "fm_post_transform"]:
                    continue
                theta_inv = simulator.transform.inv(samples)
                all_x.append(theta_inv[..., idx0].reshape(-1))
                all_y.append(theta_inv[..., idx1].reshape(-1))
        all_x = torch.cat(all_x).numpy()
        all_y = torch.cat(all_y).numpy()
        xlim = (all_x.min(), all_x.max())
        ylim = (all_y.min(), all_y.max())

    scale = 1.0
    ratio = 1 / 3
    width = TEXT_WIDTH * scale
    height = width * ratio
    n_cal_to_display = 3
    if save_path is not None:
        seed = save_path.stem.split("_")[-1]
    for obs in obs_indices:
        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(1, n_cal_to_display, figure=fig)

        ncal_list = list(theta_dict.keys())
        ax_joints = []

        for i, ncal in enumerate(ncal_list):
            if i >= n_cal_to_display:  # max 2x2 grid
                break

            # row, col = divmod(i, 2)
            row = 0
            col = i
            # joint + marginals: layout 2x2 inside this slot
            sub_gs = gs[row, col].subgridspec(
                2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.02, wspace=0.02
            )
            fig_solo = plt.figure(figsize=(width / n_cal_to_display, height))
            gs_solo = GridSpec(
                2,
                2,
                figure=fig_solo,
                height_ratios=[1, 4],
                width_ratios=[4, 1],
                hspace=0.02,
                wspace=0.02,
            )

            ax_joint = fig.add_subplot(sub_gs[1, 0])
            ax_joint_solo = fig_solo.add_subplot(gs_solo[1, 0], sharex=ax_joint, sharey=ax_joint)
            ax_marg_x = fig.add_subplot(sub_gs[0, 0], frame_on=False)  # overlay top
            ax_marg_x_solo = fig_solo.add_subplot(gs_solo[0, 0], sharex=ax_joint)
            ax_marg_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)
            ax_marg_y_solo = fig_solo.add_subplot(gs_solo[1, 1], sharey=ax_joint)
            fig.add_subplot(sub_gs[1, 0]).axis("off")  # empty bottom strip
            fig.add_subplot(sub_gs[1, 1]).axis("off")

            ax_joints.append(ax_joint)

            # collect samples for this cal size + obs
            dfs = []
            for method in ["fm_post_transform", "mf_npe", "dpe"]:
                samples = theta_dict[ncal].get(method, None)
                theta_inv = simulator.transform.inv(samples)
                arr = theta_inv[:, obs, [idx0, idx1]].detach().cpu().numpy()
                df = pd.DataFrame(arr, columns=theta_labels)
                df["method"] = NAMES.get(method, method)
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            others = df_all[df_all["method"] != NAMES["fm_post_transform"]]
            fm_df = df_all[df_all["method"] == NAMES["fm_post_transform"]]

            # scatter
            if type_plot == "scatter":
                sns.scatterplot(
                    data=df_all,
                    x=theta_labels[0],
                    y=theta_labels[1],
                    hue="method",
                    ax=ax_joint,
                    s=6,
                    alpha=0.4,
                    rasterized=rasterized,
                    palette=palette,
                )
                sns.scatterplot(
                    data=df_all,
                    x=theta_labels[0],
                    y=theta_labels[1],
                    hue="method",
                    ax=ax_joint_solo,
                    s=6,
                    alpha=0.4,
                    rasterized=rasterized,
                    palette=palette,
                )
            elif type_plot == "kde":
                if simulator.name == "pure_gaussian" and i == 0:
                    bw = 3.0
                else:
                    bw = 1.0
                thresh = 0.05
                sns.kdeplot(
                    data=others,
                    x=theta_labels[0],
                    y=theta_labels[1],
                    hue="method",
                    ax=ax_joint,
                    fill=True,
                    alpha=0.6,
                    palette=palette,
                    levels=10,
                    bw_adjust=bw,
                    thresh=thresh,
                )
                sns.kdeplot(
                    data=others,
                    x=theta_labels[0],
                    y=theta_labels[1],
                    hue="method",
                    ax=ax_joint_solo,
                    fill=True,
                    alpha=0.6,
                    palette=palette,
                    levels=10,
                    bw_adjust=bw,
                    thresh=thresh,
                )
                sns.kdeplot(
                    data=fm_df,
                    x=theta_labels[0],
                    y=theta_labels[1],
                    hue="method",
                    ax=ax_joint,
                    fill=True,
                    alpha=0.6,
                    palette=palette,
                    levels=10,
                    bw_adjust=bw,
                    thresh=thresh,
                    zorder=10,
                )
                sns.kdeplot(
                    data=fm_df,
                    x=theta_labels[0],
                    y=theta_labels[1],
                    hue="method",
                    ax=ax_joint_solo,
                    fill=True,
                    alpha=0.6,
                    palette=palette,
                    levels=10,
                    bw_adjust=bw,
                    thresh=thresh,
                    zorder=10,
                )
            else:
                raise ValueError(f"Unknown plot type {type_plot}")
            # true parameter as crosshairs
            theta0_true, theta1_true = theta_true[obs, idx0].item(), theta_true[obs, idx1].item()
            ax_joint.axvline(theta0_true, color="black", linestyle="--", linewidth=0.5)
            ax_joint.axhline(
                theta1_true, color="black", linestyle="--", linewidth=0.5, label=r"$\theta^*$"
            )

            ax_joint.set_xlim(xlim)
            ax_joint.set_ylim(ylim)

            # Solo fig
            ax_joint_solo.axvline(theta0_true, color="black", linestyle="--", linewidth=0.5)
            ax_joint_solo.axhline(
                theta1_true, color="black", linestyle="--", linewidth=0.5, label=r"$\theta^*$"
            )

            ax_joint_solo.set_xlim(xlim)
            ax_joint_solo.set_ylim(ylim)

            # inline text instead of title
            ax_joint.text(
                0.02,
                0.95,
                rf"$N_{{\mathrm{{cal}}}} = {ncal}$",
                transform=ax_joint.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )

            # marginals
            sns.kdeplot(
                data=df_all,
                x=theta_labels[0],
                hue="method",
                ax=ax_marg_x,
                fill=True,
                alpha=0.4,
                legend=False,
                palette=palette,
            )
            sns.kdeplot(
                data=df_all,
                y=theta_labels[1],
                hue="method",
                ax=ax_marg_y,
                fill=True,
                alpha=0.4,
                legend=False,
                palette=palette,
            )

            # Solo marginals
            sns.kdeplot(
                data=df_all,
                x=theta_labels[0],
                hue="method",
                ax=ax_marg_x_solo,
                fill=True,
                alpha=0.4,
                legend=False,
                palette=palette,
            )
            sns.kdeplot(
                data=df_all,
                y=theta_labels[1],
                hue="method",
                ax=ax_marg_y_solo,
                fill=True,
                alpha=0.4,
                legend=False,
                palette=palette,
            )

            # clean marginals
            for ax in [ax_marg_x, ax_marg_y, ax_marg_x_solo, ax_marg_y_solo]:
                ax.set_frame_on(False)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.set_xlabel("")
                ax.set_ylabel("")

            # remove labels inside
            # if row == 0:
            #     ax_joint.set_xlabel("")
            if col != 0:
                ax_joint.set_ylabel("")
                ax_joint.set_yticklabels([])
                ax_joint_solo.set_ylabel("")
                ax_joint_solo.set_yticklabels([])
            if i == 2 and simulator.name == "pendulum":
                # if i == 2 and simulator.name == "light_tunnel":
                # Get handles & labels from scatter (methods)
                handles, labels = ax_joint_solo.get_legend_handles_labels()

                # Add custom handle for True θ (black dashed cross)
                # true_handle = mlines.Line2D(
                #     [], [], color="black", linestyle="--", lw=1, label=r"True $\theta$"
                # )
                for meth, c in palette.items():
                    true_handle = mlines.Line2D([], [], color=c, linestyle="-", lw=1, label=meth)
                    handles.append(true_handle)
                    labels.append(meth)

                # Put legend on the last joint axis
                ax_joint_solo.legend(
                    handles, labels, frameon=True, loc="lower right", framealpha=1.0
                )
            else:
                ax_joint_solo.get_legend().remove()

            # fig_solo.tight_layout()
            if save_path is not None:
                out_file_solo = save_path / f"{type_plot}_joint_obs{obs}_seed{seed}_ncal{ncal}.pdf"
                fig_solo.savefig(out_file_solo, bbox_inches="tight")
                plt.close(fig_solo)

        # put legend in the last joint axis
        if ax_joints:
            handles, labels = ax_joints[-1].get_legend_handles_labels()
            for ax in ax_joints:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            # ax_joints[-1].legend(handles, labels, frameon=False, loc="upper right")

        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            out_file = save_path / f"{type_plot}_joint_obs{obs}_seed{seed}.pdf"
            fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

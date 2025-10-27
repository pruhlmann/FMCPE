import copy
import mlxp


@mlxp.launch(config_path="./configs")
def main(ctx: mlxp.Context) -> None:
    from pathlib import Path
    from typing import List
    import pickle

    import torch
    import json
    from mlxp.logger import Logger

    from baselines import compute_baselines
    from baselines.trainers import Trainer
    from simulator import get_simulator, generate_simulation_dataset, generate_calibration_dataset
    from baselines.trainers import train_npe
    from baselines.trainers import train_flow_matching
    from utils.misc import get_model_path_for_multirun

    cfg = ctx.config
    logger: Logger = ctx.logger
    task = cfg["task"]
    num_samples = task["num_samples"]
    num_cal: List[int] = task["num_cal"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Found device: {device}")

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    # Get simulator and generate dataset
    print("Generating dataset for task: ", task.name)
    simulator = get_simulator(task.name, **task.simulator.params)

    log_path = Path(ctx.mlxp.logger.parent_log_dir)
    data_path = log_path / "data" / task["name"]
    data_path.mkdir(parents=True, exist_ok=True)

    # Always generate a set amount of calib data to maintain consitency among runs
    num_cal_max = min(num_samples, task["max_cal"])
    assert task["max_cal"] >= max(num_cal)
    cal_values = sorted(num_cal + [num_cal_max])

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    # TODO : Maybe force a seed for sim data and only
    # change for calibration

    # Simulation Data
    if cfg["load_data"]:
        try:
            data = torch.load(data_path / "simulations.pt")
            theta_sim = data["theta"]
            x_sim = data["x"]
        except FileNotFoundError:
            raise FileNotFoundError("Simulation data not found. Please generate it first.")
    else:
        theta_sim, x_sim = generate_simulation_dataset(simulator, num_samples)
        torch.save(
            {"theta": theta_sim, "x": x_sim},
            data_path / "simulations.pt",
        )

    # Calibration Data
    if cfg["load_data"]:
        try:
            data = torch.load(data_path / "calibrations.pt")
            theta_cal = data["theta"]
            x_cal = data["x"]
            y_cal = data["y"]
        except FileNotFoundError:
            raise FileNotFoundError("Calibration data not found. Please generate it first.")
    else:
        theta_cal, x_cal, y_cal = generate_calibration_dataset(
            simulator, num_cal_max, task["generation"]
        )
        torch.save({"theta": theta_cal, "x": x_cal, "y": y_cal}, data_path / "calibrations.pt")

    # Path to save intermediate weights
    model_path = get_model_path_for_multirun(log_path / "models", cfg["exp_name"], task, cfg)
    model_path.mkdir(parents=True, exist_ok=True)

    sim_model_path = log_path / "models" / task["name"]

    # Save simulator
    with open(sim_model_path / "simulator.json", "w") as f:
        json.dump(task["simulator"], f)
    with open(model_path / "simulator.json", "w") as f:
        json.dump(task["simulator"], f)

    # Train NPE on (theta,x)
    print("##############################")
    print("Training NPE on (theta,x)")
    npe = train_npe(
        task,
        theta_sim,
        x_sim,
        device=device,
        model_path=sim_model_path,
        logger=logger,
        logname="npe",
        save=True,
        load=True,
    )
    npe.cpu()
    torch.save(npe.state_dict(), model_path / "npe.pth")
    with open(model_path / "npe.json", "w") as f:
        json.dump(task["npe"], f)
    with open(model_path / "npe.pkl", "wb") as f:
        pickle.dump(npe, f)
    # Count parameters
    total_params = sum(p.numel() for p in npe.parameters())
    print(f"Total parameters: {total_params}")
    print("##############################")

    # Train NPE with Flow Matching
    print("##############################")
    print("Training NPE using Flow Matching on (theta,x)")
    fmpe = train_flow_matching(
        theta_sim,
        x_sim,
        task["fmpe"]["config"],
        device=device,
        logger=logger,
        logname="npe_fmpe",
        save=True,
        load=True,
        model_path=sim_model_path,
        **task["fmpe"]["training_params"],
    )
    fmpe.cpu()
    torch.save(fmpe.state_dict(), model_path / ("npe_fmpe" + ".pth"))
    with open(model_path / "npe_fmpe.json", "w") as f:
        json.dump(task["fmpe"]["config"], f)
    # Count parameters
    total_params = sum(p.numel() for p in fmpe.parameters())
    print(f"Total parameters: {total_params}")
    print("##############################")

    #####################
    ### Train methods ###
    #####################
    assert cfg["methods_to_train"], (
        "Provide at least one method to train in the configuration file."
    )
    results_dict = {}
    for method in cfg["methods_to_train"]:
        trainer = Trainer(method, logger, model_path, task)
        models = trainer.train(
            (theta_sim, x_sim),
            (theta_cal, x_cal, y_cal),
            cal_values,
            task["naugment"],
            device,
            load=False,
            compute_reference=cfg["compute_reference"],
            **task[method]["training_params"],
        )
        results_dict[method] = models

    # Save all methods, structure is {'method_name': {'num_cal': posterior}}
    logger.log_artifacts(
        results_dict,
        artifact_name="methods",
        artifact_type="pickle",
    )

    ###################################
    ###### Baselines computation ######
    ###################################

    if cfg["train_baselines"]:
        print("\n##############################")
        print("Computing baselines")
        print("##############################\n")

        baselines = {"dpe": {}, "mf_npe": {}}
        for ncal in cal_values:
            if ncal == cal_values[-1] and not cfg["compute_reference"]:
                print("Skipping reference computation as per configuration.")
                continue
            print("Training baselines for ncal =", ncal)
            posteriors = compute_baselines(
                task,
                theta_cal[: ncal * task["naugment"]],
                x_cal[: ncal * task["naugment"]],
                y_cal[: ncal * task["naugment"]],
                theta_sim,
                x_sim,
                simulator,
                device,
                model_path=model_path,
                logger=logger,
                npe=npe,
                max_cal=cal_values[-1],
            )
            if ncal == cal_values[-1]:
                baselines["dpe"]["ref"] = posteriors["dpe"]
                baselines["mf_npe"]["ref"] = posteriors["mf_npe"]
            else:
                baselines["dpe"][str(ncal)] = posteriors["dpe"]
                baselines["mf_npe"][str(ncal)] = posteriors["mf_npe"]
        logger.log_artifacts(
            baselines,
            artifact_name="baselines",
            artifact_type="pickle",
        )
    # log embedding not for x and y
    dpe_ref = train_npe(
        task=task,
        theta=theta_cal[: cal_values[-1] * task["naugment"]],
        x=y_cal[: cal_values[-1] * task["naugment"]],
        device=device,
        model_path=model_path,
        load=True,
        save=True,
        logger=logger,
        logname="dpe_ref",
    )
    dpe_ref.cpu()
    embedding_nets = {}
    embedding_nets["x"] = copy.deepcopy(npe.embedding_net)
    embedding_nets["y"] = copy.deepcopy(dpe_ref.embedding_net)

    logger.log_artifacts(embedding_nets, artifact_name="embedding_networks", artifact_type="pickle")


if __name__ == "__main__":
    main()

import torch
from simulator import generate_dataset, get_simulator
from utils.metrics import classifier_two_samples_test
from utils.misc import rescale

# Task parameters
task_params = {
    "dim": 200,  # Number of time points (N)
    "low_w0": 1.0,  # Lower bound for w
    "high_w0": 3.0,  # Upper bound for w
    "low_A": 0.5,  # Lower bound for A
    "high_A": 5.0,  # Upper bound for A
    "tmax": 10.0,  # Maximum time
}


def main(
    num_samples: int = 200,
    warmups_steps: int = 200,
):
    # generate dataset
    pendulum = get_simulator("pendulum", **task_params)
    theta, x, y, scales = generate_dataset(
        pendulum, num_samples, rescale=None, generation="transitive"
    )
    theta_estimates = pendulum.sample_reference_posterior(
        1, x, misspecified=True, theta_true=theta, warmup_steps=warmups_steps
    )
    scores = classifier_two_samples_test(theta, theta_estimates).item()
    print(scores)
    print(f"Classifier two samples test: {scores * 100:.2f}%")


if __name__ == "__main__":
    main()

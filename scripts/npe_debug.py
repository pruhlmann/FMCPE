from time import time
import torch
import seaborn as sns
from sbi import analysis, utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import matplotlib.pyplot as plt

from simulator import get_simulator

seed = 0
torch.manual_seed(seed)

# data
# Get simulator and generate dataset
task_params = {
    "dim": 200,  # Number of time points (N)
    "low_w0": 0.0,  # Lower bound for w
    "high_w0": 3.0,  # Upper bound for w
    "low_A": 0.5,  # Lower bound for A
    "high_A": 10.0,  # Upper bound for A
    "tmax": 10.0,  # Maximum time
    "noise": 0.1,  # Noise level
}
simulator_pen = get_simulator("pendulum", **task_params)
num_samples = 50000

embedding_net = CNNEmbedding(
    input_shape=(200,),
    in_channels=1,
    out_channels_per_layer=[6],
    num_conv_layers=1,
    num_linear_layers=1,
    output_dim=8,
    kernel_size=5,
    pool_kernel_size=8,
)
prior = utils.BoxUniform(
    low=torch.tensor([task_params["low_w0"], task_params["low_A"]]),
    high=torch.tensor([task_params["high_w0"], task_params["high_A"]]),
)
prior, num_parameters, prior_returns_numpy = process_prior(prior)

simulator_wrapper = process_simulator(
    simulator_pen.get_simulator(False), prior, prior_returns_numpy
)
check_sbi_inputs(simulator_wrapper, prior)


# instantiate the neural density estimator
neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net)

# setup the inference procedure with NPE
inferer = NPE(prior=prior, density_estimator=neural_posterior)

# run the inference procedure on one round and 10000 simulated data points
theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=num_samples)
density_estimator = inferer.append_simulations(theta, x).train(training_batch_size=256)
posterior = inferer.build_posterior(density_estimator)
true_parameter = prior.sample((1,))
x_obs = simulator_pen.get_simulator(False)(true_parameter)
samples = posterior.set_default_x(x_obs).sample((1000,))
# create the figure
fig, ax = analysis.pairplot(
    samples,
    points=true_parameter,
    labels=[r"$\omega_0$", r"$A$"],
    limits=[[0, 3], [0.5, 10]],
    fig_kwargs=dict(
        points_colors="r",
        points_offdiag={"markersize": 6},
    ),
    figsize=(5, 5),
)
plt.savefig("figures/pendulum_posterior_sbi.pdf", dpi=300, bbox_inches="tight")
plt.close()
samples_g = []
for i in range(1):
    thetas = prior.sample((1000,))
    x_obs_g = simulator_pen.get_simulator(False)(thetas)
    print("Start log prob")
    tb = time()
    log_probs = posterior.log_prob_batched(thetas.unsqueeze(0), x_obs_g)
    te = time()
    print("End log prob", "Elapsed time: ", te - tb)
    print("Start timer")
    tb = time()
    samples_g.append(posterior.sample_batched((1,), x_obs_g)[0])
    te = time()
    print("End sampling", "Elapsed time: ", te - tb)
samples_g = torch.cat(samples_g, dim=0)
sns.jointplot(x=samples_g[:, 0], y=samples_g[:, 1])
plt.savefig("figures/pendulum_posterior_sbi_g.pdf", dpi=300, bbox_inches="tight")
plt.close()

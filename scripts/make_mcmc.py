import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from jax import random
from tqdm import tqdm
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path


rng_key = random.PRNGKey(42)

t = jnp.linspace(0, 10.0, 200)
sg_x = 1
sg_y = 0.1


def model(prior_A, prior_w, x_obs=None, y_obs=None):
    A = numpyro.sample("A", prior_A)
    w = numpyro.sample("f", prior_w)
    phi = numpyro.sample("phi", dist.Uniform(low=-jnp.pi, high=jnp.pi))
    s = A * jnp.cos(w * t + phi)
    x = numpyro.sample("x", dist.Normal(loc=s, scale=sg_x), obs=x_obs)
    alpha = numpyro.sample("alpha", dist.Uniform(low=0, high=1))
    y = numpyro.sample("y", dist.Normal(loc=jnp.exp(-alpha * t) * x, scale=sg_y), obs=y_obs)


def simulate_data(rng_key, prior_A, prior_w, n_simulations):
    rng_key, subkey = random.split(rng_key)
    A_df = prior_A.sample(subkey, (n_simulations,))
    f_df = prior_w.sample(subkey, (n_simulations,))
    x_df = []
    y_df = []
    for A_i, f_i in zip(A_df, f_df):
        predictive = Predictive(
            condition(partial(model, prior_A=prior_A, prior_w=prior_w), {"A": A_i, "f": f_i}),
            num_samples=1,
        )
        rng_key, subkey = random.split(rng_key)
        data = predictive(subkey)
        xi = data["x"]
        yi = data["y"]
        x_df.append(xi)
        y_df.append(yi)
    x_df = np.asarray(jnp.stack(x_df))
    y_df = np.asarray(jnp.stack(y_df))

    data = {}
    data["A"] = A_df
    data["f"] = f_df
    data["x"] = x_df
    data["y"] = y_df

    return data


def generate_posterior_samples_x(
    rng_key, prior_A, prior_w, x_obs, A_obs, f_obs, num_samples=10_000
):
    kernel = NUTS(
        partial(model, prior_A=prior_A, prior_w=prior_w),
        init_strategy=init_to_value(None, values={"A": A_obs, "f": f_obs}),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=num_samples,
    )
    rng_key, subkey = random.split(rng_key)
    mcmc.run(rng_key=subkey, x_obs=x_obs)

    samples_A = mcmc.get_samples()["A"]
    samples_f = mcmc.get_samples()["f"]

    return jnp.concat([samples_A, samples_f], axis=1)


def generate_posterior_samples_y(
    rng_key, prior_A, prior_w, y_obs, A_obs, f_obs, num_samples=10_000
):
    kernel = NUTS(
        partial(model, prior_A=prior_A, prior_w=prior_w),
        init_strategy=init_to_value(None, values={"A": A_obs, "f": f_obs}),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=num_samples,
    )
    rng_key, subkey = random.split(rng_key)
    mcmc.run(rng_key=subkey, y_obs=y_obs)

    samples_A = mcmc.get_samples()["A"]
    samples_f = mcmc.get_samples()["f"]

    return jnp.concat([samples_A, samples_f], axis=1)


df_x_list = []
df_y_list = []
theta_list = [[1.0, 3.0], [1.5, 5.0], [2.0, 7.0]]
for i, theta_obs in enumerate(theta_list):
    theta_obs = jnp.array(theta_obs)
    f_obs, A_obs = theta_obs
    prior_f = dist.Uniform(low=jnp.array([0.0]), high=jnp.array([3.0]))
    prior_A = dist.Uniform(low=jnp.array([0.5]), high=jnp.array([10.0]))
    rng_key, subkey = random.split(rng_key)
    predictive = Predictive(
        condition(partial(model, prior_A=prior_A, prior_w=prior_f), {"A": A_obs, "f": f_obs}),
        num_samples=1,
    )
    rng_key, subkey = random.split(rng_key)
    data = predictive(subkey)
    x_obs = data["x"]
    y_obs = data["y"]

    rng_key, subkey = random.split(rng_key)

    samples_x = generate_posterior_samples_x(
        rng_key=rng_key,
        prior_A=prior_A,
        prior_w=prior_f,
        x_obs=x_obs,
        A_obs=A_obs,
        f_obs=f_obs,
        num_samples=1000,
    )
    df_x = pd.DataFrame(data=samples_x)
    df_x["example"] = [i] * len(samples_x)
    df_x_list.append(df_x)

    samples_y = generate_posterior_samples_y(
        rng_key=rng_key,
        prior_A=prior_A,
        prior_w=prior_f,
        y_obs=y_obs,
        A_obs=A_obs,
        f_obs=f_obs,
        num_samples=1000,
    )
    df_y = pd.DataFrame(data=samples_y)
    df_y["example"] = [i] * len(samples_y)
    df_y_list.append(df_y)


fig, ax = plt.subplots(figsize=(13, 4), ncols=3)

# which example to consider
for i in range(3):
    # plot the posterior for x
    axi = ax[i]
    df = df_x_list[i]
    sns.kdeplot(df[df["example"] == i], x=0, y=1, thresh=0.10, fill=False, cmap="Reds", ax=axi)

    # plot the posterior for y
    axi = ax[i]
    df = df_y_list[i]
    sns.kdeplot(df[df["example"] == i], x=0, y=1, thresh=0.10, fill=False, cmap="Greens", ax=axi)

    axi.set_title(f"example {i + 1}")

    axi.set_xlabel("A")
    axi.set_ylabel("")

    axi.scatter(theta_list[i][1], theta_list[i][0], c="blue", s=30, zorder=2)

    axi.plot([], [], c="red", lw=2.0, label=r"$p(\theta \mid x)$")
    axi.plot([], [], c="green", lw=2.0, label=r"$p(\theta \mid y)$")

    # axi.set_xlim(0.0, 3.0)
    # axi.set_ylim(0.5, 10.0)

ax[0].set_ylabel("f")
ax[0].legend()
fig.savefig("pendulum_mcmc_samples.pdf", format="pdf")
fig.show()

import time
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import arviz as az
from jax import random, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp, expit
import numpyro
from numpyro import handlers
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
# from numpyro.contrib.nested_sampling import NestedSampler
from sklearn.preprocessing import LabelEncoder
import numpy as np

def simple_model(bus, duration=None):
    # Need to prep data with label encoder
    n_bus = len(np.unique(bus))
    with numpyro.plate("bus", n_bus):
        mu = numpyro.sample("mu", dist.Normal(90, 90))
    sigma = numpyro.sample("sigma", dist.HalfNormal(40))
    numpyro.sample("obs", dist.Normal(mu[bus], sigma), obs=duration)

def expit_model(station, departure, duration=None):
    region_base = numpyro.sample("region_base", dist.Normal(50, 20))
    hwy_sigma = numpyro.sample("hwy_sigma", dist.Exponential(1.0/20))
    am_rush_hour_start = numpyro.sample("am_rush_hour_start", dist.Normal(17e3, 5e3))
    am_rush_hour_length = numpyro.sample("am_rush_hour_length",
                                         dist.Exponential(1.0/7200))
    am_rush_hour_onset = numpyro.sample("am_rush_hour_onset",
                                             dist.Exponential(1.0/2e3))
    am_rush_hour_fade = numpyro.sample("am_rush_hour_fade",
                                             dist.Exponential(1.0/2e3))
    pm_rush_hour_start = numpyro.sample("pm_rush_hour_start",
                                        dist.Uniform(60e3, 5e3))
    pm_rush_hour_length = numpyro.sample("pm_rush_hour_length",
                                         dist.Exponential(1.0/7200))
    pm_rush_hour_onset = numpyro.sample("pm_rush_hour_onset",
                                             dist.Exponential(1.0/2e3))
    pm_rush_hour_fade = numpyro.sample("pm_rush_hour_fade",
                                             dist.Exponential(1.0/2e3))
    am_rush_hour_end = am_rush_hour_start + am_rush_hour_length
    pm_rush_hour_end = pm_rush_hour_start + pm_rush_hour_length
    am_rush = (expit(am_rush_hour_onset * (departure - am_rush_hour_start)) *
               expit(-am_rush_hour_fade * (departure - am_rush_hour_end)))
    pm_rush = (expit(pm_rush_hour_onset * (departure - pm_rush_hour_start)) *
               expit(-pm_rush_hour_fade * (departure - pm_rush_hour_end)))
    am_rush_penalty = numpyro.sample("am_rush_penalty",
                                     dist.Exponential(1.0/7200))
    pm_rush_penalty = numpyro.sample("pm_rush_penalty",
                                     dist.Exponential(1.0/7200))
    sigma = numpyro.sample("sigma", dist.Exponential(1/1800.))

    n_stations = 2
    with numpyro.plate("station", n_stations):
        hwy_base = numpyro.sample(
                "hwy_base",
                dist.Normal(region_base, hwy_sigma))
    mu = ( hwy_base[station]
          + am_rush * am_rush_penalty
          + pm_rush * pm_rush_penalty)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=duration)

# squared exponential kernel with diagonal noise term
def sqexp_kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

def gp_model(station, departure, duration=None):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel, but I don't like how this is making the kernel n x n
    # while it could be k x k, where n is the number of bus rides and k is
    # the number of distinct departure times.
    k = sqexp_kernel(departure, departure, var, length, noise)

    # n_departure = len(np.unique(departure))

    # sample durations according to the standard gaussian process formula
    numpyro.sample(
        "obs",
        dist.MultivariateNormal(loc=jnp.zeros(departure.shape[0]), covariance_matrix=k),
        obs=duration,
    )

def fit_simple_model(rng_key_, buses_ix, duration):

    # Run NUTS.
    kernel = NUTS(simple_model)
    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        bus=buses_ix,
        duration=duration,
    )
    return mcmc

def fit_rush_hour_model(bus_dset, rng_key_):
    station_labels = bus_dset["station"]
    station_le = LabelEncoder()
    stations_ix = station_le.fit_transform(station_labels)

    # Run NUTS.
    kernel = NUTS(expit_model)
    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        station=stations_ix,
        departure=bus_dset['time.departure'].values,
        duration=bus_dset.duration.values,
    )
    # mcmc.print_summary()
    # samples_1 = mcmc.get_samples()
    return mcmc

def fit_gp_model(rng_key, station, departure, duration):
    kernel = NUTS(gp_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    mcmc.run(rng_key, station, departure)
    return mcmc

def summary_plots(rng_key, mcmc, bus_dset, buses_le):
    fig, ax = plt.subplots(figsize=(12, 12,))
    sns.scatterplot(bus_dset, x="time.departure", y="duration",
                    hue="station", ax=ax)
    fig.savefig("raw_durations.png")
    data = az.from_numpyro(mcmc)
    az.plot_trace(data)
    plt.savefig("posterior_samples.png")
    pred = Predictive(simple_model, posterior_samples=mcmc.get_samples())
    duration_pred_0900_nashua = pred(rng_key, buses_le.transform(np.array(["09:00_nashua"])))["obs"]
    fig, ax = plt.subplots(figsize=(12, 12,))
    az.plot_dist(duration_pred_0900_nashua, ax=ax)
    fig.savefig("9am_nashua_duration_prediction.png")
    fig, ax = plt.subplots(figsize=(12, 12,))
    az.plot_ecdf(np.asarray(duration_pred_0900_nashua).copy(), ax=ax)
    fig.savefig("9am_nashua_duration_cdf.png")
    # Plot hypothetical journeys
    duration_pred = pred(rng_key, buses_le.transform(buses_le.classes_))["obs"]
    mean_pred = pd.DataFrame(duration_pred.mean(axis=0), index=buses_le.classes_)
    pred_df = pd.DataFrame(duration_pred.T, index=buses_le.classes_).melt(ignore_index=False, var_name="iteration", value_name="duration")
    bus_table = bus_dset[['bus_label', 'station', 'departure', 'time.departure']].drop_duplicates().sort_values(by="time.departure")
    bus_table = bus_table.set_index("bus_label")
    pred_df = pred_df.join(bus_table).reset_index(names="bus_label")
    fig, ax = plt.subplots(figsize=(12, 12,))
    sns.violinplot(pred_df, x="bus_label", y="duration", hue="station",
                   inner=None, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    fig.savefig("violins_hypothetical_journeys.png")
    bus_summary = bus_dset.groupby("bus_label")["duration"].agg(["mean", "count"]).join(mean_pred.set_axis(["simulated"], axis=1))
    return bus_summary

def arrive_by(pred_df, arrival):
    arr = time.strptime(arrival, "%H:%M")
    time_arrival = arr.tm_hour * 3600 + arr.tm_min * 60 + arr.tm_sec
    pred_df["time.arrival"] = pred_df["time.departure"] + 60 * pred_df["duration"]
    pred_df["ontime"] = pred_df["time.arrival"] < time_arrival
    return pred_df.groupby("bus_label")["ontime"].mean()

def main():
    # Data setup
    bus_dset = pd.read_csv("bus-data.csv")
    bus_dset["bus_label"] = bus_dset["departure"].str.cat(bus_dset["station"], sep="_")
    buses_le = LabelEncoder()
    buses_ix = buses_le.fit_transform(bus_dset['bus_label'])
    stations_le = LabelEncoder()
    stations_ix = stations_le.fit_transform(bus_dset['bus_label'])
     # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(0)

    rng_key, rng_key_ = random.split(rng_key)
    mcmc_simple = fit_simple_model(
            rng_key_,
            buses_ix,
            bus_dset.duration.values,
            )

    rng_key, rng_key_ = random.split(rng_key)
    summary_plots(rng_key_, mcmc_simple, bus_dset, buses_le)

    # rng_key, rng_key_ = random.split(rng_key)
    # mcmc_rush_hour = fit_rush_hour_model(bus_dset, rng_key_)

    rng_key, rng_key_ = random.split(rng_key)
    mcmc_gp = fit_gp_model(rng_key_, stations_ix, bus_dset["time.departure"].values, bus_dset.duration.values)

    return (
            mcmc_simple,
            # mcmc_rush_hour,
            mcmc_gp,
            )

if __name__ == "__main__":
    (mcmc_simple, mcmc_gp, ) = main()

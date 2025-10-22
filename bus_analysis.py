import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import random, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
from numpyro import handlers
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

dset = pd.read_csv("bus-data.csv")

def simple_model(station, departure, duration):
    stations = station.unique()
    departures = departure.unique()
    bus = [(stat, dep) for stat in stations for dep in departures]
    with numpyro.plate("bus", len(bus)):
        mu = numpyro.sample("mu", dist.Normal(90, 90))
    sigma = numpyro.sample("sigma", dist.HalfNormal(40))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=duration)

def model(station, departure, duration):
    region_base = numpyro.sample("region_base", dist.Normal(50, 20))
    hwy_sigma = numpyro.sample("hwy_sigma", dist.HalfNormal(20))
    am_rush_hour_start = numpyro.sample("am_rush_hour_start", dist.Uniform(0, 34800))
    am_rush_hour_length = numpyro.sample("am_rush_hour_end", dist.HalfNormal(7200))
    am_rush_hour_onset = numpyro.sample("am_rush_hour_onset",
                                             dist.HalfNormal(1.0/2e3))
    am_rush_hour_fade = numpyro.sample("am_rush_hour_fade",
                                             dist.HalfNormal(1.0/2e3))
    pm_rush_hour_start = numpyro.sample("pm_rush_hour_start",
                                        dist.Uniform(55800, 64800))
    pm_rush_hour_length = numpyro.sample("pm_rush_hour_end", dist.HalfNormal(7200))
    pm_rush_hour_onset = numpyro.sample("pm_rush_hour_onset",
                                             dist.HalfNormal(1.0/2e3))
    pm_rush_hour_fade = numpyro.sample("pm_rush_hour_fade",
                                             dist.HalfNormal(1.0/2e3))
    am_rush = (1 / (1 + jnp.exp(-am_rush_hour_onset * (departure -
                                                         am_rush_hour_start)))
            / (1 + jnp.exp(-am_rush_hour_fade * (am_rush_hour_end
                                                      - departure))))
    pm_rush = (1 / (1 + jnp.exp(-pm_rush_hour_onset * (departure -
                                                     pm_rush_hour_start)))
             / (1 + jnp.exp(-pm_rush_hour_fade * (pm_rush_hour_end
                                                  - departure))))
    am_rush_penalty = numpyro.sample("am_rush_penalty", dist.HalfNormal(7200))
    pm_rush_penalty = numpyro.sample("pm_rush_penalty", dist.HalfNormal(7200))
    sigma = numpyro.sample("sigma", dist.Exponential(1/1800.))

    n_stations = 2
    with numpyro.plate("station", n_stations):
        hwy_base = numpyro.sample(
                "hwy_base",
                dist.Normal(region_base, hwy_sigma))
    mu = (region_base
          + hwy_base[station]
          + am_rush * am_rush_penalty
          + pm_rush * pm_rush_penalty)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=duration)

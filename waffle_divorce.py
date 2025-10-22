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

DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
dset = pd.read_csv(DATASET_URL, sep=";")

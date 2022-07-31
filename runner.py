import jaxlib
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from matplotlib import pyplot as plt

from model import min_objective
from utils import random_weighted_dag, sample_covariance

dim = 5
sparsity = .65
N_samples = 1000
iters = 1000
step_size = .1
spar_const = .1
DAG_const = .1

noise_cov = np.diag([.5, 2, 1, 5, 1])

dag = random_weighted_dag(dim, sparsity)

data, sample_cov = sample_covariance(dag, noise_cov, N_samples)

omega = np.linalg.inv(sample_cov)

A_init = omega

dag_est, logger = min_objective(data, omega, omega, step_size, iters, spar_const, DAG_const)

plt.plot(logger.loss)
plt.show()

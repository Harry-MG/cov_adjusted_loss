import jaxlib
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from matplotlib import pyplot as plt

from model import min_objective, min_alt_objective, min_B_objective
from utils import random_weighted_dag, sample_covariance

dim = 4
sparsity = .4
N_samples = 20
iters = 1000
step_size = .1
spar_const = .1
DAG_const = .001

noise_cov = np.diag([.5, 2, 1, 5])

dag = random_weighted_dag(dim, sparsity)
rand_perm = np.random.permutation(dim)
P = np.eye(dim)
P[list(range(dim))] = P[list(rand_perm)]
dag = P @ dag @ np.transpose(P)  # now dag represents a DAG not necessarily in topological order

data, sample_cov = sample_covariance(dag, noise_cov, N_samples)

omega = np.linalg.inv(sample_cov)

A_init = omega

B_init = jnp.eye(dim) - omega

dag_est, logger = min_B_objective(data, B_init, omega, step_size, iters, spar_const, DAG_const)

plt.plot(logger.loss)
plt.show()

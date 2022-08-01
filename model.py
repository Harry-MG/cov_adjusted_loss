import jaxlib
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from matplotlib import pyplot as plt

from utils import loss_fn, objective, DAG_pen, sparsity_pen, alt_objective


class Logs:
    def __init__(self, loss, l1_norm, DAGness):
        self.loss = loss
        self.l1_norm = l1_norm
        self.DAGness = DAGness


def min_objective(data, A_init, omega, step_size, iters, spar_const, DAG_const):
    A = A_init  # initialise A

    loss = np.zeros(iters)
    l1_norm = np.zeros(iters)
    DAGness = np.zeros(iters)

    for t in range(iters):
        A = A - step_size * jax.grad(objective, argnums=0)(A, omega, data, spar_const, DAG_const)
        print(A)

        loss[t] = loss_fn(A, omega, data)
        l1_norm[t] = sparsity_pen(A)
        DAGness[t] = DAG_pen(A)

    logger = Logs(loss=loss, l1_norm=l1_norm, DAGness=DAGness)
    return A, logger


def min_alt_objective(data, A_init, omega, step_size, iters, spar_const, DAG_const):
    A = A_init  # initialise A

    loss = np.zeros(iters)
    l1_norm = np.zeros(iters)
    DAGness = np.zeros(iters)

    for t in range(iters):
        A = A - step_size * jax.grad(alt_objective, argnums=0)(A, omega, data, spar_const, DAG_const)
        print(A)

        loss[t] = loss_fn(A, omega, data)
        l1_norm[t] = sparsity_pen(A)
        DAGness[t] = DAG_pen(A)

    logger = Logs(loss=loss, l1_norm=l1_norm, DAGness=DAGness)
    return A, logger

import jaxlib
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from matplotlib import pyplot as plt

from utils import loss_fn, objective, DAG_pen, sparsity_pen, alt_objective, B_objective, B_loss_fn, diag_objective, \
    diag_loss_fn


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


def min_B_objective(data, B_init, omega, step_size, iters, spar_const, DAG_const):
    n = np.shape(omega)[0]
    B = B_init  # initialise B (usually as I - omega)

    loss = np.zeros(iters)
    l1_norm = np.zeros(iters)
    DAGness = np.zeros(iters)

    for t in range(iters):

        print('det(B) = '+str(jnp.linalg.det(B)))
        print('loss = '+str(B_loss_fn(B, omega, data)))
        print('DAG pen = '+str(DAG_pen(jnp.eye(n) - B)))
        print('l1 pen = ' + str(sparsity_pen(jnp.eye(n) - B)))

        B = B - step_size * jax.grad(B_objective, argnums=0)(B, omega, data, spar_const, DAG_const)

        loss[t] = B_loss_fn(B, omega, data)
        l1_norm[t] = sparsity_pen(jnp.eye(n) - B)
        DAGness[t] = DAG_pen(jnp.eye(n) - B)

    logger = Logs(loss=loss, l1_norm=l1_norm, DAGness=DAGness)
    return B, logger


def min_objective_fn(objective_fn, data, B_init, omega, step_size, iters, spar_const, DAG_const):
    n = np.shape(omega)[0]
    B = B_init  # initialise B (usually as I - omega)

    loss = np.zeros(iters)
    l1_norm = np.zeros(iters)
    DAGness = np.zeros(iters)

    for t in range(iters):

        print('det(B) = '+str(jnp.linalg.det(B)))
        print('loss = '+str(B_loss_fn(B, omega, data)))
        print('DAG pen = '+str(DAG_pen(jnp.eye(n) - B)))
        print('l1 pen = ' + str(sparsity_pen(jnp.eye(n) - B)))

        B = B - step_size * jax.grad(objective_fn, argnums=0)(B, omega, data, spar_const, DAG_const)

        loss[t] = B_loss_fn(B, omega, data)
        l1_norm[t] = sparsity_pen(jnp.eye(n) - B)
        DAGness[t] = DAG_pen(jnp.eye(n) - B)

    logger = Logs(loss=loss, l1_norm=l1_norm, DAGness=DAGness)
    return B, logger


def min_diag_objective(data, B_init, omega, step_size, iters, spar_const, DAG_const):
    n = np.shape(omega)[0]
    B = B_init  # initialise B (usually as I - omega)

    loss = np.zeros(iters)
    l1_norm = np.zeros(iters)
    DAGness = np.zeros(iters)

    for t in range(iters):

        print('det(B) = '+str(jnp.linalg.det(B)))
        print('loss = '+str(diag_loss_fn(B, omega, data)))
        print('DAG pen = '+str(DAG_pen(jnp.eye(n) - B)))
        print('l1 pen = ' + str(sparsity_pen(jnp.eye(n) - B)))

        B = B - step_size * jax.grad(diag_objective, argnums=0)(B, omega, data, spar_const, DAG_const)

        loss[t] = diag_loss_fn(B, omega, data)
        l1_norm[t] = sparsity_pen(jnp.eye(n) - B)
        DAGness[t] = DAG_pen(jnp.eye(n) - B)

    logger = Logs(loss=loss, l1_norm=l1_norm, DAGness=DAGness)
    return B, logger


def min_B_loss(data, B_init, omega, step_size, iters):
    B = B_init  # initialise B (usually as I - omega)

    loss = np.zeros(iters)

    for t in range(iters):

        print('det(B) = '+str(jnp.linalg.det(B)))
        print('loss = '+str(B_loss_fn(B, omega, data)))

        B = B - step_size * jax.grad(B_loss_fn, argnums=0)(B, omega, data)

        loss[t] = B_loss_fn(B, omega, data)

    return B, loss

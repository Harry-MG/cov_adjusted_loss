import jaxlib
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from matplotlib import pyplot as plt


def loss_fn(A, omega, data):
    """
    Loss function (1 / 2 * n_samples)||D^{-1/2} (data - A @ data)||_F^2, where D^{-1} = (I-A)^{-T} @ omega @ (I-A)^{-1}.

    Args:
        A (np.ndarray): matrix A as above.
        omega (np.ndarray): matrix omega as above. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix as above.

     Returns:
        loss (float): value of loss function.

    """
    n = jnp.shape(A)[0]
    n_samples = jnp.shape(data)[1]
    B = jnp.eye(n) - A
    C = jnp.linalg.inv(B)
    D_inv = C.T @ omega @ C
    loss = jnp.linalg.norm(jsp.linalg.sqrtm(D_inv) @ B @ data) ** 2 / (2 * n_samples)
    return loss


def alt_loss_fn(A, omega, data):
    """
    Loss function (1 / 2*n_samples)||D^{-1/2} (data - A @ data)||_F^2, where D^{-1} = (I-A)^{-T} @ omega @ (I-A)^{-1}.

    Args:
        A (np.ndarray): matrix A as above.
        omega (np.ndarray): matrix omega as above. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix as above.

     Returns:
        loss (float): value of loss function.

    """
    n = jnp.shape(A)[0]
    n_samples = jnp.shape(data)[1]
    B = jnp.eye(n) - A
    C = jnp.linalg.inv(B)
    D_inv = C.T @ omega @ C
    evals, evecs = jnp.linalg.eigh(D_inv)
    sq = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T  # D_inv is positive-definite
    loss = jnp.linalg.norm(sq @ B @ data) ** 2 / (2 * n_samples)
    return loss


def B_loss_fn(B, omega, data):
    """
    Loss function (1 / 2 * n_samples)||D^{-1/2} @ B @ data||_F^2, where D^{-1} = B^{-T} @ omega @ B^{-1}.

    Args:
        B (np.ndarray): matrix B as above. B = I - A where A is the DAG in the usual objective.
        omega (np.ndarray): matrix omega as above. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix as above.

    Returns:
        loss (float): value of loss function.

    """
    n_samples = jnp.shape(data)[1]
    C = jnp.linalg.inv(B)
    D_inv = C.T @ omega @ C
    evals, evecs = jnp.linalg.eigh(D_inv)
    sq = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T  # D_inv is positive-definite
    loss = (jnp.linalg.norm(sq @ B @ data) ** 2) / (2 * n_samples)
    return loss


def diag_loss_fn(B, omega, data):
    """
    Loss function (1 / 2 * n_samples)||D^{-1/2} @ B @ data||_F^2, where D^{-1} = diag(B^{-T} @ omega @ B^{-1}).

    Args:
        B (np.ndarray): matrix B as above. B = I - A where A is the DAG in the usual objective.
        omega (np.ndarray): matrix omega as above. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix as above.

    Returns:
        loss (float): value of loss function.

    """
    n_samples = jnp.shape(data)[1]
    C = jnp.linalg.inv(B)
    D_inv = C.T @ omega @ C
    sq = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(D_inv), jnp.zeros(jnp.shape(B)[0]))))
    loss = (jnp.linalg.norm(sq @ B @ data) ** 2) / (2 * n_samples)
    return loss


def sparsity_pen(A):
    """
    Sparsity penalisation term ||A||_1.

    Args:
        A (np.ndarray)

    Returns:
         (float): value of sparsity penalisation.

    """
    return jnp.linalg.norm(A, 1)


def DAG_pen(A):
    """
    DAG-ness penalisation term (as in NOTEARS): trace(exp(A * A)) - dim(A).

    Args:
        A (np.ndarray)

    Returns:
         (float): value of DAG penalisation.

    """
    n = np.shape(A)[0]
    return jnp.trace(jsp.linalg.expm(A * A)) - n


def objective(A, omega, data, spar_const, DAG_const):
    """
    Objective function to minimise. Loss function plus sparsity and DAG-ness penalisation terms.

    Args:
        A (np.ndarray): matrix A as main objective argument.
        omega (np.ndarray): matrix omega. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix.
        spar_const (float): positive coefficient on sparsity penalisation term.
        DAG_const (float): positive coefficient on DAG-ness penalisation term.

    Returns:
         (float): value of objective function.

    """
    return loss_fn(A, omega, data) + spar_const * sparsity_pen(A) + DAG_const * DAG_pen(A)


def alt_objective(A, omega, data, spar_const, DAG_const):
    """
    Objective function to minimise. Loss function plus sparsity and DAG-ness penalisation terms.

    Args:
        A (np.ndarray): matrix A as main objective argument.
        omega (np.ndarray): matrix omega. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix.
        spar_const (float): positive coefficient on sparsity penalisation term.
        DAG_const (float): positive coefficient on DAG-ness penalisation term.

    Returns:
         (float): value of objective function.

    """
    return alt_loss_fn(A, omega, data) + spar_const * sparsity_pen(A) + DAG_const * DAG_pen(A)


def B_objective(B, omega, data, spar_const, DAG_const):
    """
    Objective function to minimise. Loss function plus sparsity and DAG-ness penalisation terms.

    Args:
        B (np.ndarray): matrix B as main objective argument.
        omega (np.ndarray): matrix omega. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix.
        spar_const (float): positive coefficient on sparsity penalisation term.
        DAG_const (float): positive coefficient on DAG-ness penalisation term.

    Returns:
         (float): value of objective function.

    """
    n = np.shape(B)[0]
    idty = jnp.eye(n)
    return B_loss_fn(B, omega, data) + spar_const * sparsity_pen(idty - B) + DAG_const * DAG_pen(idty - B)


def diag_objective(B, omega, data, spar_const, DAG_const):
    """
    Objective function to minimise. Loss function plus sparsity and DAG-ness penalisation terms.

    Args:
        B (np.ndarray): matrix B as main objective argument.
        omega (np.ndarray): matrix omega. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix.
        spar_const (float): positive coefficient on sparsity penalisation term.
        DAG_const (float): positive coefficient on DAG-ness penalisation term.

    Returns:
         (float): value of objective function.

    """
    n = np.shape(B)[0]
    idty = jnp.eye(n)
    return diag_loss_fn(B, omega, data) + spar_const * sparsity_pen(idty - B) + DAG_const * DAG_pen(idty - B)


def noloss_objective(B, omega, spar_const, DAG_const):
    """
    Objective function to minimise. Loss function plus sparsity and DAG-ness penalisation terms.

    Args:
        B (np.ndarray): matrix B as main objective argument.
        omega (np.ndarray): matrix omega. An estimate of the inverse covariance from the data.
        data (np.ndarray): data matrix.
        spar_const (float): positive coefficient on sparsity penalisation term.
        DAG_const (float): positive coefficient on DAG-ness penalisation term.

    Returns:
         (float): value of objective function.

    """
    C = jnp.linalg.inv(B)
    D_inv = C.T @ omega @ C
    offdiag = jnp.linalg.norm(D_inv - jnp.diag(D_inv)) ** 2
    return offdiag + spar_const * DAG_pen(B) + DAG_const * DAG_pen(B)


def random_dag(dim, sparsity):
    """
    Generates random binary upper triangular matrix with given upper triangular sparsity, representing the
    adjacency matrix of a DAG

    Args:
        dim (int): dimension of matrix / number of nodes in DAG.
        sparsity (float): expected sparsity of upper triangular part. in (0, 1).

     Returns:
        dag (np.ndarray): adjacency matrix of a dag

    """

    A = np.random.rand(dim, dim)

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    # make binary

    A[np.abs(A) > 0] = 1

    return A


def random_weighted_dag(dim, sparsity):
    """
    Generate random weighted (weights in (.1, .9)) upper triangular matrix with given upper triangular sparsity,
    representing the adjacency matrix of a DAG

    Args:
        dim (int): dimension of matrix / number of nodes in DAG.
        sparsity (float): expected sparsity of upper triangular part. in (0, 1).

    Returns:
        dag (np.ndarray): weighted adjacency matrix of a dag

    """

    A = np.random.rand(dim, dim)  # threshold the matrix entries in [.1, .9]

    A = A + (0.1 / (0.9 - 0.1)) * np.ones((dim, dim))

    A = A * (0.9 - 0.1)

    zero_indices = np.random.choice(np.arange(A.size), replace=False,
                                    size=int(A.size * sparsity))

    A[np.unravel_index(zero_indices, A.shape)] = 0

    A = np.transpose(np.tril(A))

    A = A - np.diag(np.diag(A))

    return A


def sample_covariance(dag, noise_cov, N):
    """
    Produce sample covariance matrix from samples of linear SEM X = AX + Z

    Args:
        dag (np.ndarray): DAG adjacency matrix A in the linear SEM
        noise_cov (np.ndarray): diagonal noise covariance matrix Z
        N (int): number of samples of the SEM used to produce the sample matrix

    Returns:
        invcov (np.ndarray): resulting sample covariance matrix
    """
    n = np.shape(dag)[0]
    data = np.zeros((n, N))
    for i in range(N):
        noise = np.random.multivariate_normal(mean=np.zeros(n), cov=noise_cov)
        X = np.linalg.inv(np.eye(n) - dag) @ noise
        data[:, i] = X
    covmat = np.cov(data)

    return data, covmat

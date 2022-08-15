import scipy.linalg
import numpy as np
from scipy.optimize import shgo


def argument(B, omega):
    C = np.linalg.inv(B)
    D_inv = C.T @ omega @ C  # D_inv is SPD
    inn = scipy.linalg.sqrtm(D_inv) @ B
    return inn

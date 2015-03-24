import numpy as np

def make_landau_matrix():

    # Define the frenel number.
    F = 8

    # Define the dimension.
    N = 250

    # Nodes and weights for Gaussian quadrature.
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # construct matrix B
    B = np.zeros((N, N), dtype=complex)
    for k in range(N):
        W = -1j * np.pi * F * np.square(nodes[k] - nodes.conj())
        B[k, :] = weights.conj() * np.sqrt(1j * F) * np.exp(W)

    # Weight the matrix with Gaussian quadrature weights.
    w = np.sqrt(weights)
    for j in range(N):
        B[:, j] = w * B[:, j] / w[j]

    return B

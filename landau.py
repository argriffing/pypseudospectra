"""
Computation of Pseudospectra
Lloyd N. Trefethen
https://people.maths.ox.ac.uk/trefethen/publication/PDF/1999_83.pdf

"So far as I know, the first person to define the notion of pseudospectra
was Henry Landau at Bell Laboratories in the 1970s, who was motivated
in part by applications in lasers and optical resonators."

A complex symmetric but not hermitian compact integral operator.

http://www.cs.ox.ac.uk/projects/pseudospectra/thumbnails/landau.html

"""
from __future__ import print_function, division

from functools import partial

import numpy as np
from numpy.testing import assert_equal
import scipy.linalg
from scipy.sparse.linalg.isolve.lsqr import _sym_ortho

import matplotlib.pyplot as plt
from matplotlib import colors, cm

def gaussian_nodes_and_weights(N):
    # Nodes and weights for Gaussian quadrature.
    # Nodes are eigenvalues.
    # Weights are somehow related to eigenvectors.
    U = np.arange(1, N) # Is this [1:N-1] in Matlab?
    beta = 0.5 * (1 - (2 * U)**(-2))**(-0.5)
    T = np.diag(beta, k=1) + np.diag(beta, k=-1)
    w, v = scipy.linalg.eigh(T)
    nodes = w
    #weights = 2 * np.square(v[:, 0])
    weights = 2 * np.square(v[0, :])
    return nodes, weights


def make_landau_matrix():

    # Define the frenel number.
    F = 8

    # Define the dimension.
    N = 250

    # Nodes and weights for Gaussian quadrature.
    # Nodes are eigenvalues.
    # Weights are somehow related to eigenvectors.
    #U = np.arange(1, N) # Is this [1:N-1] in Matlab?
    #beta = 0.5 * (1 - (2 * U)**(-2))**(-0.5)
    #T = np.diag(beta, k=1) + np.diag(beta, k=-1)
    #w, v = scipy.linalg.eigh(T)
    #nodes = w
    ##weights = 2 * np.square(v[:, 0])
    #weights = 2 * np.square(v[0, :])
    #print(weights)

    #nodes, weights = gaussian_nodes_and_weights(N)
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # construct matrix B
    B = np.zeros((N, N), dtype=complex)
    for k in range(N):
        W = -1j * np.pi * F * np.square(nodes[k] - nodes.conj())
        B[k, :] = weights.conj() * np.sqrt(1j * F) * np.exp(W)

    # Weight the matrix with Gaussian quadrature weights.
    w = np.sqrt(weights)
    for j in range(N):
        #B[:, j] = w * B[:, j] / w[j]
        B[:, j] = w * B[:, j] / w[j]

    return B

    # Compute Schur form and compress to interesting subspace.

    # This Matlab function returns U and T so that B = U.dot(T).dot(U.H)
    # and U.H.dot(U) = identity, and T is a Schur matrix.
    #U, T = schur(B)

    # Do the same thing using scipy instead of Matlab.
    T, Z = scipy.linalg.schur(B)
    U = Z

    # Something about a subspace.
    # Maybe the idea is to use only the subspace involving
    # eigenpairs whose eigenvalues have magnitude greater than 0.1?
    eigB = np.diag(T)
    indices, = np.where(abs(eigB) > 0.0001)
    #indices, = np.where(abs(eigB) > 0.1)
    n = indices.shape[0]
    for i in range(n):
        #for k in select(i) - 1 : -1 : i:
        for k in range(indices[i]-1, i-1, -1):
            a = T[k, k+1]
            b = T[k, k] - T[k+1, k+1]
            x = np.array([a, b])
            G, r = planerot(x)
            J = np.arange(k, k+2)
            T[:, J] = T[:, J].dot(G)
            T[J, :] = G.conj().T.dot(T[J, :])
    T = np.triu(T[:n+1, :n+1])

    return T


"""
def planerot(x):
    a, b = x
    r = np.hypot(a, b)
    c = a / r
    s = b / r
    G = np.array([
        [c, s],
        [s, -c]])
    #G = np.array([
        #[c, s],
        #[-s, c]])
    return G, r
"""

def planerot(x):
    # This is like symGivens2 in pylearn2 and planerot in Matlab.
    a, b = x
    c, s, r = _sym_ortho(a, b)
    G = np.array([
        [c, s],
        [s, -c]])
    return G, r


def test_planerot():
    """
    Check a Matlab function vs. a Python equivalent.

    x = [3 4]
    [G, y] = planerot(x')
    G =
        0.6  0.8
       -0.8  0.6
    y =
        5
        0

    """
    x = np.array([3, 4], dtype=float)
    #G, r = planerot(x)
    G, r = sym_ortho(x)
    print('x:', x)
    print('G:', G)
    print('r:', r)
    print('Gx:', np.dot(G, x))

    

def resolvent(A, z):
    # A: matrix
    # z: complex number
    n, m = A.shape
    assert_equal(n, m)
    I = np.eye(n, dtype=float)
    B = np.linalg.inv(z*I - A)
    return B


def resolvent_infnorm(A, z):
    try:
        B = resolvent(A, z)
    except np.linalg.LinAlgError as e:
        return 0
    return np.linalg.norm(B, ord=np.inf)


def resolvent_onenorm(A, z):
    try:
        B = resolvent(A, z)
    except np.linalg.LinAlgError as e:
        return 0
    return np.linalg.norm(B, ord=1)


def check_gaussian_nodes_and_weights(N):
    nodes, weights = gaussian_nodes_and_weights(N)

    print('custom gaussian nodes and weights')
    print('nodes:')
    print(nodes)
    print('weights:')
    print(weights)
    print()

    print('scipy gaussian nodes and weights')
    nodes, weights = np.polynomial.legendre.leggauss(N)
    print('nodes:')
    print(nodes)
    print('weights:')
    print(weights)
    print()


def main():

    #check_gaussian_nodes_and_weights(10)
    #return

    #a = np.array([_eulerian(n, i) for i in range(1, n+1)], dtype=float)
    #log_a = np.log(a)
    A = make_landau_matrix()

    #print('row sums of probability matrix:')
    #print(P.sum(axis=1))

    #pi = np.exp(log_a - gammaln(n+1))

    #print('sum of stationary probabilities:', pi.sum())
    #A = P - pi

    print('creating the figure...')
    figure(None, A)


def figure(t, A):

    levels = np.power(10, np.linspace(1, 10, 19))
    if t is None:
        #f = np.vectorize(partial(resolvent_infnorm, A))
        f = np.vectorize(partial(resolvent_onenorm, A))
    else:
        pass
        #f = np.vectorize(partial(resolvent_infnormest, t, A))

    low = -1.5
    high = 1.5
    #u = np.linspace(low, high, 81)
    #u = np.linspace(low, high, 101)
    #u = np.linspace(low, high, 41)
    u = np.linspace(low, high, 201)
    #u = np.linspace(low, high, 21)
    X, Y = np.meshgrid(u, u)
    z = u[np.newaxis, :] + 1j*u[:, np.newaxis]
    Z = f(z)
    print(Z)

    print('eigenvalues of decay matrix:')
    print(scipy.linalg.eigvals(A))

    # Add contour lines at predefined levels.
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.contour( X, Y, Z, levels=levels, colors='k')

    # Add dashed unit circle.
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    fig.gca().add_artist(circle1)

    plt.show()


main()

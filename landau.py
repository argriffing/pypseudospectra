"""
Computation of Pseudospectra
Lloyd N. Trefethen
https://people.maths.ox.ac.uk/trefethen/publication/PDF/1999_83.pdf

"So far as I know, the first person to define the notion of pseudospectra
was Henry Landau at Bell Laboratories in the 1970s, who was motivated
in part by applications in lasers and optical resonators."

A complex symmetric but not hermitian compact integral operator.

http://www.cs.ox.ac.uk/projects/pseudospectra/thumbnails/landau.html

block algorithm for matrix 1-norm estimation
with an application to 1-norm pseudospectra
http://eprints.ma.man.ac.uk/321/01/covered/MIMS_ep2006_145.pdf

"""
from __future__ import print_function, division

from functools import partial

import numpy as np
from numpy.testing import assert_equal
import scipy.linalg
from scipy.sparse.linalg.isolve.lsqr import _sym_ortho

from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator

import matplotlib.pyplot as plt
from matplotlib import colors, cm



def get_landau_resolvent_operator(T, Z, eps_recip):
    n = T.shape[0]
    T = np.diag(np.ones(n) * eps_recip) - T
    return LandauResolventOperator(T, Z)


class LandauResolventOperator(LinearOperator):

    def __init__(self, M, Z, lower=False):
        self.M = M
        self.Z = Z
        self.lower = lower
        self.shape = M.shape

    def _matmat(self, B):
        M = self.M
        Z = self.Z
        ZH = self.Z.conj().T
        C = scipy.linalg.solve_triangular(M, Z.dot(B), lower=self.lower)
        return ZH.dot(C)

    def _transpose(self):
        return LandauResolventOperator(self.M.T, self.Z,
                lower=(not self.lower))

    def _adjoint(self):
        return LandauResolventOperator(self.M.conj().T, self.Z,
                lower=(not self.lower))


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


def resolvent_onenorm(A, z):
    try:
        B = resolvent(A, z)
    except np.linalg.LinAlgError as e:
        return 0
    return np.linalg.norm(B, ord=1)


def resolvent_onenormest(t, T, Z, eps_recip):
    op = get_landau_resolvent_operator(T, Z, eps_recip)
    return scipy.sparse.linalg.onenormest(op, t=t)


def main():

    A = make_landau_matrix()

    print('creating the figures...')
    t = 8
    figure(t, A)
    #for t in None, 1, 2, 4, 16:
        #print('t =', t, '...')
        #figure(t, A)


def figure(t, A):

    if t is None:
        filename = 'landau.svg'
    else:
        filename = 'landau_t_%d.svg' % t

    levels = np.power(10, np.linspace(1, 10, 19))
    if t is None:
        f = np.vectorize(partial(resolvent_onenorm, A))
    else:
        T, Z = scipy.linalg.schur(A)
        f = np.vectorize(partial(resolvent_onenormest, t, T, Z))

    low = -1.5
    high = 1.5
    #grid_count = 201
    grid_count = 100
    u = np.linspace(low, high, grid_count)
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

    #plt.show()
    plt.savefig(filename)


main()

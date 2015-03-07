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


from resolvent import resolvent_onenorm, resolvent_onenormest


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
    

"""
def resolvent(A, z):
    # A: matrix
    # z: complex number
    n, m = A.shape
    assert_equal(n, m)
    I = np.eye(n, dtype=float)
    B = np.linalg.inv(z*I - A)
    return B
"""




def main():

    A = make_landau_matrix()
    #t_values = [2]
    t_values = [None, 1, 2, 4, 8, 16]
    base_filename = 'landau.new'

    print('creating the figures for t in', t_values, '...')
    for t in t_values:
        print('t =', t, '...')
        figure(base_filename, t, A)


def figure(base_filename, t, A):

    if t is None:
        filename = base_filename + '.svg'
    else:
        filename = base_filename + '.' + str(t) + '.svg'

    levels = np.power(10, np.linspace(1, 10, 19))

    # The outputs of the Schur decomposition are:
    # T : a triangular matrix, upper triangular by default.
    # Z : A unitary matrix
    T, Z = scipy.linalg.schur(A, output='complex')

    # Check the decomposition.
    #ZH = Z.conj().T
    #W = Z.dot(T).dot(ZH)
    #error = np.linalg.norm(W - A)
    #print('norm of original matrix:', np.linalg.norm(A))
    #print('norm of reconstructed matrix:', np.linalg.norm(W))
    #print('norm of difference:', error)

    #raise Exception

    if t is None:
        f = np.vectorize(partial(resolvent_onenorm, T, Z))
    else:
        f = np.vectorize(partial(resolvent_onenormest, t, T, Z))

    low = -1.5
    high = 1.5
    grid_count = 201
    #grid_count = 100
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

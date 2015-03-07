"""
Riffle shuffle.

Reference:
A Numerical Analyst Looks at the Cutoff Phenomenon
in Card Shuffling and Other Markov Chains
Gudbjorn F. Jonsson and Lloyd N. Trefethen
http://eprints.maths.ox.ac.uk/1313/1/NA-97-12.pdf

one-norm estimation for pseudospectra:
http://eprints.ma.man.ac.uk/321/01/covered/MIMS_ep2006_145.pdf

"""
from __future__ import print_function, division

from functools import partial

import numpy as np
from numpy.testing import assert_equal
from scipy.special import gammaln
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core

import matplotlib.pyplot as plt
from matplotlib import colors, cm

from util import memoized
from resolvent import resolvent_onenorm, resolvent_onenormest


#@lru_cache()
@memoized
def _eulerian_recurrence(r, k):
    if (r, k) == (1, 1):
        return 1
    elif r == 1:
        return 0
    else:
        a = k * _eulerian_recurrence(r-1, k)
        b = (r - k + 1) * _eulerian_recurrence(r-1, k-1)
        return a + b

#@lru_cache()
@memoized
def _eulerian(n, r):
    return _eulerian_recurrence(n, r)

def log_binom(a, b):
    return gammaln(a+1) - gammaln(b+1) - gammaln(a-b+1)


def riffle_transition_matrix(n):
    # Eq. (7.2)
    a = np.array([_eulerian(n, i) for i in range(n+1)], dtype=float)
    log_a = np.log(a)
    log_P = np.zeros((n+1, n+1), dtype=float)
    for i in range(1, n+1):
        for j in range(1, n+1):
            log_P[i, j] -= n * np.log(2)
            log_P[i, j] += log_binom(n+1, 2*i - j)
            log_P[i, j] += log_a[j] - log_a[i]
    return np.exp(log_P[1:, 1:])

"""
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


def resolvent_onenormest(t, A, z):
    # TODO use decomposition instead of inversion
    try:
        #n = A.shape[0]
        #I = np.eye(n, dtype=float)
        #Binv = z*I - A
        B = resolvent(A, z)
        L = scipy.sparse.linalg.aslinearoperator(B)
        return scipy.sparse.linalg.onenormest(L, t=t)
    except np.linalg.LinAlgError as e:
        return 0
"""


def main():
    n = 52
    a = np.array([_eulerian(n, i) for i in range(1, n+1)], dtype=float)
    log_a = np.log(a)
    P = riffle_transition_matrix(n)

    print('row sums of probability matrix:')
    print(P.sum(axis=1))

    pi = np.exp(log_a - gammaln(n+1))

    print('sum of stationary probabilities:', pi.sum())
    A = P - pi

    print('creating the figure...')
    for t in 1, 2, 3:
        figure_7_2(t, A.T)
    #figure_7_2(None, A)


def figure_7_2(t, A):
    """
    Figure 7.2.

    """
    levels = np.power(10, [1, 1.5, 2, 2.5, 3, 3.5, 4])

    # The outputs of the Schur decomposition are:
    # T : a triangular matrix, upper triangular by default.
    # Z : A unitary matrix
    T, Z = scipy.linalg.schur(A, output='complex')

    if t is None:
        f = np.vectorize(partial(resolvent_onenorm, T, Z))
    else:
        f = np.vectorize(partial(resolvent_onenormest, t, T, Z))

    low = -1.5
    high = 1.5
    u = np.linspace(low, high, 201)
    #u = np.linspace(low, high, 101)
    X, Y = np.meshgrid(u, u)
    z = u[np.newaxis, :] + 1j*u[:, np.newaxis]
    Z = f(z)
    print(Z)

    print('eigenvalues of decay matrix:')
    print(scipy.linalg.eigvals(A))

    # Add contour lines at predefined levels.
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.contour(X, Y, Z, levels=levels, colors='k')

    # Add dashed unit circle.
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    fig.gca().add_artist(circle1)

    #plt.show()

    if t is None:
        filename = 'riffle.svg'
    else:
        filename = 'riffle_t_%d.svg' % t
    plt.savefig(filename)


main()

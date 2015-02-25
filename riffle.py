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

#from functools import lru_cache, partial
from functools import partial
import collections

from math import factorial

import numpy as np
from numpy.testing import assert_equal
import scipy.misc
from scipy.special import gammaln
import scipy.linalg

import matplotlib.pyplot as plt
from matplotlib import colors, cm

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
             # uncacheable. a list, for instance.
             # better to not cache than blow up.
             return self.func(*args)
        if args in self.cache:
             return self.cache[args]
        else:
             value = self.func(*args)
             self.cache[args] = value
             return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def eulerian_logs(n):
    # from the appendix
    # seems buggy
    a = np.zeros(n)
    anew = np.zeros(n)
    for j in range(2, n):
        x = np.arange(1, j-1)
        y = np.exp(a[1:j-1] - a[:j-2])
        z = np.arange(j-1, 1, -1)
        print(x.shape)
        print(y.shape)
        print(z.shape)
        u = x * y + z
        anew[1:j-1] = np.log(u) + a[:j-2]
        a = anew
    return a


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
    #a = np.array([_eulerian(n, i) for i in range(n+0)], dtype=float)
    log_a = np.log(a)
    log_P = np.zeros((n+1, n+1), dtype=float)
    #log_P = np.zeros((n+0, n+0), dtype=float)
    #for i in range(0, n+0):
        #for j in range(0, n+0):
    for i in range(1, n+1):
        for j in range(1, n+1):
            log_P[i, j] -= n * np.log(2)
            log_P[i, j] += log_binom(n+1, 2*i - j)
            log_P[i, j] += log_a[j] - log_a[i]
    return np.exp(log_P[1:, 1:])
    #return np.exp(log_P)

def stationary_distribution(n):
    eulerian = partial(_eulerian, n)
    #return [eulerian
    #TODO fixme


def resolvent_norm(A, z):
    # A: matrix
    # z: complex number
    n, m = A.shape
    assert_equal(n, m)
    I = np.eye(n, dtype=float)
    try:
        B = np.linalg.inv(z*I - A)
    except np.linalg.LinAlgError as e:
        return 0
    return np.linalg.norm(B, ord=np.inf)

def main():
    n = 52
    #print(np.exp(eulerian_logs(n)))
    a = np.array([_eulerian(n, i) for i in range(1, n+1)], dtype=float)
    #a = np.array([_eulerian(n, i) for i in range(0, n+0)], dtype=float)
    log_a = np.log(a)
    #print(p)
    #print(np.log(p.astype(float)))
    P = riffle_transition_matrix(n)

    print('row sums of probability matrix:')
    print(P.sum(axis=1))

    pi = np.exp(log_a - gammaln(n+1))

    print('sum of stationary probabilities:', pi.sum())
    A = P - pi

    print('creating the figure...')
    #figure_7_2(2 * A)
    figure_7_2(A)


def figure_7_2(A):
    """
    Figure 7.2.

    epsilon level sets:
    1e-1, 1e-1.5, 1e-2, ..., 1e-4
    dashed curve marks unit circle on the complex plane
    """
    levels = np.power(10, [1, 1.5, 2, 2.5, 3, 3.5, 4])
    f = np.vectorize(partial(resolvent_norm, A))

    low = -1.5
    high = 1.5
    u = np.linspace(low, high, 201)
    #u = np.linspace(low, high, 40)
    X, Y = np.meshgrid(u, u)
    z = u[np.newaxis, :] + 1j*u[:, np.newaxis]
    Z = f(z)
    print(Z)

    print('eigenvalues of decay matrix:')
    print(scipy.linalg.eigvals(A))

    #fig = plt.figure()
    fig, ax = plt.subplots()
    #plt.axes().
    ax.set_aspect('equal')

    plt.contour(
            X, Y,
            Z,
            #cmap=cm.hot,
            levels=levels,
            #norm=colors.LogNorm(),
            colors='k', # negative contours will be dashed by default
            )

    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)

    fig.gca().add_artist(circle1)


    """
    def animate(i):
        cont = plt.contour(
                X, Y,
                solutions[i].reshape((n, n)),
                cmap=cm.hot,
                levels=levels,
                norm=colors.LogNorm(),
                )
        return cont
    """

    """
    anim = animation.FuncAnimation(
            fig,
            animate,
            #init_func=init,
            init_func=None,
            frames=FRAMES,
            #frames=11,
            #interval=20,
            #blit=True,
            )
    """

    plt.show()


main()

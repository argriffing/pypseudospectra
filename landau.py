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

NOTE: the original radius was 1.5; this has been changed to 1.2.
If this new discretization introduces different artifacts than the old
discretization, then maybe revert to the old discretization.

"""
from __future__ import print_function, division

from functools import partial
import argparse

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


def my_multi_figure(base_filename, grid_count, t0, t1, t2, t3, A):
    """
    Make four plots in the same figure.

    This matplotlib code is inspired by the code for the layouts in
    http://matplotlib.org/1.4.1/examples/pylab_examples/subplots_demo.html

    """
    filename = base_filename + '.four.plots.svg'

    # These levels of the reciprocal of the pseudospectrum epsilon
    # are taken from the Tisseur and Higham publication.
    levels = np.power(10, np.linspace(1, 10, 19))

    # The outputs of the Schur decomposition are:
    # T : a triangular matrix, upper triangular by default.
    # Z : A unitary matrix
    T, Z_unitary = scipy.linalg.schur(A, output='complex')

    # Get the norms to plot.
    XYZ_triples = []
    for t in t0, t1, t2, t3:
        if t is None:
            f = np.vectorize(partial(resolvent_onenorm, T, Z_unitary))
        else:
            f = np.vectorize(partial(resolvent_onenormest, t, T, Z_unitary))
        low = -1.2
        high = 1.2
        u = np.linspace(low, high, grid_count)
        X, Y = np.meshgrid(u, u)
        z = u[np.newaxis, :] + 1j*u[:, np.newaxis]
        Z = f(z)
        XYZ_triples.append((X, Y, Z))

    # row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    ## remove most of the spacing between the subplots
    f.subplots_adjust(hspace=0.001, wspace=0.001)

    for ax in ax1, ax2, ax3, ax4:

        # Preserve the aspect ratio of the complex plane.
        # For example, the unit circle in the complex plane should
        # look like a circle and not like an ellipse.
        ax.set_aspect('equal')

        # Do not draw the axes.
        ax.axis('off')

    # plot the upper left contour plot
    X, Y, Z = XYZ_triples[0]
    ax1.contour(X, Y, Z, levels=levels, colors='k')
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    # gca means get current axis, creating one if necessary
    #fig.gca().add_artist(circle1)
    ax1.add_artist(circle1)

    # plot the upper right contour plot
    X, Y, Z = XYZ_triples[1]
    ax2.contour(X, Y, Z, levels=levels, colors='k')
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    ax2.add_artist(circle1)

    # plot the lower left contour plot
    X, Y, Z = XYZ_triples[2]
    ax3.contour(X, Y, Z, levels=levels, colors='k')
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    ax3.add_artist(circle1)

    # plot the lower right contour plot
    X, Y, Z = XYZ_triples[3]
    ax4.contour(X, Y, Z, levels=levels, colors='k')
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    ax4.add_artist(circle1)

    #plt.show()
    plt.savefig(filename)


def figure(base_filename, grid_count, t, A):

    if t is None:
        filename = base_filename + '.svg'
    else:
        filename = base_filename + '.' + str(t) + '.svg'

    # These levels of the reciprocal of the pseudospectrum epsilon
    # are taken from the Tisseur and Higham publication.
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

    low = -1.2
    high = 1.2
    u = np.linspace(low, high, grid_count)
    X, Y = np.meshgrid(u, u)
    z = u[np.newaxis, :] + 1j*u[:, np.newaxis]
    Z = f(z)

    # Add contour lines at predefined levels.
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.contour(X, Y, Z, levels=levels, colors='k')

    # Add dashed unit circle.
    circle1 = plt.Circle((0, 0), 1, color='k', linestyle='dashed', fill=False)
    fig.gca().add_artist(circle1)

    #plt.show()
    plt.savefig(filename)



def main(args):

    # Create the landau matrix.
    A = make_landau_matrix()

    print('first few eigenvalues of the matrix:')
    w = scipy.linalg.eigvals(A)
    print(w[:10])
    print('last few eigenvalues of the matrix:')
    print(w[-10:])

    print('trying to plot the figure with four contour maps...')
    my_multi_figure(args.base_filename, args.grid_count, 1, 4, 8, 16, A)

    return

    # Define the values of t to use in the loop.
    # In this context, t is the block size of the block approximation.
    # When t is None, the pseudospectrum is computed exactly.
    t_values = [2]
    #t_values = [None, 1, 2, 4, 8, 16]

    # Create an svg file for each requested block size,
    # using the requested output filename base and using
    # the requested number of lines in the real and imaginary
    # axis for the complex plane in the pseudospectrum discretization.
    print('creating the figures for t in', t_values, '...')
    for t in t_values:
        print('t =', t, '...')
        figure(args.base_filename, args.grid_count, t, A)


if __name__ == '__main__':

    # Initialize default values of options.
    default_grid_count = 201
    default_base_filename = 'landau'

    # Show the numpy and scipy versions.
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    # Define the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-count',
            default=default_grid_count,
            help=(
                'The pseudospectrum discretization has this many lines '
                'in each direction (Default: %d).' % default_grid_count))
    parser.add_argument('--base-filename',
            default=default_base_filename,
            help=(
                'The base filename for the output .svg image files '
                '(Default: %s).' % default_base_filename))

    # Parse the command line arguments.
    args = parser.parse_args()

    # Report the values of the command line arguments,
    # and call the main procedure.
    print('base filename:', args.base_filename)
    print('grid count:', args.grid_count)
    main(args)

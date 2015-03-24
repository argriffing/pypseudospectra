"""
An attempt to reproduce the landau pseudospectra using the pyoperators package.

"""
from __future__ import print_function, division

from functools import partial
import argparse

import numpy as np
from numpy.testing import assert_equal
import scipy.linalg

import matplotlib.pyplot as plt
from matplotlib import colors, cm

from create_landau import make_landau_matrix

from pyoperators import pcg, Operator, DenseOperator, I

from _onenormest import operator_onenormest


_default_grid_count = 21
_default_base_filename = 'landau.pyoperators'


def resolvent(A, z):
    return (A - z*I).I

def resolvent_norm(A, t, z):
    return operator_onenormest(resolvent(A, z), t=t)


def my_multi_figure(base_filename, grid_count, t0, t1, t2, t3):
    """
    Make four plots in the same figure.

    This matplotlib code is inspired by the code for the layouts in
    http://matplotlib.org/1.4.1/examples/pylab_examples/subplots_demo.html

    """
    filename = base_filename + '.four.plots.svg'


    print('creating landau ndarray...')
    A_ndarray = make_landau_matrix()

    print('converting the landau ndarray to a pyoperators operator...')
    A = DenseOperator(
            A_ndarray,
            #shape=A_ndarray.shape,
            shapein=A_ndarray.shape[0],
            shapeout=A_ndarray.shape[1],
            )

    print('dense operator shape:', A.shape)

    # These levels of the reciprocal of the pseudospectrum epsilon
    # are taken from the Tisseur and Higham publication.
    levels = np.power(10, np.linspace(1, 10, 19))

    # Get the norms to plot.
    XYZ_triples = []
    for t in t0, t1, t2, t3:
        f = np.vectorize(partial(resolvent_norm, A, t))
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



def main(args):
    my_multi_figure(args.base_filename, args.grid_count, 1, 4, 8, 16)



if __name__ == '__main__':

    # Show the numpy and scipy versions.
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    # Define the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-count',
            default=_default_grid_count,
            help=(
                'The pseudospectrum discretization has this many lines '
                'in each direction (Default: %d).' % _default_grid_count))
    parser.add_argument('--base-filename',
            default=_default_base_filename,
            help=(
                'The base filename for the output .svg image files '
                '(Default: %s).' % _default_base_filename))

    # Parse the command line arguments.
    args = parser.parse_args()

    # Report the values of the command line arguments,
    # and call the main procedure.
    print('base filename:', args.base_filename)
    print('grid count:', args.grid_count)
    main(args)

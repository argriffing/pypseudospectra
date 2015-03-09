"""

"""
from __future__ import print_function, division

import numpy as np
import scipy.linalg
from numpy.testing import assert_equal

from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg._onenormest import _onenormest_core


def resolvent_onenorm(T, Z, eps_recip):
    n = T.shape[0]
    ident = np.eye(n)
    op = get_resolvent_operator(T, Z, eps_recip)
    try:
        #B = resolvent(A, z)
        B = np.asarray(op.matmat(ident))
    except np.linalg.LinAlgError as e:
        return 0
    return np.linalg.norm(B, ord=1)


def resolvent_onenormest(t, T, Z, eps_recip):
    op = get_resolvent_operator(T, Z, eps_recip)
    #return scipy.sparse.linalg.onenormest(op, t=t)
    stuff = _onenormest_core(op, op.H, t=t, itmax=5)
    #stuff = _onenormest_core(op, op.H, t=t, itmax=10)
    est, v, w, nmults, nresamples = stuff
    #print(nmults, nresamples)
    #assert_equal(v.dtype, np.dtype(float))
    #assert_equal(w.dtype, np.dtype(complex))
    return est


def get_resolvent_operator(T, Z, eps_recip):
    # M is triangular
    # Z is unitary
    n = T.shape[0]
    T = np.diag(np.ones(n) * eps_recip) - T
    return _ResolventOperator(T, Z)


class _ResolventOperator(LinearOperator):

    def __init__(self, M, Z, lower=False):
        # M is triangular
        # Z is unitary
        self.M = M
        self.Z = Z
        self.lower = lower
        self.shape = M.shape
        self.dtype = np.dtype(complex)

    def _matmat(self, B):
        M = self.M
        Z = self.Z
        ZH = self.Z.conj().T
        #assert_equal(self.M.shape, self.Z.shape)
        #assert_equal(self.M.shape, self.ZH.shape)
        print('shapes in resolvent matrix multiplication:')
        print('M:', M.shape)
        print('Z:', Z.shape)
        print('ZH:', ZH.shape)
        print('B:', B.shape)
        print()
        C = scipy.linalg.solve_triangular(M, ZH.dot(B), lower=self.lower)
        return Z.dot(C)

    def _transpose(self):
        return _ResolventOperator(self.M.T, self.Z,
                lower=(not self.lower))

    def _adjoint(self):
        return _ResolventOperator(self.M.conj().T, self.Z,
                lower=(not self.lower))

#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
#    This is a custom 2D polynomial fitter. It performs basic
# polynomial fitting. It differs from other versions (such as the
# PolyFit2D in centroid_tools) in that the lower and higher degree
# are specified. This allows fitting of higher-order terms without
# low-order terms.
#
# Rob Siverd
# Created:       2018-01-21
# Last modified: 2019-03-26
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Modules:
import os
import sys
import time
import numpy as np

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## Polynomial fitting class:
class CustomPolyFit2D(object):

    def __init__(self):
        self._deg1  = 0
        self._deg2  = 0
        self._ij    = None
        self._model = None
        return

    def set_degree(self, deg1, deg2):
        self._deg1 = deg1
        self._deg2 = deg2
        self._ij   = self._powlist(deg1, deg2)
        return

    def get_model(self):
        return self._model

    def _have_model(self):
        return isinstance(self._model, np.array)

    # make exponents list:
    @staticmethod
    def _powlist(deg1, deg2):
        xpow, ypow = np.meshgrid(range(deg2+1), range(deg2+1))
        use = (deg1 <= xpow + ypow) & (xpow + ypow <= deg2)
        return list(zip(xpow[use], ypow[use]))

    # Surface fitter (my enhancement):
    #def fit(self, x, y, z, degree):
    def fit(self, x, y, z):
        ij = self._ij
        G = np.zeros((x.size, len(ij)), dtype=np.float)
        for k, (i,j) in enumerate(ij):
            G[:,k] = x.flatten()**i * y.flatten()**j
        m, _, _, _ = np.linalg.lstsq(G, z.flatten())
        self._model = m
        #return m
        return

    # Surface evaluator:
    #def eval(self, x, y, m, degree):
    def eval(self, x, y):
        funcname = sys._getframe().f_code.co_name
        if not self._have_model():
            sys.stderr.write("No model set!  Must fit first ...\n")
        ij = self._ij
        m  = self._model
        if (len(ij) != m.size):
            sys.stderr.write("%s: model does not match degree!\n" % funcname)
            return
        z = np.zeros_like(x, dtype=np.float)
        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j
        return z


######################################################################
# CHANGELOG (custom_polyfit.py):
#---------------------------------------------------------------------
#
#  2019-03-26:
#     -- Increased __version__ to 0.1.0.
#     -- Include CustomPolyFit2D based on PolyFit2D
#     -- First created custom_polyfit.py.
#


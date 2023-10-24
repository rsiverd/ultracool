#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Generalized tangent projection helper module. This version provides a
# class with getters and setters to simplify its use.
#
# Rob Siverd
# Created:       2023-07-19
# Last modified: 2023-07-19
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Modules:
import gc
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import fov_rotation
rfov = fov_rotation.RotateFOV()

##--------------------------------------------------------------------------##

## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
## Matrix printer:
def mprint(matrix):
    for row in matrix:
        sys.stderr.write("  %s\n" % str(row))
    return

## Reflection matrices:
xref_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
yref_mat = np.array([[-1.0, 0.0], [0.0, 1.0]])
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array([[1.0, 0.0], [0.0, 1.0]])

##--------------------------------------------------------------------------##


##--------------------------------------------------------------------------##
##------------------       Tangent Projection Module        ----------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## New-style string formatting (more at https://pyformat.info/):



######################################################################
# CHANGELOG (tangent_projection.py):
#---------------------------------------------------------------------
#
#  2023-07-19:
#     -- Increased __version__ to 0.0.1.
#     -- First created tangent_projection.py.
#

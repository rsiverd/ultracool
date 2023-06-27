#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# My own robust statistics implementations, all in one place!
#
# Rob Siverd
# Created:       2018-10-17
# Last modified: 2019-06-20
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.2"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt

##--------------------------------------------------------------------------##
_MAD_scalefactor = 1.482602218
_IQR_scalefactor = 0.5 * _MAD_scalefactor

##--------------------------------------------------------------------------##

## Outlier identification with specified midpoint (MAD-based scale):
def calc_specific_MAD(a, med_val, scalefactor=_MAD_scalefactor, axis=None):
    """Return median absolute deviation (scaled to normal) relative to the
    specified reference point."""
    return (1.482602218 * np.median(np.abs(a - med_val), axis=axis))

## Robust location/scale estimate using median/MAD:
def calc_ls_med_MAD(a, scalefactor=_MAD_scalefactor, axis=None):
    """Return median and median absolute deviation of *a* (scaled to normal)."""
    med_val = np.median(a, axis=axis)
    sig_hat = (scalefactor * np.median(np.abs(a - med_val), axis=axis))
    return (med_val, sig_hat)

## Robust location/scale estimate using median/IQR:
def calc_ls_med_IQR(a, scalefactor=_IQR_scalefactor, axis=None):
    """Return median and inter-quartile range of *a* (scaled to normal)."""
    pctiles = np.percentile(a, [25, 50, 75], axis=axis)
    med_val = pctiles[1]
    sig_hat = (0.741301109 * (pctiles[2] - pctiles[0]))
    return (med_val, sig_hat)

## Select inliners given specified sigma threshold:
def pick_inliers(data, sig_thresh):
    med, sig = calc_ls_med_IQR(data)
    return ((np.abs(data - med) / sig) <= sig_thresh)

## Robust-clipped average:
def clipped_average(data, sig_thresh):
    return np.average(data[pick_inliers(data, sig_thresh)])

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Iterative clipped mean/stddev:
def _icms_helper(data, sig_thresh):
    tavg, tstd = data.mean(), data.std()
    outly_mask = (np.abs((data - tavg) / tstd) > sig_thresh)
    return tavg, tstd, outly_mask

def iterclip_mean_stddev(data, sig_thresh, maxiter=20,
        vlevel=0, stream=sys.stdout):
    this_func = sys._getframe().f_code.co_name
    wrk_data = data.copy()

    # Initial stats/outlier estimate:
    cavg, cstddev, outly_mask = _icms_helper(wrk_data, sig_thresh)
    n_outliers = outly_mask.sum()
    niter = 0
    while (n_outliers > 0):
        niter += 1
        if (vlevel >= 2):
            stream.write("On pass %d have %d of %d outliers.\n"
                    % (niter, n_outliers, wrk_data.size))
        wrk_data = wrk_data[~outly_mask]    # drop outliers
        cavg, cstddev, outly_mask = _icms_helper(wrk_data, sig_thresh)
        n_outliers = outly_mask.sum()
        if (niter >= maxiter):
            if (vlevel >= 0):
                stream.write("%s: max iterations exceeded!\n" % this_func)
            break
    if (vlevel >= 0):
        stream.write("%s: converged in %d iterations.\n" % (this_func, niter))
    return cavg, cstddev


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Unscaled MAD:
def unscaled_MAD(a, mid, axis=None):
    """Return unscaled median absolute deviation of *a* array values."""
    return np.median(np.abs(a - mid), axis=axis)


## Biweight midvariance:
def biweight_midvariance(data, mid=None, tuning=9.0, subsamp=True, axis=None):
    location = mid if (mid != None) else np.median(data, axis=axis)
    uabs_dev = np.median(np.abs(data - location), axis=axis)
    #uvals    = (data - location) / (tuning * uabs_dev)
    #safe     = (np.abs(uvals) < 1)     # these are used for the estimate
    #ngood    = np.sum(safe)
    usquared = ((data - location) / (tuning * uabs_dev))**2.0
    safe     = (np.abs(usquared) < 1)   # these are used for the estimate
    n_used   = float(np.sum(safe)) if subsamp else float(data.size)
    sys.stderr.write("Allowing %d of %d data points (%.2f%%).\n"
            % (int(n_used), data.size, 100. * n_used / float(data.size)))
    numer = np.sum((data[safe] - location)**2 * (1.0 - usquared[safe])**4)
    denom = np.sum((1.0 - usquared[safe]) * (1.0 - 5.*usquared[safe]))**2
    return (n_used * numer / denom)

## Standard deviation approximately:
def bivar_dev(data, mid=None, tuning=9.0, subsamp=True, axis=None):
    return np.sqrt(biweight_midvariance(data, mid, tuning, subsamp, axis))



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##





######################################################################
# CHANGELOG (robust_stats.py):
#---------------------------------------------------------------------
#
#  2019-06-20:
#     -- Increased __version__ to 0.1.2.
#     -- Added iterative clipped mean/stddev calculator.
#
#  2018-11-27:
#     -- Increased __version__ to 0.1.1.
#     -- Added simple clipped_average() routine.
#
#  2018-10-17:
#     -- Increased __version__ to 0.1.0.
#     -- Added NEW methods unscaled_MAD, biweight_midvariance, and bivar_dev.
#     -- Added calc_ls_med_MAD, calc_ls_med_IQR, and pick_inliers (template).
#     -- First created robust_stats.py.
#

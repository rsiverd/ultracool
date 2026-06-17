#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module contains a new generation of astrometric parameter models
# and evaluators. The models and evaluators in this file are complete with
# respect to parameters (some of which may be fixed) to avoid a proliferation
# of fit and evaluation codes. These routines may prove suitable for use with
# more generalized approaches such as that of lmfit.
#
# Rob Siverd
# Created:       2026-06-09
# Last modified: 2026-06-09
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
#import shutil
#import signal
#import glob
#import math
#import ast
#import io
#import gc
import os
import sys
import time
#import pprint
#import pickle
#import vaex
#import calendar
#import ephem
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
#import scipy.stats as scst
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import operator
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Helpers provides the inverse tangent routine:
import helpers
reload(helpers)

## Tangent projection routines:
import tangent_proj as tp

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Home-brew robust statistics:
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

## Various from astropy:
#try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
#except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
#    sys.exit(1)




##--------------------------------------------------------------------------##
##------------------      Wrapped Tangent Projection        ----------------##
##--------------------------------------------------------------------------##

## Tangent projection evaluator:
def eval_cdmcrv(cdm_crv, xrel, yrel):
    this_cdmat = cdm_crv[:4]
    cv1, cv2   = cdm_crv[4:]
    return tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)

## Inverse tangent projection evaluator:
def inverse_tan_cdmcrv(cdm_crv, ra_deg, de_deg):
    this_cdmat = cdm_crv[:4].reshape(2, 2)
    cv1, cv2   = cdm_crv[4:]
    return tp.sky2xy_cd(this_cdmat, ra_deg, de_deg, cv1, cv2)


##--------------------------------------------------------------------------##
##------------------         Distortion Routines            ----------------##
##--------------------------------------------------------------------------##

## Pretty good guess for the distortion:
guess_distmod = np.array([ 1.757279e-01,  1.175609e-03,  1.047979e-06,
                           5.059445e-11,  4.450758e-13, -1.039322e-16])

## NOTE: the following are copied from slv_par_tools.py. They implement the
## "old" additive way of handling distortion. These can be replaced with
## the multiplicative variants as needed.
def poly_eval5(r, c0, c1, c2, c3, c4, c5):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r*c5))))

def poly_eval(r, model):
    return poly_eval5(r, *model)

def calc_rdist_corrections(xrel, yrel, model):
    rdist = np.hypot(xrel, yrel)     # distance from CRPIX
    rcorr = poly_eval(rdist, model)  # total correction magnitude
    theta = np.arctan2(yrel, xrel)
    xcorr = rcorr * np.cos(theta)
    ycorr = rcorr * np.sin(theta)
    return xcorr, ycorr

calc_rdist_corr_sky2det = calc_rdist_corrections

## Manually reverse correction direction going det -> sky:
def calc_rdist_corr_det2sky(xrel, yrel, model):
    xcorr, ycorr = calc_rdist_corrections(xrel, yrel, model)
    return -xcorr, -ycorr

##--------------------------------------------------------------------------##
##------------------         Affine Transformations         ----------------##
##--------------------------------------------------------------------------##

def mkaffine(par):
    return np.array([[par[0], par[1], par[4]],
                     [par[2], par[3], par[5]],
                     [   0.0,    0.0,    1.0]])

##--------------------------------------------------------------------------##
##------------------         Residual Calculation           ----------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Single-sensor residual calculator (lowest level). This routine implements
## an astrometric model consisting of:
## * radial focal plane distortion
## * standard 8-parameter WCS with tangent projection
##
## Inputs to this routine include:
## * params: 1-D numpy array of parameters under test/calculation
##      --> params[0:4] => cd11, cd12, cd21, cd22
##      --> params[4:8] => crpix1, crpix2, crval1, crval2
## * xdet, ydet: detector positions of matched sources
## * app_ra_deg: apparent RA (Gaia) in degrees (rel aberr/refract removed)
## * app_de_deg: apparent DE (Gaia) in degrees (rel aberr/refract removed)
##
## Outputs are x_residuals, y_residuals in pixel units.
def xyr_calculator(params, xdet, ydet, app_dra, app_dde):
    # parse parameters
    this_cdmat = params[0:4].reshape(2, 2)
    crp1, crp2 = params[4:6]
    crv1, crv2 = params[6:8]
    dist_model = params[8:]

    # compute undistorted xrel/yrel from sky coordinates:
    undist_xrel, undist_yrel = \
            tp.sky2xy_cd(this_cdmat, app_dra, app_dde, crv1, crv2)

    # calculate distortion corrections and distorted xrel/yrel:
    xnudge, ynudge = \
            calc_rdist_corr_sky2det(undist_xrel, undist_yrel, dist_model)
    calc_xdet = undist_xrel + xnudge + crp1
    calc_ydet = undist_yrel + ynudge + crp2

    # return residuals:
    return np.vstack((xdet - calc_xdet, ydet - calc_ydet))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Breakout version:
def xyr_calc_breakout(cdmat, crp1, crp2, crv1, crv2, dist_model,
                                        xdet, ydet, app_dra, app_dde):
    # compute undistorted xrel/yrel from sky coordinates:
    undist_xrel, undist_yrel = \
            tp.sky2xy_cd(cdmat, app_dra, app_dde, crv1, crv2)

    # calculate distortion corrections and distorted xrel/yrel:
    xnudge, ynudge = \
            calc_rdist_corr_sky2det(undist_xrel, undist_yrel, dist_model)
    calc_xdet = undist_xrel + xnudge + crp1
    calc_ydet = undist_yrel + ynudge + crp2

    # return residuals:
    #return (xdet - calc_xdet, ydet - calc_ydet)
    return np.vstack((xdet - calc_xdet, ydet - calc_ydet))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Single-image residuals driver that expects:
## * 8x WCS parameters, 18x txform params, and Dpar distortion params
## * 4-sensor input data (ordered NE, NW, SE, SW)
def xyr_driver_single_txform(params, data_arrays, 
                             xcol='x', ycol='y', 
                             racol='gra', decol='gde',
                             scale_nstars=False):
    atxpar, wcspar, dstpar = np.split(params, (18, 26))
    #txpars = params[:18]
    #remain = params[18:]
    #wcspar = remain[:8]
    #remain = remain[8:]
    nw_txpar, se_txpar, sw_txpar = atxpar.reshape(3, -1)
    cdmat = wcspar[:4].reshape(2, 2)
    cp1, cp2, cv1, cv2 = wcspar[4:]

    # Star-count scaling:
    if scale_nstars:
        det_nstars = [len(x) for x in solo_gstars.values()]
        avg_nstars = sum(det_nstars) / 4.0
        #nstar_scale_factor = [avg_nstars/x for x in ndet_nstars]
        nsf_ne, nsf_nw, nsf_se, nsf_sw = [avg_nstars/x for x in ndet_nstars]
    else:
        #nstar_scale_factor = [1.0, 1.0, 1.0, 1.0]
        nsf_ne, nsf_nw, nsf_se, nsf_sw = 1.0, 1.0, 1.0, 1.0

    # Data breakout:
    ne_data, nw_data, se_data, sw_data = data_arrays

    # Empty residuals list:
    res_vecs = []

    # Append NE residuals (no transform):
    #xr, yr = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar, 
    xy_res = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar, 
                          ne_data[xcol].values, ne_data[ycol].values,
                          ne_data[racol].values, ne_data[decol].values)
    #res_vecs += [(xr * nsf_ne, yr * nsf_ne)]
    res_vecs += [xy_res * nsf_ne]

    # Append NW:
    xform = mkaffine(nw_txpar)
    tx, ty = nw_data[xcol].values, nw_data[ycol].values
    x_ne, y_ne, _ = xform @ np.vstack((tx, ty, np.ones_like(tx)))
    # transform errors here ...
    #xr, yr = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar,
    xy_res = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar,
              x_ne, y_ne, nw_data[racol].values, nw_data[decol].values)
    res_vecs += [xy_res * nsf_nw]
    #res_vecs += [(xr * nsf_nw, yr * nsf_nw)]

    # Append SE:
    xform = mkaffine(se_txpar)
    tx, ty = se_data[xcol].values, se_data[ycol].values
    x_ne, y_ne, _ = xform @ np.vstack((tx, ty, np.ones_like(tx)))
    # transform errors here ...
    #xr, yr = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar, 
    xy_res = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar, 
              x_ne, y_ne, se_data[racol].values, se_data[decol].values)
    res_vecs += [xy_res * nsf_se]
    #res_vecs += [(xr * nsf_se, yr * nsf_se)]

    # Append SW:
    xform = mkaffine(sw_txpar)
    tx, ty = sw_data[xcol].values, sw_data[ycol].values
    x_ne, y_ne, _ = xform @ np.vstack((tx, ty, np.ones_like(tx)))
    # transform errors here ...
    #xr, yr = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar, 
    xy_res = xyr_calc_breakout(cdmat, cp1, cp2, cv1, cv2, dstpar, 
              x_ne, y_ne, sw_data[racol].values, sw_data[decol].values)
    res_vecs += [xy_res * nsf_sw]
    #res_vecs += [(xr * nsf_sw, yr * nsf_sw)]

    # Return the whole set:
    return res_vecs

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Wrapper to compute the sum of squared residuals for minimization:
def ssr_scalarize(params, routine):
    res_vecs = routine(params)
    return sum([np.sum(x**2) for x in res_vecs])

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (uniform_model.py):
#---------------------------------------------------------------------
#
#  2026-06-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created uniform_model.py.
#

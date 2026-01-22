#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Routines shared by multiple scripts in this investigation.
#
# Rob Siverd
# Created:       2025-02-03
# Last modified: 2025-02-03
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Modules:
#import argparse
import os
import sys
import time
import math
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Tangent projection:
import tangent_proj as tp

### Experimental tangent projection with distortion fix:
##import dist_tangent_proj as dtp
#import dist_tangent_proj
#reload(dist_tangent_proj)
#dtp  = dist_tangent_proj
#

try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##



## Useful definitions:
xref_mat  = np.array((( 1.0, 0.0), (0.0, -1.0)))
yref_mat  = np.array(((-1.0, 0.0), (0.0,  1.0)))
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array((( 1.0, 0.0), (0.0,  1.0)))

#_use_pscale_arcsec = 0.3070
_use_pscale_arcsec = 0.3062
_use_pscale_arcsec = 0.3060
#_use_pscale_arcsec = 0.3040
#_use_pscale_arcsec = 0.3000
_use_pscale_deg = _use_pscale_arcsec / 3600.0

## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array((c, -s, s, c)).reshape(2, 2)

## Matrix printer:
def mprint(matrix):
    for row in matrix:
        sys.stderr.write("  %s\n" % str(row))
    return

## Make a CD matrix given pixel scale and PA:
def make_cdmat(pa_rad, pxscale_deg):
    #rmat = rotation_matrix(pa_rad)
    return pxscale_deg * np.dot(xflip_mat, rotation_matrix(pa_rad))

def make_cdmat_wrap(pa_deg, pxscale):
    return make_cdmat(math.radians(pa_deg), pxscale / 3.6e3)

def make_test_cdmat(pa_deg):
    return make_cdmat(math.radians(pa_deg), _use_pscale_deg)

_ne_nw_rot_deg = -0.2407517
_ne_se_rot_deg = -0.1577913
_ne_sw_rot_deg = +0.0085392
#_ne_sw_rot_deg = -0.085392
#_ne_sw_rot_deg -= 0.25
_ne_nw_rot = math.radians(_ne_nw_rot_deg)
_ne_se_rot = math.radians(_ne_se_rot_deg)
_ne_sw_rot = math.radians(_ne_sw_rot_deg)

def make_four_cdmats(ne_pa_deg):
    ne_cdmat = make_test_cdmat(ne_pa_deg)
    nw_cdmat = np.dot(rotation_matrix(_ne_nw_rot), ne_cdmat)
    se_cdmat = np.dot(rotation_matrix(_ne_se_rot), ne_cdmat)
    sw_cdmat = np.dot(rotation_matrix(_ne_sw_rot), ne_cdmat)
    return {'NE':ne_cdmat.ravel(), 'NW':nw_cdmat.ravel(),
            'SE':se_cdmat.ravel(), 'SW':sw_cdmat.ravel()}


##--------------------------------------------------------------------------##
##------------------         Vector Math Helpers            ----------------##
##--------------------------------------------------------------------------##

def _vec_length(vector):
    return np.sqrt(np.sum(vector**2))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Loader for skypix output:
def load_skypix_output(filename):
    with open(filename, 'r') as ff:
        content = [x.strip().split() for x in ff.readlines()]
    xx, yy, _, ra, de = zip(*content)
    return np.float_(xx), np.float_(yy), np.float_(ra), np.float_(de)

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

## Inverse tangent projection evaluator WITH DISTORTION CORRECTION:
def inverse_tan_cdmcrv_distmod(cdm_crv, distmod, ra_deg, de_deg):
    this_cdmat = cdm_crv[:4].reshape(2, 2)
    cv1, cv2   = cdm_crv[4:]
    return dtp.sky2xy_cd_distfix(this_cdmat, ra_deg, de_deg, cv1, cv2, distmod)

## Get an astropy Time object from WIRCAM header:
def wircam_timestamp_from_header(header):
    obs_time = astt.Time(header['MJD-OBS'], scale='utc', format='mjd') \
            + 0.5 * astt.TimeDelta(header['EXPTIME'], format='sec')
    return obs_time

## Analyze CD matrix
def analyze(data):
    cd_matrix = data[:4].reshape(2,2)
    cd_pscales = np.sqrt(np.sum(cd_matrix**2, axis=1))
    #cd_pscales = np.sqrt(np.sum(cd_matrix**2, axis=0))
    norm_cdmat = cd_matrix / cd_pscales
    cd_ang_rad = np.arccos(norm_cdmat[0, 0])
    cd_ang_deg = np.degrees(cd_ang_rad)
    return cd_ang_deg, cd_pscales[0], cd_pscales[1]


def test_eval_pa_crv(pa_deg, cv1, cv2, xrel, yrel):
    this_cdmat = make_test_cdmat(pa_deg)
    tra, tde = tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)
    return (tra % 360.0, tde)


## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##

sensor_order = ['NE', 'NW', 'SE', 'SW']
sensor_qqmap = {kk:vv for kk,vv in enumerate(sensor_order)}

_sord = sensor_order
_smap = sensor_qqmap

## ----------------------------------------------------------------------- ##
## Parameter-parser for solving and diagnostics:
def sift_params(params):
    parsleft = params.copy()

    # Peel CDxx, CRPIXx from the front:
    cdmcrpix = parsleft[:24].reshape(-1, 6)
    parsleft = parsleft[24:]
    #cdmat_4s = cdmcrpix[:, :4]
    test_cdm_calc = {qq:vv for qq,vv in zip(sensor_order, cdmcrpix[:, :4])}
    test_sensor_crpix = {qq:vv for qq,vv in zip(sensor_order, cdmcrpix[:, 4:])}

    # Peel CRVAL1, CRVAL2 from the front next:
    cv1, cv2 = parsleft[:2]
    #crvals12 = parsleft[:2]
    #parsleft = parsleft[2:]

    # Remaining parameters are the distortion model:
    rdist_pars = parsleft[2:]

    return {'cdmat':test_cdm_calc, 'crpix':test_sensor_crpix,
            'crval':[cv1, cv2], 'rpars':rdist_pars}

## ----------------------------------------------------------------------- ##
## Calculate sensor-to-sensor gutters implied by various CRPIX values:
def describe_gutters(sifted):
    sensor_crpix = sifted['crpix']
    ne_crp1, ne_crp2 = sensor_crpix['NE']
    nw_crp1, nw_crp2 = sensor_crpix['NW']
    se_crp1, se_crp2 = sensor_crpix['SE']
    sw_crp1, sw_crp2 = sensor_crpix['SW']
    ne_nw = ne_crp1 - nw_crp1 - 2048.
    se_sw = se_crp1 - sw_crp1 - 2048.
    ne_se = se_crp2 - ne_crp2 - 2048.
    nw_sw = sw_crp2 - nw_crp2 - 2048.
    sys.stderr.write("NE <--> NW (upper): %.2f\n" % ne_nw)
    sys.stderr.write("SE <--> SW (lower): %.2f\n" % se_sw)
    sys.stderr.write("NE <--> SE ( left): %.2f\n" % ne_se)
    sys.stderr.write("NW <--> SW (right): %.2f\n" % nw_sw)
    return

def describe_rotations(sifted):
    sensor_cdmat = sifted['cdmat']
    #_data = np.array([helpers.analyze(sensor_cdmat[x]) for x in _sord])
    _data = np.array([analyze(sensor_cdmat[x]) for x in _sord])
    #sensor_padeg = _data[:, 0]
    sensor_padeg = {qq:vv for qq,vv in zip(_sord, _data[:, :1])}
    sensor_scale = {qq:vv for qq,vv in zip(_sord, _data[:, 1:]*3600)}
    #sensor_padeg, sensor_xscale, sensor_yscale = {}, {}, {}
    for pair in sensor_padeg.items():
        sys.stderr.write("Sensor %s PA: %.3f\n" % pair)
    for qq,psc in sensor_scale.items():
        sys.stderr.write("Sensor %s pixscales: %s\n" % (qq, str(psc)))
    # Relative PAs:
    for ii,jj in itt.combinations(range(len(_sord)), 2):
        qi, qj = _smap[ii], _smap[jj]
        pi, pj = sensor_padeg[qi], sensor_padeg[qj]
        sys.stderr.write("Between %s and %s: %.4f\n" % (qi, qj, pi-pj))
    return

def describe_answer(sifted):
    describe_gutters(sifted)
    return




######################################################################
# CHANGELOG (helpers.py):
#---------------------------------------------------------------------
#
#  2025-02-03:
#     -- Increased __version__ to 0.0.1.
#     -- First created helpers.py.
#

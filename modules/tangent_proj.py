#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Generalized tangent projection helper module.
#
# Rob Siverd
# Created:       2023-06-02
# Last modified: 2023-10-24
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Logging setup:
import logging
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

## Current version:
__version__ = "0.1.1"

## Modules:
#import argparse
#import shutil
import os
import sys
import time
import math
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
#import matplotlib.gridspec as gridspec
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

import fov_rotation
rfov = fov_rotation.RotateFOV()

## Rotation matrix builder:
#def slow_rotation_matrix(theta):
#    """Generate 2x2 rotation matrix for specified input angle (radians)."""
#    return np.array([[np.cos(theta), -np.sin(theta)],
#                        [np.sin(theta), np.cos(theta)]])

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

## Reflection matrices:
xref_mat  = np.array((( 1.0, 0.0), (0.0, -1.0)))
yref_mat  = np.array(((-1.0, 0.0), (0.0,  1.0)))
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array((( 1.0, 0.0), (0.0,  1.0)))

## Radian-to-degree converter:
_radeg = 180.0 / np.pi

## Tangent projection:
def _tanproj(prj_xx, prj_yy):
    prj_rr = np.hypot(prj_xx, prj_yy)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    #prj_rr = np.sqrt(prj_xx**2 + prj_yy**2)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    useful = (prj_rr > 0.0)
    prj_theta = np.ones_like(prj_xx) * np.pi * 0.5
    prj_theta[useful] = np.arctan(np.degrees(1.0 / prj_rr[useful]))
    #prj_theta[useful] = np.arctan(_radeg / prj_rr[useful])
    #prj_phi = np.arctan2(prj_xx, prj_yy)
    prj_phi = np.arctan2(prj_xx, -prj_yy)
    #return prj_phi, prj_theta
    return np.degrees(prj_phi), np.degrees(prj_theta)

## Inverse tangent projection:
def _inv_tanproj(prj_phi_deg, prj_theta_deg):
    prj_phi_rad   = np.radians(prj_phi_deg)
    prj_theta_rad = np.radians(prj_theta_deg)
    sin_phi       = np.sin(prj_phi_rad)
    cos_phi       = np.cos(prj_phi_rad)
    sin_theta     = np.sin(prj_theta_rad)
    cos_theta     = np.cos(prj_theta_rad)
    prj_xx        = np.zeros_like(prj_phi_rad) #+ sin_phi
    prj_yy        = np.zeros_like(prj_phi_rad) #+ cos_phi
    prj_rr        = np.zeros_like(prj_phi_rad)
    which         = (sin_theta != 0.0)      # calculate for these
    prj_rr[which] = np.degrees(cos_theta / sin_theta)
    prj_xx[which] =        prj_rr * sin_phi
    prj_yy[which] = -1.0 * prj_rr * cos_phi
    return prj_xx, prj_yy


## Low-level WCS tangent processor:
def _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=False):
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    if debug:
        sys.stderr.write("%.3f < prj_xx < %.3f\n"
                % (prj_xx.min(), prj_xx.max()))
        sys.stderr.write("%.3f < prj_yy < %.3f\n"
                % (prj_yy.min(), prj_yy.max()))

    # Perform tangent projection:
    prj_phi, prj_theta = _tanproj(prj_xx, prj_yy)
    if debug:
        sys.stderr.write("%.3f < prj_theta < %.3f\n"
                % (prj_theta.min(), prj_theta.max()))
        sys.stderr.write("%.3f < prj_phi   < %.3f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Change variable names to avoid confusion:
    rel_ra, rel_de = prj_phi, prj_theta
    if debug:
        phi_range = prj_phi.max() - prj_phi.min()
        sys.stderr.write("phi range: %.4f < phi < %.4f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Shift to 
    old_fov = (0.0, 90.0, 0.0)
    new_fov = (crval1, crval2, 0.0)
    stuff = rfov.migrate_fov_deg(old_fov, new_fov, (rel_ra, rel_de))
    return stuff

## Convert X,Y to RA, Dec using CD matrix and CRVAL pair:
#def xycd2radec(cdmat, xpix, ypix, crval1, crval2, debug=False):
def xycd2radec(cdmat, rel_xx, rel_yy, crval1, crval2, debug=False):
    thisCD = np.array(cdmat).reshape(2, 2)
    #rel_xx = xpix - _aks_crpix1
    #rel_yy = ypix - _aks_crpix2
    #relpix = np.array([xpix - _aks_crpix1, ypix - _aks_crpix2])
    #relpix = np.array([rel_xx, rel_yy])
    #relpix = np.vstack((rel_xx, rel_yy))   # rel time = 3.81 us
    #relpix = np.array((rel_xx, rel_yy))    # rel time = 1.10 us
    relpix = np.array((rel_xx, rel_yy))     # rel time = 1.05 us
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)

## Convert RA, Dec back to relative X,Y using CD matrix and CRVAL pair:
def sky2xy_cd(cdmat, ra_deg, de_deg, crval1, crval2):
    inv_CD  = np.linalg.inv(cdmat)
    new_fov = (0.0, 90.0, 0.0)
    old_fov = (crval1, crval2, 0.0)
    prj_phi, prj_theta = rfov.migrate_fov_deg(old_fov, new_fov, (ra_deg, de_deg))
    prj_xx, prj_yy = _inv_tanproj(prj_phi, prj_theta)
    rel_foc = np.array((prj_xx, prj_yy))
    ccd_xrel, ccd_yrel = np.matmul(inv_CD, rel_foc)
    return ccd_xrel, ccd_yrel


def slow_make_cdmat(pa_deg, pscale):
    pa_rad = np.radians(pa_deg)
    thisCD = np.matmul(xflip_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
    return thisCD
    #rotmat = rotation_matrix(pa_rad)

def make_cdmat(pa_deg, pscale):
    pa_rad = math.radians(pa_deg)
    return np.matmul(xflip_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)

def xypas2radec(pa_deg, pscale, rel_xx, rel_yy, crval1, crval2):
    thisCD = make_cdmat(pa_deg, pscale)
    return xycd2radec(thisCD, rel_xx, rel_yy, crval1, crval2)

## Convert X,Y to RA, Dec (single-value), uses position angle and fixed pixel scale:
#def xypa2radec(pa_deg, xpix, ypix, crval1, crval2, channel, debug=False):
#def xypa2radec(pa_deg, rel_xx, rel_yy, crval1, crval2, channel, debug=False):
#    pa_rad = np.radians(pa_deg)
#    #rel_xx = xpix - _aks_crpix1
#    #rel_yy = ypix - _aks_crpix2
#    relpix = np.array([xpix - _aks_crpix1, ypix - _aks_crpix2])
#    #pscale = _pxscale[channel]
#    #sys.stderr.write("pscale: %.4f\n" % pscale)
#    #rotmat = rotation_matrix(pa_rad)
#    ##sys.stderr.write("rotmat:\n")
#    ##mprint(rotmat)
#    #rscale = (pscale / 3600.0) * rotmat
#    #sys.stderr.write("rscale:\n")
#    #mprint(rscale)
#    thisCD = np.matmul(xflip_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
#    #thisCD = np.dot(ident_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
#    #thisCD = np.matmul(ident_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
#    if debug:
#        sys.stderr.write("thisCD:\n")
#        mprint(thisCD)
#    #rel_ra, rel_de = np.dot(thisCD, relpix)
#
#    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)






######################################################################
# CHANGELOG (tangent_proj.py):
#---------------------------------------------------------------------
#
#  2023-06-02:
#     -- Increased __version__ to 0.0.1.
#     -- First created tangent_proj.py.
#

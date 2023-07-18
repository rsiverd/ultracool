#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module to provide RJS + AC polynomial correction.
#
# Rob Siverd
# Created:       2023-07-10
# Last modified: 2023-07-17
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.0"

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
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

### On-sky rotation module:
#import fov_rotation
#reload(fov_rotation)
#rfov = fov_rotation.RotateFOV()

## Polynomial fitter and evaluator
import custom_polyfit
reload(custom_polyfit)

## Tangent projection helper:
import tangent_proj
reload(tangent_proj)
tp = tangent_proj

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Channel mapping:
#_chmap = {
#        1:1,  '1':1, 'ch1':1,
#        2:2,  '2':2, 'ch2':2,
#        3:3,  '3':3, 'ch3':3,
#        4:4,  '4':4, 'ch4':4,
#}

### Dephasing parameter dictionaries:
#_ppa = {'cryo':{}, 'warm':{}}
#_ppb = {'cryo':{}, 'warm':{}}
##_ppa, _ppb = {}, {}
##_aaa, _bbb = {}, {}

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Polynomial coefficients for the 0..3rd order polynomial fit.
_dx_0_3_coeffs = np.array([ 4.38225342e-06,  5.79866087e-04,
      -2.54657087e-06, -1.36190257e-09,  3.18136532e-03,  1.33416373e-06,
       1.77402888e-12, -2.43219056e-06, -1.14151261e-09,  4.61142044e-10])
       
_dy_0_3_coeffs = np.array([ 8.78415469e-06,  7.06077818e-03,
       5.08390050e-06,  1.04492285e-09,  1.22368313e-02, -5.08562847e-06,
      -3.86608011e-10, -2.23433628e-05,  1.12148094e-09,  8.13490376e-09])

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Pixelscale in each channel:
#_pxscale = {
#    1   :   1.2232,
#    2   :   1.2144,
#    3   :   1.2247,
#    4   :   1.2217,
#}

## Pixel scale in each filter:
_initial_pxscale = 0.3042820
_pxscale = {
    'J'  :   _initial_pxscale,
    'H2' :    _initial_pxscale,
}

### UPDATED VALUES FROM JOINT FITTING:
#_pxscale[1] = 1.22328
#_pxscale[2] = 1.21451

## CRPIX pixel coordinate we use:
_wir_crpix1 = 2122.69077900
_wir_crpix2 =  -81.6788876100

## Median CFHT CD matrix:
_wir_cdmat  = np.array([-8.452717e-05, 1.310921e-08, 9.981597e-09, 8.451805e-05])

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Radian-to-degree converter:
_radeg = 180.0 / np.pi

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

#_abby_pixscale = 0.3043

## Expose through class:
class WIRCamPoly(object):

    def __init__(self):
        #self._dpar_a = _ka
        #self._dpar_b = _kb
        self._crpix1 = _wir_crpix1
        self._crpix2 = _wir_crpix2
        #self._ppar_a = _ppa
        #self._ppar_b = _ppb
        #self._missions = list(self._ppar_a.keys())
        #self._cwcutoff = _cryo_warm_cutoff
        # Default CD matrix:
        self._wir_cd = np.copy(_wir_cdmat)
        # Create and initialize X-axis distortion model:
        self._cpf_dx = custom_polyfit.CustomPolyFit2D()
        self._cpf_dx.set_degree(0, 3)
        self._cpf_dx.set_model(_dx_0_3_coeffs)
        # Create and initialize X-axis distortion model:
        self._cpf_dy = custom_polyfit.CustomPolyFit2D()
        self._cpf_dy.set_degree(0, 3)
        self._cpf_dy.set_model(_dy_0_3_coeffs)
        # Local copy of tangent projection:
        self._tp = tangent_proj
        return

    def calc_x_correction(self, xpix, ypix, instrument):
        return self._cpf_dx.eval(xpix - self._crpix1, ypix - self._crpix2)

    def calc_y_correction(self, xpix, ypix, instrument):
        return self._cpf_dy.eval(xpix - self._crpix1, ypix - self._crpix2)

    def calc_corrected_xy(self, xpix, ypix, instrument):
        x_nudges = self.calc_x_correction(xpix, ypix, instrument)
        y_nudges = self.calc_y_correction(xpix, ypix, instrument)
        return (xpix + x_nudges, ypix + y_nudges)

    def xy2focal(self, xpix, ypix, channel):
        """Convert X,Y detector positions (IN PIXEL UNITS!!!) to focal
        plane coordinates using Adam Kraus' solution."""
        prev_shape = xpix.shape
        xrel = xpix.flatten() - self._crpix1
        yrel = ypix.flatten() - self._crpix2
        xpar = self._dpar_a[channel]
        ypar = self._dpar_b[channel]
        xfoc = self._eval_axis(xrel, yrel, xpar)
        yfoc = self._eval_axis(xrel, yrel, ypar)
        #import pdb; pdb.set_trace()
        #xfoc = self._crpix1 + self._eval_axis(xrel, yrel, xpar)
        #yfoc = self._crpix2 + self._eval_axis(xrel, yrel, ypar)
        #return xrel + xfoc.reshape(prev_shape), yrel + yfoc.reshape(prev_shape)
        return xfoc.reshape(prev_shape), yfoc.reshape(prev_shape)

    def xform_xy(self, xpix, ypix, channel):
        """Convert X,Y detector positions (IN PIXEL UNITS!!!) to modified
        X',Y' (truer?) pixel coordinates using Adam Kraus' solution."""
        xfoc, yfoc = self.xy2focal(xpix, ypix, channel)
        newx = self._crpix1 + xfoc
        newy = self._crpix2 + yfoc
        return newx, newy

    # RA/DE calculation with internal handling of the CRPIXn:
    def predicted_radec(self, cdmat, xpix, ypix, crval1, crval2):
        use_cdm = cdmat if cdmat else self._wir_cd
        xrel = xpix.flatten() - self._crpix1
        yrel = ypix.flatten() - self._crpix2
        return self._tp.xycd2radec(use_cdm, xrel, yrel, crval1, crval2)

    #def dephase(self, xpix, ypix, mission, channel):
    #    if not self._have_dephase_pars(mission, channel):
    #        sys.stderr.write("Unsupported mission/channel combo: %s/%d\n"
    #                % (mission, channel))
    #        return None, None
    #    #prev_shape = xpix.shape
    #    #xrel = (xpix - np.floor(xpix)).flatten() - 0.5
    #    #yrel = (ypix - np.floor(ypix)).flatten() - 0.5
    #    xrel = (xpix - np.floor(xpix)) - 0.5
    #    yrel = (ypix - np.floor(ypix)) - 0.5
    #    xpar = self._ppar_a[mission][channel]
    #    ypar = self._ppar_b[mission][channel]
    #    newx = xpix + self._eval_axis(xrel, yrel, xpar)
    #    newy = ypix + self._eval_axis(xrel, yrel, ypar)
    #    return newx, newy

    #def _have_dephase_pars(self, mission, channel):
    #    if not mission in self._missions:
    #        sys.stderr.write("Unsupported mission: %s\n" % mission)
    #        sys.stderr.write("Available options: %s\n" % str(self._missions))
    #        return False
    #    if not channel in self._ppar_a[mission].keys():
    #        sys.stderr.write("No channel %d for %s mission!\n"
    #                % (channel, mission))
    #        return False
    #    return True

    #@staticmethod
    #def _eval_axis(x, y, pp):
    #    newpix  = np.zeros_like(x)
    #    newpix += pp[0]
    #    newpix += pp[1]*x + pp[2]*y
    #    newpix += pp[3]*x**2 + pp[4]*x*y + pp[5]*y**2
    #    newpix += pp[6]*x**3 + pp[7]*x**2*y + pp[8]*x*y**2 + pp[9]*y**3
    #    newpix += pp[10]*x**4 + pp[11]*x**3*y + pp[12]*x**2*y**2 \
    #                        + pp[13]*x*y**3 + pp[14]*y**4
    #    newpix += pp[15]*x**5 + pp[16]*x**4*y + pp[17]*x**3*y**2 \
    #            + pp[18]*x**2*y**3 + pp[19]*x*y**4 + pp[20]*y**5
    #    return newpix

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

######################################################################
# CHANGELOG (wircam_poly.py):
#---------------------------------------------------------------------
#
#  2023-07-10:
#     -- Increased __version__ to 0.1.0.
#     -- First created wircam_poly.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module to provide Adam Kraus' Spitzer distortion solution.
#
# Rob Siverd
# Created:       2019-09-04
# Last modified: 2019-11-10
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.3.0"

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

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Build parameters:
_ka, _kb = {}, {}
_ka[1] = np.array([-1.987895E-03, +4.848572E-04, -1.909555E-04, -1.870311E-05, +2.044248E-05, +3.535586E-06, -2.312227E-07, +1.145814E-09, -2.420678E-07, +1.881476E-08, -3.250621E-10, +9.190708E-11, -1.040106E-10, -4.774988E-11, +4.641016E-11, +4.242499E-12, -4.978284E-13, +2.999052E-12, -2.219252E-13, +3.029946E-12, -5.917890E-13])
_kb[1] = np.array([+7.389430E-04, -8.399943E-04, -4.172204E-05, -9.536498E-07, -2.196779E-05, +2.340636E-05, +3.181545E-08, -1.927947E-07, +3.026354E-09, -1.725757E-07, +2.378517E-10, -9.549596E-11, -2.028054E-11, -9.499672E-11, -2.431012E-11, -3.160712E-12, -2.879781E-14, +2.327181E-13, +2.085189E-12, -6.830374E-13, -2.113676E-13])
_ka[2] = np.array([-5.741790E-05, +1.140810E-03, -8.599681E-05, +2.108325E-05, +2.941094E-05, +4.087327E-07, -1.599348E-07, +1.558857E-08, -2.030561E-07, +1.573384E-08, -1.081135E-10, +5.459438E-11, -2.647650E-12, -1.476344E-11, +1.651644E-11, -6.800286E-13, -7.283961E-13, +4.910761E-14, -4.495334E-13, +3.622077E-13, -1.006904E-12])
_kb[2] = np.array([+1.168688E-03, +6.795485E-04, +1.038700E-03, +5.767053E-06, +1.947397E-05, +3.492700E-05, -2.845218E-09, -2.057702E-07, -9.306303E-09, -1.744583E-07, -5.081412E-11, -2.867238E-11, -1.835628E-11, -2.576269E-11, -1.027837E-11, +1.565123E-13, +1.199605E-12, +4.593593E-13, +4.872015E-13, +6.879575E-13, -5.461638E-13])
_ka[3] = np.array([+5.368236E-03, +1.227243E-03, -2.090547E-04, -2.399988E-05, +3.384537E-05, -6.221500E-07, -4.867351E-09, +2.078667E-08, -1.638714E-07, +1.772616E-08, +2.996680E-10, -4.028182E-11, +5.767441E-11, +1.641023E-10, -1.128943E-10, -7.751036E-12, +7.377048E-13, -1.783869E-12, -1.278090E-12, +1.748810E-12, -9.794058E-13])
_kb[3] = np.array([+4.876272E-03, -3.383015E-03, -1.587592E-03, -9.295833E-06, -9.674543E-06, +2.349712E-05, -1.761487E-08, -1.578380E-07, +2.460535E-08, -1.182984E-07, +1.890677E-10, -3.246752E-11, +4.169965E-11, +6.119891E-12, +6.183478E-11, +7.396466E-13, +9.370199E-13, +1.502810E-12, -2.389407E-13, -2.048480E-12, -2.115334E-12])
_ka[4] = np.array([+7.229849E-03, +2.662657E-03, -4.308010E-04, +3.110385E-05, +5.271226E-05, +1.301581E-05, -2.549827E-07, +1.120662E-08, -1.888047E-07, -1.372281E-08, -9.923747E-11, -4.245489E-10, -1.116774E-10, -8.149722E-11, -1.238051E-10, +5.009643E-12, +3.457558E-12, -4.395776E-12, -5.377011E-13, +3.728625E-12, +5.916105E-13])
_kb[4] = np.array([+5.199843E-03, +3.041185E-03, -2.016066E-03, -3.051650E-06, +2.050006E-05, +4.345048E-05, +1.209439E-07, -1.190427E-07, +3.520365E-08, -1.806912E-07, +4.478562E-12, -4.181739E-10, -2.695066E-10, -2.332946E-10, -1.768701E-11, -5.066513E-12, -5.723793E-13, +4.107264E-13, -2.581602E-12, -6.170103E-13, +7.363110E-13])
for kk in _ka.keys():
    _ka[kk][0]  = 0.0
    _ka[kk][1] += 1.0
    _kb[kk][0]  = 0.0
    _kb[kk][2] += 1.0

## Expose through class:
class AKSPoly(object):

    def __init__(self):
        self._a_pars = _ka
        self._b_pars = _kb
        self._crpix1 = 128.0
        self._crpix2 = 128.0
        return

    def xy2focal(self, xpix, ypix, channel):
        """Convert X,Y detector positions (IN PIXEL UNITS!!!) to focal
        plane coordinates using Adam Kraus' solution."""
        prev_shape = xpix.shape
        xrel = xpix.flatten() - self._crpix1
        yrel = ypix.flatten() - self._crpix2
        xpar = self._a_pars[channel]
        ypar = self._b_pars[channel]
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

    @staticmethod
    def _eval_axis(x, y, pp):
        newpix  = np.zeros_like(x)
        newpix += pp[0]
        newpix += pp[1]*x + pp[2]*y
        newpix += pp[3]*x**2 + pp[4]*x*y + pp[5]*y**2
        newpix += pp[6]*x**3 + pp[7]*x**2*y + pp[8]*x*y**2 + pp[9]*y**3
        newpix += pp[10]*x**4 + pp[11]*x**3*y + pp[12]*x**2*y**2 \
                            + pp[13]*x*y**3 + pp[14]*y**4
        newpix += pp[15]*x**5 + pp[16]*x**4*y + pp[17]*x**3*y**2 \
                + pp[18]*x**2*y**3 + pp[19]*x*y**4 + pp[20]*y**5
        return newpix

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

######################################################################
# CHANGELOG (kraus.py):
#---------------------------------------------------------------------
#
#  2019-11-10:
#     -- Increased __version__ to 0.3.0.
#     -- Fixed critical error in parameter list causing gross miscalculation
#           of Y-coordinates. Module should work as intended now.
#
#  2019-09-05:
#     -- Increased __version__ to 0.2.0.
#     -- Older xform_xy() now relies on xy2focal().
#     -- Added xy2focal() method to produce focal plane coordinates. This new
#           routine also handles flatten/reshape internally to better mimic
#           the API of astropy WCS routines.
#
#  2019-09-04:
#     -- Increased __version__ to 0.1.0.
#     -- First created akspoly.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module implements the empirical error model trained on Spitzer
# astrometric fit residuals.
#
# Rob Siverd
# Created:       2021-09-20
# Last modified: 2021-09-20
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
#import glob
import os
import sys
import time
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

## Home-brew KDE:
#try:
#    import my_kde
#    reload(my_kde)
#    mk = my_kde
#except ImportError:
#    logger.error("module my_kde not found!  Install and retry.")
#    sys.stderr.write("\nError!  my_kde module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

## Fast FITS I/O:
#try:
#    import fitsio
#except ImportError:
#    logger.error("fitsio module not found!  Install and retry.")
#    sys.stderr.write("\nError: fitsio module not found!\n")
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
##------------------    Spitzer Empirical Error Model       ----------------##
##--------------------------------------------------------------------------##

## GAIN and FLUXCONV:
_irac1_sensor_pars = (3.70, 0.1257)
_irac2_sensor_pars = (3.71, 0.1447)

## GAIN / FLUXCONV multiplier:
_irac1_sensor_mult = _irac1_sensor_pars[0] / _irac1_sensor_pars[1]
_irac2_sensor_mult = _irac2_sensor_pars[0] / _irac2_sensor_pars[1]

## Error model parameters:
_irac1_asterr_pars = (2987.0, 45.26)
_irac2_asterr_pars = (3314.8, 29.85)

## Error model implementation:
class SpitzErrorModel(object):

    def __init__(self):
        self._signal_norm = {1:_irac1_sensor_mult, 2:_irac2_sensor_mult}
        self._noise_model = {1:_irac1_asterr_pars, 2:_irac2_asterr_pars}
        return

    def signal2counts(self, signal, channel):
        return signal * self._signal_norm[channel]

    def counts2error(self, counts, channel):
        big_fwhm, noise_floor = self._noise_model[channel]
        star_snr = np.sqrt(counts)
        star_rms = big_fwhm / star_snr
        return np.sqrt(star_rms**2 + noise_floor**2)

    def fluxexp2error(self, flux, exptime, channel):
        counts = self.signal2counts(flux * exptime, channel)
        return self.counts2error(counts, channel)

    def signal2error(self, signal, channel):
        counts = self.signal2counts(signal, channel)
        return self.counts2error(counts, channel)

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (spitz_error_model.py):
#---------------------------------------------------------------------
#
#  2021-09-20:
#     -- Increased __version__ to 0.1.0.
#     -- First created spitz_error_model.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# First cut at astrometry fitting for UCD project.
#
# Rob Siverd
# Created:       2020-02-07
# Last modified: 2020-02-07
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
__version__ = "0.1.0"

## Modules:
#import glob
import gc
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
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
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

##--------------------------------------------------------------------------##
##------------------       Astrometry Fitting (5-par)       ----------------##
##--------------------------------------------------------------------------##

_ARCSEC_PER_RADIAN = 180. * 3600.0 / np.pi
class AstFit(object):

    def __init__(self):
        return

    @staticmethod
    def _calc_parallax_factors(RA_rad, DE_rad, X_au, Y_au, Z_au):
        sinRA, cosRA = np.sin(RA_rad), np.cos(RA_rad)
        sinDE, cosDE = np.sin(DE_rad), np.cos(DE_rad)
        ra_factor = (X_au * sinRA - Y_au * cosRA) / cosDE
        de_factor =  X_au * cosRA * sinDE \
                  +  Y_au * sinRA * sinDE \
                  -  Z_au * cosDE
        return ra_factor, de_factor

    def apparent_radec(self, t_ref, astrom_pars, eph_obs):
        """
        t_ref       --  chosen reference epoch
        astrom_pars --  five astrometric parameters specified at the
                        reference epoch: meanRA (rad), meanDE (rad),
                        pmRA*cos(DE), pmDE, and parallax
        eph_obs     --  dict with x,y,z,t elements describing the times
                        and places of observations (numpy arrays)
        FOR NOW, assume
                    [t_ref] = JD (TDB)
                    [t]     = JD (TDB)
                    [pars]  = rad, rad, arcsec/yr, arcsec/yr, arcsec
                                       *no cos(d)*
        """
    
        rra, rde, pmra, pmde, prlx = astrom_pars
    
        t_diff_yr = (eph_obs['t'] - t_ref) / 365.25     # units of years
    
        pfra, pfde = self._calc_parallax_factors(rra, rde,
                eph_obs['x'], eph_obs['y'], eph_obs['z'])
    
        delta_ra = (t_diff_yr * pmra / _ARCSEC_PER_RADIAN) + (prlx * pfra)
        delta_de = (t_diff_yr * pmde / _ARCSEC_PER_RADIAN) + (prlx * pfde)
    
        return (rra + delta_ra, rde + delta_de)

######################################################################
# CHANGELOG (astrom_test.py):
#---------------------------------------------------------------------
#
#  2020-02-07:
#     -- Increased __version__ to 0.1.0.
#     -- First created astrom_test.py.
#

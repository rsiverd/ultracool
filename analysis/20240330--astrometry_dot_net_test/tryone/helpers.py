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
import sys
import time
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

## Tangent projection:
import tangent_proj as tp


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

## Get an astropy Time object from WIRCAM header:
def wircam_timestamp_from_header(header):
    obs_time = astt.Time(header['MJD-OBS'], scale='utc', format='mjd') \
            + 0.5 * astt.TimeDelta(header['EXPTIME'], format='sec')
    return obs_time






######################################################################
# CHANGELOG (helpers.py):
#---------------------------------------------------------------------
#
#  2025-02-03:
#     -- Increased __version__ to 0.0.1.
#     -- First created helpers.py.
#

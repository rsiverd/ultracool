#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module that provides coordinate residual detrending capability.
#
# Rob Siverd
# Created:       2022-03-10
# Last modified: 2022-03-10
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
#import matplotlib.patches as mpatches
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

## Get list of corresponding indexes in pts2 for values in pts1:
def makedex_dumb(pts1, pts2):
    ## ASSUMES UNIQUE have_pts
    lut = {x:i for x,i in zip(pts2, np.arange(len(pts2)))}
    idxpairs = []
    # iterate over pts1, identify correspondences:
    for ii,xx in enumerate(pts1):
        if xx in lut.keys():
            idxpairs.append((ii, lut[xx]))
    return idxpairs

## Build single corresponding trend vector:
def patched_matched_residuals(want_pts, have_pts, trvec):
    lookup = {p:i for p,i in zip(want_pts, np.arange(len(want_pts)))}
    matched_data = np.zeros_like(want_pts, dtype='float32')
    for jj,rr in zip(have_pts, trdata):
        #sys.stderr.write("jj: %s, rr: %s\n" % (str(jj), str(rr)))
        if jj in lookup.keys():
            matched_data[lookup[jj]] = rr
    return matched_data

## Build multiple corresponding trend vectos:
def patched_matched_residuals(want_pts, have_pts, trvec_list):
    pairs = makedex_dumb(want_pts, have_pts)
    mdidx, tvidx = zip(*pairs)
    matched_data = []
    for tv in trvec_list:
        mdata = np.zeros_like(want_pts, dtype='float32')
        mdata[mdidx,] = tv[tvidx,]
        matched_data.append(mdata)
    return matched_data


##--------------------------------------------------------------------------##
##------------------         Coordinate Detrending          ----------------##
##--------------------------------------------------------------------------##

class CoordDetrending(object):

    def __init__(self):
        return

    # Get list of corresponding indexes in pts2 for values in pts1:
    @staticmethod
    def _makedex_dumb(pts1, pts2):
        ## ASSUMES UNIQUE have_pts
        lut = {x:i for x,i in zip(pts2, np.arange(len(pts2)))}
        idxpairs = []
        # iterate over pts1, identify correspondences:
        for ii,xx in enumerate(pts1):
            if xx in lut.keys():
                idxpairs.append((ii, lut[xx]))
        return idxpairs

    # Build single corresponding trend vector:
    def _patched_matched_values(want_pts, have_pts, trvec):
        lookup = {p:i for p,i in zip(want_pts, np.arange(len(want_pts)))}
        matched_data = np.zeros_like(want_pts, dtype='float32')
        for jj,rr in zip(have_pts, trdata):
            #sys.stderr.write("jj: %s, rr: %s\n" % (str(jj), str(rr)))
            if jj in lookup.keys():
                matched_data[lookup[jj]] = rr
        return matched_data

    # Build multiple corresponding trend vectos:
    def _patched_matched_values_multi(want_pts, have_pts, trvec_list):
        pairs = self._makedex_dumb(want_pts, have_pts)
        mdidx, tvidx = zip(*pairs)
        matched_data = []
        for tv in trvec_list:
            mdata = np.zeros_like(want_pts, dtype='float32')
            mdata[mdidx,] = tv[tvidx,]
            matched_data.append(mdata)
        return matched_data

##--------------------------------------------------------------------------##
## Quick ASCII I/O:
#data_file = 'data.txt'
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = np.atleast_1d(np.genfromtxt(data_file, dtype=None, **gftkw))
#all_data = np.genfromtxt(fix_hashes(data_file), dtype=None, **gftkw)
#all_data = aia.read(data_file)

#pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
#all_data = pd.read_csv(data_file)
#all_data = pd.read_csv(data_file, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)
#nskip, cnames = analyze_header(data_file)
#all_data = pd.read_csv(data_file, names=cnames, skiprows=nskip, **pdkwargs)

#all_data.rename(columns={'old_name':'new_name'}, inplace=True)
#all_data.reset_index()
#firstrow = all_data.iloc[0]
#for ii,row in all_data.iterrows():
#    pass

#vot_file = 'neato.xml'
#vot_data = av.parse_single_table(vot_file)
#vot_data = av.parse_single_table(vot_file).to_table()

##--------------------------------------------------------------------------##


######################################################################
# CHANGELOG (detrending.py):
#---------------------------------------------------------------------
#
#  2022-03-10:
#     -- Increased __version__ to 0.1.0.
#     -- First created detrending.py.
#

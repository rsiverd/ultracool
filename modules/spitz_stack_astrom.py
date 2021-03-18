#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module provides the machinery needed to generate hybrid data
# catalogs containing positions derived using WCS of individual images
# and centroids measured on a stacked frame.
#
# Rob Siverd
# Created:       2021-03-17
# Last modified: 2021-03-18
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
#import vaex
#import calendar
#import ephem
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

##--------------------------------------------------------------------------##

## Fast FITS I/O:
#try:
#    import fitsio
#except ImportError:
#    logger.error("fitsio module not found!  Install and retry.")
#    sys.stderr.write("\nError: fitsio module not found!\n")
#    sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.exit(1)

##--------------------------------------------------------------------------##

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------         Hybrid Astrometry Class        ----------------##
##--------------------------------------------------------------------------##

class HybridAstrom(object):

    _XCOLS = ('x', 'wx', 'ppx')
    _YCOLS = ('y', 'wy', 'ppy')
    _CCMAP = {(  'x',   'y')  :  (  'dra',   'dde'),
              ( 'wx',  'wy')  :  ( 'wdra',  'wdde'),
              ('ppx', 'ppy')  :  ('ppdra', 'ppdde'),
              }

    def __init__(self):
        #self._xinfo   = None
        self._sxcdata = None
        self._xshifts = None
        self._yshifts = None
        return

    def set_stack_excat(self, stack_cat):
        self._sxcdata = stack_cat.get_catalog().copy()
        return

    def set_xcorr_metadata(self, xinfo):
        #self._xinfo = xinfo
        xshifts, yshifts = xinfo.get_stackcat_offsets()
        bxshifts = {os.path.basename(k):v for k,v in xshifts.items()}
        byshifts = {os.path.basename(k):v for k,v in yshifts.items()}
        self._xshifts = {**xshifts, **bxshifts}
        self._yshifts = {**yshifts, **byshifts}
        return

    def make_hybrid_excat(self, imexcat):
        imname = imexcat.get_imname()
        header = imexcat.get_header()
        if not imname in self._xshifts.keys():
            sys.stderr.write("Unrecognized image: %s\n" % imname)
            raise
        hdata = self._sxcdata.copy()

        # update X- and Y-coordinates:
        for cc in self._XCOLS:
            hdata[cc] = hdata[cc] + self._xshifts[imname]
        for cc in self._YCOLS:
            hdata[cc] = hdata[cc] + self._yshifts[imname]

        # update RA and Dec columns:
        wcs = awcs.WCS(header)
        for (xc,yc),(rc,dc) in self._CCMAP.items():
            hdata[rc], hdata[dc] = wcs.all_pix2world(hdata[xc], hdata[yc], 1,
                    ra_dec_order=True)

        # build ExtendedCatalog:
        ecopts = {'name':imname, 'header':header}
        return ec.ExtendedCatalog(data=hdata, **ecopts)


##--------------------------------------------------------------------------##

######################################################################
# CHANGELOG (spitz_stack_astrom.py):
#---------------------------------------------------------------------
#
#  2021-03-17:
#     -- Increased __version__ to 0.1.0.
#     -- First created spitz_stack_astrom.py.
#

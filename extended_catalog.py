#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This class provides a useful container for detected objects and image 
# metadata needed for further analysis (cross-identification and fitting).
# It also provides load/store capability to assist in development and
# share data with non-Python code.
#
# Rob Siverd
# Created:       2019-10-12
# Last modified: 2019-10-12
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
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
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

## FITS I/O:
#try:
#    import astropy.io.fits as pf
#except ImportError:
#    try:
#       import pyfits as pf
#    except ImportError:
#        logger.error("No FITS I/O module found!"
#                "Install either astropy.io.fits or pyfits and retry."))
#        logger.error("No FITS I/O module found!")
#        sys.stderr.write("\nError!  No FITS I/O module found!\n"
#               "Install either astropy.io.fits or pyfits and try again!\n\n")
#        sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
    import astropy.io.fits as pf
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
## Save FITS image with clobber (astropy / pyfits):
#def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    if header:
#        while (len(header) < padkeys):
#            header.append() # pad header
#    if os.path.isfile(iname):
#        os.remove(iname)
#    pf.writeto(iname, idata, header=header, **kwargs)
#    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
## Save FITS image with clobber (fitsio):
#def qsave(iname, idata, header=None, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    #if os.path.isfile(iname):
#    #    os.remove(iname)
#    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
#    sys.stderr.write("done.\n")




##--------------------------------------------------------------------------##
##------------------         Image Results class            ----------------##
##--------------------------------------------------------------------------##

_CAT_EXT = 'CATALOG'

class ExtendedCatalog(object):
    """Flexible storage medium that encapsulates a catalog of extracted
    objects along with relevant metadata from the source image."""

    def __init__(self, data=None, name=None, header=None):
        self._imcat = data
        self._iname = name
        self._imhdr = header
        self._imeta = None
        return

    # ---------------------------------------
    # Getters/setters:
    def set_catalog(self, data):
        self._imcat = data
        return

    def set_imname(self, iname):
        self._iname = iname
        return

    def set_header(self, header):
        self._imhdr = header
        return

    # ---------------------------------------
    # Catalog I/O:
    def save_as_fits(self, filename, **kwargs):
        """Save extended catalog information to FITS file. kwargs are
        passed to the fits.writeto() method."""

        if not self._have_required_data():
            logging.warning("data missing, output not saved!")
            return
        hdr = pf.Header()   # make this from stored metadata!
        tab = pf.BinTableHDU(data=self._imcat, header=hdr, name=_CAT_EXT)
        tab.writeto(filename, **kwargs)
        return

    def load_from_fits(self, filename):
        """Load extended catalog information from FITS file."""
        tab, hdr = pf.getdata(filename, header=True, extname=_CAT_EXT)
        return

    # ---------------------------------------
    # Helpers:
    def _have_required_data(self):
        n_missing = 0
        if self._imcat == None:
            logging.warning("object catalog not set!")
            n_missing += 1
        if self._iname == None:
            logging.warning("image name not set!")
            n_missing += 1
        if self._imeta == None:
            logging.warning("image metadata not set!")
            n_missing += 1
        if n_missing > 0:
            return False
        else:
            return True
 
##--------------------------------------------------------------------------##


######################################################################
# CHANGELOG (image_results.py):
#---------------------------------------------------------------------
#
#  2019-10-12:
#     -- Increased __version__ to 0.1.0.
#     -- First created extended_catalog.py.
#

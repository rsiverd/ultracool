#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Routines for catalog comparison, pruning, and updates.
#
# Rob Siverd
# Created:       2021-02-15
# Last modified: 2021-02-15
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
#import shutil
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
##------------------         Catalog Pruning Tools          ----------------##
##--------------------------------------------------------------------------##

## The following class is used to prune objects from an ExtendedCatalog
## using a master catalog (from stacking) and cross-correlation results.
class XCorrPruner(object):

    def __init__(self):
        self._mcatalog = None
        self._xshifts  = None
        self._yshifts  = None
        self._prev_msep = None
        return

    def set_master_catalog(self, data):
        self._mcatalog = data.copy()
        return

    def set_image_offsets(self, xshifts, yshifts):
        self._xshifts = {k:v for k,v in xshifts.items()}
        self._yshifts = {k:v for k,v in yshifts.items()}
        return

    def prune_spurious(self, cdata, ipath, rcut=2.0):
        #xcol, ycol = 'x', 'y'
        xcol, ycol = 'wx', 'wy'
 
        #tcat = cdata.copy()     # working copy of individual frame catalog
        #tcat[xcol] -= self._xshifts[ipath]
        #tcat[ycol] -= self._yshifts[ipath]

        tx = cdata[xcol] - self._xshifts[ipath]
        ty = cdata[ycol] - self._yshifts[ipath]
        mx, my = self._mcatalog[xcol], self._mcatalog[ycol]
        #sx, sy = shifted_cat[xcol], shifted_cat[

        keep = []
        mseps = []
        #for ii,(sx,sy) in enumerate(zip(tcat[xcol], tcat[ycol]), 1):
        for ii,(sx,sy) in enumerate(zip(tx, ty), 1):
            dx = sx - mx
            dy = sy - my
            ds = np.hypot(sx - mx, sy - my)
            minsep = ds.min()
            keep.append(minsep < rcut)
            mseps.append(minsep)
            pass
        self._prev_msep = np.array(mseps)
        return cdata.copy()[keep]

    #def initialize_from_xcorr(self, xcobj):
    #    self._mcatalog = xcobj.get_catalog().copy()
    #    self._xshifts, self._yshifts = xcobj.get_stackcat_offsets()
    #    return

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (catalog_cools.py):
#---------------------------------------------------------------------
#
#  2021-02-15:
#     -- Increased __version__ to 0.1.0.
#     -- First created catalog_cools.py.
#

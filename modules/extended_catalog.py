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
from collections.abc import Iterable
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

## Fast FITS I/O:
try:
    import fitsio
except ImportError:
    logger.error("fitsio module not found!  Install and retry.")
    sys.exit(1)

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

## Extension names:
_EXT = {
        'cat'   :   'CATALOG',
        'ihdr'  :   'IMGHEADER',
        'uhdr'  :   'UNCHEADER',
        }

## Components preserved by load/store:
#_PRESERVED = ['_imcat', '_iname', '_imhdr', '_uname', '_unhdr', '_imeta']
_PRESERVED = ['_imcat', '_imeta', '_imhdr', '_unhdr']

## Storage mapping for image types:
_imtypes = {
        'img'   :   '_imhdr',
        'unc'   :   '_unhdr'
        }

## Ephemeris contents:
_EPH_KEYS = ('jdtdb', 'obs_x', 'obs_y', 'obs_z', 'obs_vx', 'obs_vy', 'obs_vz')

## Container class:
class ExtendedCatalog(object):
    """Flexible storage medium that encapsulates a catalog of extracted
    objects along with relevant metadata from the source image."""

    def __init__(self, data=None, name=None, header=None,
            uname=None, uheader=None):
        self._imhdr = None
        self._unhdr = None
        self._imcat = data
        self._imeta = {
                'INAME'   :  name,
                'UNAME'   : uname,
                'ECVERS'  : __version__,
                }
        self.set_header(header, which='img')
        self.set_header(uheader, which='unc')
        return

    # --------------------------------------- #
    #           getters and setters           #
    # --------------------------------------- #

    def set_ephem(self, eph_dict):
        # take no action in case of incomplete ephemeris:
        if not all([(x in eph_dict.keys()) for x in _EPH_KEYS]):
            sys.stderr.write("Ephemeris dict is incomplete:\n")
            sys.stderr.write("Expected keys: %s\n" % str(_EPH_KEYS))
            sys.stderr.write("Received keys: %s\n" % str(eph_dict.keys()))
            return
        # store a copy of ephemeris in _imeta (upper-case keys):
        for kk in _EPH_KEYS:
            self._imeta[kk.upper()] = eph_dict[kk]

        return

    def get_catalog(self):
        return self._imcat.copy()

    def set_catalog(self, data):
        self._imcat = data
        return

    def get_imname(self):
        return self._imeta['INAME']

    def set_imname(self, iname):
        self._imeta['INAME'] = iname
        return

    def get_header(self, which='img'):
        if not which in _imtypes:
            sys.stderr.write("FIXME: unhandled which '%s'\n" % which)
            return
        return getattr(self, _imtypes[which])

    def set_header(self, header, which='img'):
        if not which in _imtypes:
            sys.stderr.write("FIXME: unhandled which '%s'\n" % which)
            return
        if isinstance(header, pf.Header):
            thdr = header.copy(strip=True)
            if ('EXTNAME' in thdr):
                thdr.pop('EXTNAME')
            setattr(self, _imtypes[which], thdr)
        #else:
        #    logging.warn("ignoring non-header (imhdr not set)")
        return

    # --------------------------------------- #
    #              catalog I/O                #
    # --------------------------------------- #
 
    # Write structure contents to FITS file:
    def save_as_fits(self, filename, **kwargs):
        """Save extended catalog information to FITS file. kwargs are
        passed to the fits.writeto() method."""

        if not self._have_required_data():
            logging.warning("data missing, output not saved!")
            return
        self._imeta['SAVEDATE'] = self._current_timestamp()
        hdu_list = pf.HDUList([pf.PrimaryHDU()])
        hdu_list.append(self._make_catalog_hdu())
        if isinstance(self._imhdr, pf.Header):
            hdu_list.append(self._header_only_hdu(self._imhdr, _EXT['ihdr']))
        if isinstance(self._unhdr, pf.Header):
            hdu_list.append(self._header_only_hdu(self._unhdr, _EXT['uhdr']))
        hdu_list.writeto(filename, **kwargs)
        return

    # Reload structure contents from FITS file:
    def load_from_fits(self, filename):
        """Load extended catalog information from FITS file."""

        with pf.open(filename, mode='readonly') as hdulist:
            for hdu in hdulist[1:]:
                logging.debug("hdu.name: %s" % hdu.name)
                if (hdu.name == _EXT['cat']):
                    cln_data = self._unfitsify_recarray(hdu.data)
                    self.set_catalog(cln_data) #.byteswap().newbyteorder())
                    self._imeta.update(self._meta_from_header(hdu.header))
                if (hdu.name == _EXT['ihdr']):
                    self.set_header(hdu.header, which='img')
                if (hdu.name == _EXT['uhdr']):
                    self.set_header(hdu.header, which='unc')
        #cdata, chdrs = fitsio.read(filename, header=True, ext=_EXT['cat'])
        #self.set_catalog(cdata)
        #self._imeta.update(self._meta_from_header(chdrs))
        return

    # Structure data comparison (helps test store/load):
    def has_same_data(self, othercat):
        for item in _PRESERVED:
            sys.stderr.write("Checking component: %s ... " % item)
            this_one = getattr(self, item)
            that_one = getattr(othercat, item)
            looks_ok = self._truth_summary(this_one == that_one)
            if not looks_ok:
                sys.stderr.write("not equal!\n")
                return False
            else:
                sys.stderr.write("equal!\n")
        return True

    # --------------------------------------- #
    #                helpers                  #
    # --------------------------------------- #

    # Ensure necessary components are set:
    def _have_required_data(self):
        n_missing = 0
        if not isinstance(self._imcat, np.ndarray):
            logging.warning("object catalog not set!")
            n_missing += 1
        if self._imeta == None:
            logging.warning("image metadata not set!")
            n_missing += 1
        if self._imeta['INAME'] == None:
            logging.warning("image name not set!")
            n_missing += 1
        if n_missing > 0:
            return False
        else:
            return True

    # Summarize truth value of argument by using all() on iterables:
    @staticmethod
    def _truth_summary(thing):
        return all(thing) if isinstance(thing, Iterable) else thing

    @staticmethod
    def _current_timestamp():
        now_sec = time.time()
        now_gmt = time.gmtime(now_sec)
        dsec = now_gmt.tm_sec + (now_sec % 1.0)
        time_obs = "%02d:%02d:%06.3f" \
              % (now_gmt.tm_hour, now_gmt.tm_min, dsec)
        date_obs = "%04d-%02d-%02d" \
              % (now_gmt.tm_year, now_gmt.tm_mon, now_gmt.tm_mday)
        return "%sT%s" % (date_obs, time_obs)

    @staticmethod
    def _prune_header(header):
        cleaned = header.copy(strip=True)
        if ('EXTNAME' in cleaned):
            cleaned.pop('EXTNAME')
        return cleaned

    @staticmethod
    def _header_only_hdu(header, extname):
        return pf.BinTableHDU(header=header.copy(), name=extname)

    def _meta_to_header(self):
        hdr = pf.Header()
        hdr.update(self._imeta)
        return hdr

    def _meta_from_header(self, header):
        return dict(self._prune_header(header).items())

    def _make_catalog_hdu(self):
        return pf.BinTableHDU(data=self._imcat,
                header=self._meta_to_header(), name=_EXT['cat'])

    # Unpack a FITSRecord into a plain record array:
    @staticmethod
    def _unfitsify_recarray(fitsrec):
        cols = fitsrec.dtype.names
        vals = [fitsrec[x].byteswap().newbyteorder() for x in cols]
        return np.core.records.fromarrays(vals, names=','.join(cols))

##--------------------------------------------------------------------------##


######################################################################
# CHANGELOG (image_results.py):
#---------------------------------------------------------------------
#
#  2019-10-12:
#     -- Increased __version__ to 0.1.0.
#     -- First created extended_catalog.py.
#

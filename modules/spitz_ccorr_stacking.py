#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# The routines in this module are used to shift and stack a list of
# images obtained from a single AOR/channel in Spitzer/IRAC. Bright
# pixels are identified by thresholding and rapidly cross-correlated
# in X and Y to find offsets. The 
#
# Rob Siverd
# Created:       2021-02-02
# Last modified: 2021-02-02
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
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
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
try:
    import robust_stats
    reload(robust_stats)
    rs = robust_stats
except ImportError:
    logger.error("module robust_stats not found!  Install and retry.")
    sys.stderr.write("\nError!  robust_stats module not found!\n"
           "Please install and try again ...\n\n")
    sys.exit(1)

## Fast FITS I/O:
try:
    import fitsio
except ImportError:
    logger.error("fitsio module not found!  Install and retry.")
    sys.stderr.write("\nError: fitsio module not found!\n")
    sys.exit(1)

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
def qsave(iname, idata, header=None, **kwargs):
    this_func = sys._getframe().f_code.co_name
    parent_func = sys._getframe(1).f_code.co_name
    sys.stderr.write("Writing to '%s' ... " % iname)
    #if os.path.isfile(iname):
    #    os.remove(iname)
    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------        Cross-Correlator Class          ----------------##
##--------------------------------------------------------------------------##

class SpitzerXCorr(object):

    def __init__(self, vlevel=0):
        self._vlevel = vlevel
        self._bp_thresh = 20.0
        self._reset()
        return

    def _reset(self):
        self._dimen    = (None, None)
        self._padpix   = 0
        self._im_paths = []
        self._im_data  = []
        self._bp_masks = []
        self._row_sums = [] # formerly xsmashed
        self._col_sums = [] # formerly ysmashed
        self._x_shifts = []
        self._y_shifts = []
        return

    # --------------------------------------------------------- #
    #                  Parameter Adjustment:                    #
    # --------------------------------------------------------- #

    def set_bp_thresh(self, thresh):
        self._bp_thresh = thresh
        return

    def set_vlevel(self, vlevel):
        self._vlevel = vlevel
        return

    # --------------------------------------------------------- #
    #                  High-Level Routines:                     #
    # --------------------------------------------------------- #

    def shift_and_stack(self, img_list):
        result = {'error':None}

        # load images, patch NaNs, set dimen/padding:
        self._load_frames(img_list)

        # generate bright pixel masks:
        self._make_bp_masks(thresh=self._bp_thresh)

        # UNPADDED cross-correlation to find pixel shifts.
        # Sum across rows to produce average column along columns
        # for average row:
        self._run_xy_xcorr(self._bp_masks)

        return result

    #def _make_result(self):
    #    result = 

    # --------------------------------------------------------- #
    #               Cross-Correlation Helpers:                  #
    # --------------------------------------------------------- #


    def _load_frames(self, img_list):
        self._im_paths = [x for x in img_list]
        self._im_data  = [self._nanfix(fitsio.read(ff)) \
                for ff in self._im_paths]
        self._dimen = self._im_data[0].shape
        self._padpix = max([int(0.5 * pix) for pix in self._dimen])
        return

    # Cross-correlation of reduced bright pixel masks:
    def _run_xy_xcorr(self, frames):
        # UNPADDED cross-correlation to find pixel shifts.
        # Sum across rows to produce average column along columns
        # for average row:
        xsmashed = [np.sum(im, axis=1) for im in frames]    # sum each row
        ysmashed = [np.sum(im, axis=0) for im in frames]    # sum each col
        self._row_sums, self._col_sums = xsmashed, ysmashed

        # Cross-correlate to find pixel shifts:
        xnudges = [self.qcorr(ysmashed[0], rr) for rr in ysmashed]
        ynudges = [self.qcorr(xsmashed[0], cc) for cc in xsmashed]
        self._x_shifts, self._y_shifts = xnudges, ynudges
        return

    @staticmethod
    def _nanfix(idata):
        which = np.isnan(idata) | np.isinf(idata)
        replacement = np.median(idata[~which])
        fixed = idata.copy()
        fixed[which] = replacement
        return fixed

    # create bright pixel masks for cross-correlation:
    def _make_bp_masks(self, thresh):
        self._bp_masks = []
        for frame in self._im_data:
            pix_med, pix_iqrn = rs.calc_ls_med_IQR(frame)
            bright = (frame - pix_med >= thresh * pix_iqrn)
            self._bp_masks.append(bright)
        return

    # 1-D cross-correlation driver routine:
    def qcorr(self, rowcol1, rowcol2):
        npix = rowcol1.size
        corr = self._ccalc(rowcol1, rowcol2)
        nshift = corr.argmax()
        if self._vlevel >= 1:
            self._qc_report(npix, corr, nshift)
        if (nshift > 0.5*npix):
            nshift -= npix
        return nshift

    # Performs FFT-assisted cross-correlation:
    @staticmethod
    def _ccalc(rowcol1, rowcol2):
        cft1 = np.fft.fft(rowcol1)
        cft2 = np.fft.fft(rowcol2)
        cft2.imag *= -1.0
        corr = np.fft.ifft(cft1 * cft2)
        return corr

    @staticmethod
    def _qc_report(npix, corr, nshift):
        sys.stderr.write("--------------------------------\n")
        if (nshift > 0):
            sys.stderr.write("corr[%d]: %10.5f\n" % (nshift-1, corr[nshift-1]))
        sys.stderr.write("corr[%d]: %10.5f\n" % (nshift+0, corr[nshift+0]))
        if (nshift < npix - 1):
            sys.stderr.write("corr[%d]: %10.5f\n" % (nshift+1, corr[nshift+1]))
        sys.stderr.write("--------------------------------\n")
        return

    # Median-combination of listed overlapping frames:
    @staticmethod
    def dumb_stack(im_list):
        tstack = np.median(im_list, axis=0)
        for im in im_list:
            which = np.isnan(im) | np.isinf(im)
            tstack[which] = np.nan
        return tstack

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (spitz_stacking.py):
#---------------------------------------------------------------------
#
#  2021-02-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created spitz_ccorr_stacking.py.
#

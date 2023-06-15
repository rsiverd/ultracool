#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module that provides cross-matching of Gaia sources.
#
# Rob Siverd
# Created:       2019-09-09
# Last modified: 2023-06-14
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.2"

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
import pandas as pd
import logging
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Time system handling from astropy:
import astropy.time as astt

## Spherical and angular math routines:
import angle
reload(angle)

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

## Home-brew robust statistics:
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)




##--------------------------------------------------------------------------##
##------------------    Reference Epoch Date Conversions    ----------------##
##--------------------------------------------------------------------------##

## From the Gaia DR2 (Lindegren et al. 2018) paper, section 3.1:
## "The astrometric parameters in Gaia DR2 refer to the reference epoch 
## J2015.5 = JD 2457 206.375 (TCB) = 2015 July 2, 21:00:00 (TCB). The
## positions and proper motions refer to the ICRS thanks to the special
## frame alignment procedure (Sect. 5.1)."
_gaia_dr2_epoch = astt.Time(2457206.375, scale='tcb', format='jd')


## The following lookup table provides astropy Time objects that can be used
## to reliably calculate time differences between the test observations and
## Gaia catalog values. We use this to propagate Gaia catalog positions to
## the epoch of the observations before matching.
_gaia_epoch_lookup = {
        2015.5 : gaia_dr2_epoch,
}

##--------------------------------------------------------------------------##
##------------------    Load and Match to Gaia CSV Stars    ----------------##
##--------------------------------------------------------------------------##

class GaiaMatch(object):

    def __init__(self):
        self._srcdata = None
        self._ra_key = 'ra'
        self._de_key = 'dec'
        self._angdev = 180.0
        return

    ### FIXME: consider an alternative to pandas (astropy.table?) ###

    @staticmethod
    def _min_and_idx(a):
        """Return minimum value of array and index position of that value."""
        idx = np.argmin(a)
        return (a[idx], idx)

    def _have_sources(self):
        return True if isinstance(self._srcdata, pd.DataFrame) else False

    def get_gaia_columns(self):
        if not self._have_sources():
            logging.error("No sources loaded. Load data and try again.")
        return self._srcdata.keys()

    def load_sources_csv(self, filename):
        # error if file not present:
        if not os.path.isfile(filename):
            logging.error("file %s not found!" % filename)
            raise IOError
        # load sources from CSV:
        try:
            self._srcdata = pd.read_csv(filename)
        except:
            logging.error("failed to load sources from %s" % filename)
            logging.error("unexpected error: %s" % sys.exc_info()[0])
            raise

        # spread estimate:
        dde = self._srcdata[self._de_key].values
        self._angdev = np.median(np.abs(dde - np.median(dde)))
        return

    def nearest_star_dumb(self, ra, dec):
        """
        Identify the source closest to the given position. RA and DE must
        be given in decimal degrees. No accelerated look-up is performed
        but a match is guaranteed. Matches may not be exclusive.

        Params:
        -------
        ra      -- R.A. in decimal degrees
        dec     -- Dec in decimal degrees
 
        Returns:
        --------
        info    -- record of nearest match from source data
                    (match distance recorded as 'dist' key)
        """
        if not self._have_sources():
            logging.error("No sources loaded. Load data and try again.")

        # Working coordinate arrays:
        sra = self._srcdata[self._ra_key].values
        sde = self._srcdata[self._de_key].values
        sep_deg = angle.dAngSep(ra, dec, sra, sde)
        origidx = np.argmin(sep_deg)      # best match index in subset
        match   = self._srcdata.iloc[[origidx]].copy()
        match['dist'] = sep_deg[origidx]
        return match

    def nearest_star(self, ra, dec, tol_deg):
        #, toler=None):
        """
        Identify the source closest to the given position. RA and DE must
        be given in decimal degrees. 

        Params:
        -------
        ra      -- R.A. in decimal degrees
        dec     -- Dec in decimal degrees
        tol_deg -- maximum matching distance in degrees
 
        Returns dictionary containing:
        ------------------------------
        match   -- True if match found, otherwise False
        record  -- record of nearest match from source data when found
                    (match distance recorded as 'dist' key). If no match,
                    info will contain 'None'
        """
        if not self._have_sources():
            logging.error("No sources loaded. Load data and try again.")
        result = {'match':False, 'record':None}

        # Working coordinate arrays:
        sra = self._srcdata[self._ra_key].values
        sde = self._srcdata[self._de_key].values

        # Initial cut in Dec:
        decnear = (np.abs(sde - dec) <= tol_deg)
        sub_idx = decnear.nonzero()[0]
        if sub_idx.size == 0:   # nothing within tolerance
            return result

        # Full trigonometric calculation:
        sub_ra  = sra[sub_idx]
        sub_de  = sde[sub_idx]
        tru_sep = angle.dAngSep(ra, dec, sub_ra, sub_de)
        sep, ix = self._min_and_idx(tru_sep)
        if (sep > tol_deg):     # best match exceeds tolerance
            return result

        # Select matching record:
        nearest = self._srcdata.iloc[[sub_idx[ix]]].copy()
        nearest['dist'] = sep

        # Return result:
        result['match'] = True
        result['record'] = nearest
        return result

    ### FIXME: implement bulk comparison. For performance reasons, it would
    ### make more sense to identify the indexes of matches and do a single
    ### subset selection from the master catalog (concatenation is slow).

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (gaia_match.py):
#---------------------------------------------------------------------
#
#  2019-09-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created gaia_match.py.
#

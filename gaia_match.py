#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module that provides cross-matching of Gaia sources.
#
# Rob Siverd
# Created:       2019-09-09
# Last modified: 2019-09-11
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
##------------------    Load and Match to Gaia CSV Stars    ----------------##
##--------------------------------------------------------------------------##

class GaiaMatch(object):

    def __init__(self):
        self._srcdata = None
        self._ra_key = 'ra'
        self._de_key = 'dec'
        self._angdev = 180.0
        return

    @staticmethod
    def _min_and_idx(a):
        """Return minimum value of array and index position of that value."""
        idx = np.argmin(a)
        return (a[idx], idx)

    def _have_sources(self):
        return True if isinstance(self._srcdata, pd.DataFrame) else False

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

    def nearest_star(self, ra, dec, toler=None):
        """
        Identify the source closest to the given position. RA and DE must
        be given in decimal degrees. 

        Params:
        -------
        ra      -- R.A. in decimal degrees
        dec     -- Dec in decimal degrees
        toler   -- [optional] maximum matching distance. If not specified,
                        the auto-computed angdev is used for this.
 
        Returns:
        --------
        sep     -- distance in DEGREES to nearest match
        info    -- record of nearest match from source data 
        """
        if not self._have_sources():
            logging.error("No sources loaded. Load data and try again.")

        # Working coordinate arrays:
        sra = self._srcdata[self._ra_key].values
        sde = self._srcdata[self._de_key].values

        ## Initial cut in Dec:
        #use_tol = toler if toler else self._angdev
        #decnear = (np.abs(self._srcdata[self._de_key] - dec) <= use_tol)
        #subset  = self._srcdata[decnear].copy()

        if toler:
            use_tol = toler
            #decnear = (np.abs(self._srcdata[self._de_key] - dec) <= use_tol)
            decnear = (np.abs(sde - dec) <= use_tol)
            wrk_idx = decnear.nonzero()[0]
            #subset  = self._srcdata[decnear].copy()
            #sys.stderr.write("nearidx: %s\n" % str(nearidx))
            sub_ra  = sra[wrk_idx]
            sub_de  = sde[wrk_idx]
            #sys.stderr.write("winner: %d\n" % winner)
        else:
            #subset  = self._srcdata.copy()
            wrk_idx = np.arange(len(sra))
            sub_ra  = sra
            sub_de  = sde

        # Proper angular separation:
        sep_deg = angle.dAngSep(ra, dec, sub_ra, sub_de)
        hit_idx = np.argmin(sep_deg)      # best match index in subset
        origidx = wrk_idx[hit_idx]        # best match index in dataset
        match   = self._srcdata.iloc[[origidx]].copy()
        match['dist'] = sep_deg[hit_idx]
        return match
        ## Full calculation on remaining objects:
        #subset['dist'] = angle.dAngSep(ra, dec,
        #        subset[self._ra_key], subset[self._de_key])
        #return subset.iloc[[subset.dist.values.argmin()]]
        ##return subset[subset.dist == subset.dist.min()]

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (gaia_match.py):
#---------------------------------------------------------------------
#
#  2019-09-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created gaia_match.py.
#

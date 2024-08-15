#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module that provides cross-matching of Gaia sources.
#
# Rob Siverd
# Created:       2019-09-09
# Last modified: 2023-10-30
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.4.0"

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
from functools import reduce
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
        2015.5 : _gaia_dr2_epoch,
}


##--------------------------------------------------------------------------##
##------------------     Expected Gaia CSV Column Names     ----------------##
##--------------------------------------------------------------------------##

## The following items are expected column names from the Gaia CSV data. If
## these are not correct, the matching module will not function correctly.


## Reference epoch for Gaia astrometric solution:
_gaia_epoch_colname = 'ref_epoch'

## Columns containing RA / Dec in decimal degrees:
_gaia_degra_colname = 'ra'
_gaia_degde_colname = 'dec'

## Gaia magnitude columns for reference:
_gaia_bpmag_colname = 'phot_bp_mean_mag'
_gaia_rpmag_colname = 'phot_rp_mean_mag'
_gaia_g_mag_colname = 'phot_g_mean_mag'

## Columns required to be non-NaN for propagation:
_drop_NaNs = ['pmra', 'pmdec', 'phot_rp_mean_mag']
#hits = np.logical_and.reduce([~np.isnan(gsrc[cc]) for cc in drpnan])

##--------------------------------------------------------------------------##
##------------------    Load and Match to Gaia CSV Stars    ----------------##
##--------------------------------------------------------------------------##

class GaiaMatch(object):

    ### FIXME: consider an alternative to pandas (astropy.table?) ###

    def __init__(self):
        #self._srcdata  = None
        self._angdev   = 180.0
        self._ep_col   = _gaia_epoch_colname
        self._ra_key   = _gaia_degra_colname
        self._de_key   = _gaia_degde_colname
        self._reqkeys  = [self._ep_col, self._ra_key, self._de_key]
        #self._ep_user  = None       # the user-supplied epoch
        #self._ep_gaia  = None       # the Gaia catalog epoch
        #self._tdiff_yr = 0.0        # epoch diff (Julian yrs) for PM calc.

        # by default, magnitude limits are broad:
        self._g_mag_lim = 99.0
        self._bpmag_lim = 99.0      # not used
        self._rpmag_lim = 99.0      # not used

        # the list of columns we scour for NaNs:
        self._nancols  = [x for x in _drop_NaNs]

        # reset/initialize Gaia catalog content:
        self._reset_gaia_catalog()

        # reset/initialize user-supplied epoch:
        self._reset_user_epoch()
        return

    def _reset_user_epoch(self):
        self._ep_user  = None       # user-supplied evaluation epoch
        self._tdiff_yr = 0.0        # ep_user - ep_catalog in Julian years

    def _reset_gaia_catalog(self):
        self._rawdata  = None       # Gaia catalog DataFrame
        self._limdata  = None       # Gaia catalog DataFrame with user limits
        self._clndata  = None       # Gaia catalog DataFrame with NaNs purged
        self._srcdata  = None       # Gaia catalog DataFrame at user epoch
        self._ep_gaia  = None       # Gaia catalog reference epoch
        self._origidx  = None       # index of _srcdata objects in _rawdata

    # ---------------------------------------
    # Miscellaneous helper routines      
    # ---------------------------------------

    # reset routine for 

    # handy routine to find minimum and its index:
    @staticmethod
    def _min_and_idx(a):
        """Return minimum value of array and index position of that value."""
        idx = np.argmin(a)
        return (a[idx], idx)

    def _have_sources(self):
        return True if isinstance(self._srcdata, pd.DataFrame) else False

    # The following contains some sanity checks for CSV content. If these
    # checks fail, the module will not work as expected.
    def _srcdata_looks_okay(self):
        if not self._have_sources():
            logging.error("validation failed: no sources")
            return False

        for kk in self._reqkeys:
            if not kk in self._srcdata.keys():
                logging.error("Gaia data is missing key: '%s'" % kk)
                #raise KeyError("Gaia data is missing key: '%s'" % kk)
                raise RuntimeError("Gaia data is missing key: '%s'" % kk)
        return True
 
    # ---------------------------------------
    # User-facing utility routines
    # ---------------------------------------

    def get_gaia_columns(self):
        """Return the columns in the Gaia catalog DataFrame."""
        if not self._have_sources():
            logging.error("No sources loaded. Load data and try again.")
        return self._srcdata.keys()

    # Require certain Gaia columns to be non-NaN:

    # Set an epoch that differs from the catalog reference:
    def set_epoch(self, epoch):
        """Set user-supplied epoch, propagate catalog."""

        # Gaia catalog and epoch must already be loaded:
        if (not self._have_sources()) or (not self._have_gaia_epoch()):
            error_message = "Gaia catalog not yet loaded."
            logging.error(error_message)
            raise RuntimeError(error_message)

        # input must be a Time object:
        if not isinstance(epoch, astt.Time):
            error_message = "set_epoch() expects astropy.Time input!"
            logging.error(error_message)
            raise RuntimeError(error_message)

        # accept user epoch, update time difference:
        self._ep_user  = epoch
        #self._tdiff_yr = (self._ep_gaia.tdb.jd - self._ep_user.tdb.jd) / 365.25
        self._tdiff_yr = (self._ep_user.tdb.jd - self._ep_gaia.tdb.jd) / 365.25
        self._apply_gaia_tdiff()
        return

    # Set limiting (faint) G magnitude for matching:
    def set_Gmag_limit(self, gmag):
        """Only match against sources with Gmag < gmag."""
        self._g_mag_lim = gmag

        # re-apply catalog limits and user epoch:
        self._apply_user_limits()
        return

    # ---------------------------------------
    # Gaia catalog loading and validation
    # ---------------------------------------

    def load_sources_csv(self, filename):
        """Load Gaia catalog data from a CSV file."""

        # error if file not present:
        if not os.path.isfile(filename):
            error_message = "file %s not found!" % filename
            logging.error(error_message)
            raise RuntimeError(error_message)

        # reset/initialize catalog info:
        self._reset_gaia_catalog()
        self._reset_user_epoch()

        # load sources from CSV:
        try:
            self._rawdata = pd.read_csv(filename)
            self._clndata = self._rawdata           # init with ref epoch
            self._srcdata = self._rawdata           # init with ref epoch
        except:
            logging.error("failed to load sources from %s" % filename)
            logging.error("unexpected error: %s" % sys.exc_info()[0])
            raise

        # a quick post-load sanity check:
        if not self._srcdata_looks_okay():
            raise RuntimeError("srcdata validation failed!")

        # update the Gaia epoch after load:
        #self._usedata = self._srcdata
        self._update_gaia_epoch()

        # apply user limits:
        self._apply_user_limits()

        # purge NaNs from identified columns:
        #self._purge_gaia_NaNs()

        # spread estimate:
        dde = self._srcdata[self._de_key].values
        self._angdev = np.median(np.abs(dde - np.median(dde)))
        return

    # Extract Gaia catalog epoch from first entry:
    def _update_gaia_epoch(self):
        first = self._srcdata[self._ep_col].values[0]
        if not first in _gaia_epoch_lookup.keys():
            raise RuntimeError("Unrecognized Gaia epoch: %s" % first)
        self._ep_gaia = _gaia_epoch_lookup.get(first)
        return

    # whether or not Gaia epoch is set:
    def _have_gaia_epoch(self):
        return isinstance(self._ep_gaia, astt.Time)

    # impose limits on Gaia catalog (e.g., gmag, RA, Dec):
    def _apply_user_limits(self):
        # first, axe sources with NaNs in important columns:
        conditions = [~np.isnan(self._rawdata[cc]) for cc in self._nancols]
        # Gmag threshold:
        gmag_okay = (self._rawdata[_gaia_g_mag_colname] <= self._g_mag_lim)
        conditions.append(gmag_okay)

        # ADD MORE CONDITIONS HERE

        # joint ANDing of conditions:
        keep_gaia = reduce(lambda x,y: x&y, conditions)
        self._clndata = self._rawdata[keep_gaia]

        # Redo user epoch adjustment:
        self._apply_gaia_tdiff()
        return

    # drop items from an array if NaNs are found in the specified column:
    #def _purge_gaia_NaNs(self):
    #    keep_cols = [~np.isnan(self._rawdata[cc]) for cc in self._nancols]
    #    keep_gaia = reduce(lambda x,y: x&y, keep_cols)
    #    self._clndata = self._rawdata[keep_gaia]
    #    return

    # compute updated Gaia positions for comparison using
    def _apply_gaia_tdiff(self):
        #has_pmra = ~np.isnan(self._rawdata['pmra'])
        #has_pmde = ~np.isnan(self._rawdata['pmdec'])
        #self._origidx = (has_pmra & has_pmde)
        #tmp_gaia = self._rawdata[self._origidx].copy()
        tmp_gaia = self._clndata.copy()
        cos_dec  = np.cos(np.radians(tmp_gaia['dec'])).values
        dde_adjustment_mas = self._tdiff_yr * tmp_gaia['pmdec']
        dra_adjustment_mas = self._tdiff_yr * tmp_gaia['pmra'] / cos_dec
        #sys.stderr.write("dde_adjustment_mas.max(): %f\n"
        #        % dde_adjustment_mas.max())
        #sys.stderr.write("dra_adjustment_mas.max(): %f\n"
        #        % dra_adjustment_mas.max())
        #sys.stderr.write("median dde adjustment: %f\n"
        #        % np.median(dde_adjustment_mas))
        #sys.stderr.write("median dra adjustment: %f\n"
        #        % np.median(dra_adjustment_mas))
        tmp_gaia['ra']  = tmp_gaia['ra']  + (dra_adjustment_mas / 3.6e6)
        tmp_gaia['dec'] = tmp_gaia['dec'] + (dde_adjustment_mas / 3.6e6)
        self._srcdata   = tmp_gaia  # store for cross-match
        #self._srcdata   = tmp_gaia.reset_index()  # store for cross-match
        return

    # ---------------------------------------
    # Cross-matching routines
    # ---------------------------------------

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

    # multi-dimensional cross-match for single source:
    def _similar_star(self, ra, dec, tol_deg, 
            gaia_subset=None, constraints=[]):
        #, toler=None):
        """
        Identify a source closest to the given position, subject to the
        positional tolerance and other constraints. Test RA and DE must
        be given in decimal degrees. 

        Params:
        -------
        ra          -- R.A. in decimal degrees
        dec         -- Dec in decimal degrees
        tol_deg     -- maximum matching distance in degrees
        gaia_subset -- optional (cleaned) subset of Gaia catalog for matching.
                            This subset should have NaNs removed.
        constraints -- optional non-positional matching constraints. This is a
                            list of tuples containing the following:
                            #(star_col, gaia_col, min_diff, max_diff)
                            (gaia_col, min_val, max_val)

        Returns dictionary containing:
        ------------------------------
        match   -- True if match found, otherwise False
        record  -- record of nearest match from source data when found
                    (match distance recorded as 'dist' key). If no match,
                    info will contain 'None'
        """
        #if not self._have_sources():
        #    logging.error("No sources loaded. Load data and try again.")
        result = {'match':False, 'record':None}

        # Working coordinate arrays:
        sra = self._srcdata[self._ra_key].values
        sde = self._srcdata[self._de_key].values

        # Initial cut in Dec:
        decnear = (np.abs(sde - dec) <= tol_deg)
        sub_idx = decnear.nonzero()[0]
        if sub_idx.size == 0:   # nothing within tolerance
            return result
        #import pdb; pdb.set_trace()
        remain  = self._srcdata.iloc[sub_idx]
        #import pdb; pdb.set_trace()

        # Apply additional constraints:
        for gcol,lower,upper in constraints:
            which  = (lower <= remain[gcol]) & (remain[gcol] <= upper)
            remain = remain[which]

        # Full trigonometric calculation:
        sub_ra  = remain[self._ra_key].values
        sub_de  = remain[self._de_key].values
        #sub_ra  = sra[sub_idx]
        #sub_de  = sde[sub_idx]
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

    # Batch-mode matching. The following routine accepts a numpy record array
    # (or equivalent) as input and identifies likely matches from the Gaia
    # catalog using multiple criteria. Positional matching is always used
    # but other criteria can be optionally included. Cross-matching requires
    # the user to provide column names of specific items within the input
    # record array.
    def multipar_gaia_matches(self, stars, ra_col, de_col, tol_arcsec,
            constraints=[]):
        """
        Perform multidimensional matching of an input list to the currently
        loaded Gaia sources. Positional matching is required; other parameters
        are optional.

        Inputs:
        -------
        stars       --  numpy record array (or similar) with sources to match
        ra_col      --  column name in 'stars' with RA in decimal degrees
        de_col      --  column name in 'stars' with DE in decimal degrees
        tol_arcsec  --  positional matching tolerance in arcseconds
        constraints --  list of non-positional constraints for Gaia matches

        Constraints:
        ------------
        The constraints input requires some care in order to get the desired
        results. This input is a list of tuples, where each tuple describes
        a specific constraint. Each constraint tuple consists of:
            --> (stars_colname, gaia_colname, lower, upper)

        When applied, Gaia sources are accepted if (NOTE THE SIGN):
                lower <= (gaia_vals - stars_val) <= upper,
        where:
                stars_val = stars[stars_colname] for that detection
                gaia_vals = gaia_data[gaia_colname]

        The ordering of the differences (gaia - stars) was chosen to make
        the most sense when using a magnitude cutoff. The convention for
        colors from magnitude differences is [bluer] - [redder].


        Returns:
        --------
        idxarray    --  array containing input indexes of matched sources
        gaia_data   --  Gaia record data corresponding to matched sources
        """

        # abort if Gaia catalog is not primed:
        if not self._have_sources():
            logging.error("No sources loaded. Load data and try again.")

        tol_deg = tol_arcsec / 3600.0
        #matches = []
        match_idx = []
        match_rec = []
        for ix,target in enumerate(stars):
            sra, sde = target[ra_col], target[de_col]
            #sxx, syy = target[xx_col], target[yy_col]
            npcon = []
            for scol,gcol,lo,hi in constraints:
                npcon.append((gcol, target[scol]+lo, target[scol]+hi))

            result = self._similar_star(sra, sde, tol_deg, constraints=npcon)
            if result['match']:
                #gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
                #matches.append((sxx, syy, sra, sde, *gcoords))
                #matches.append((ix, result['record']))
                match_idx.append(ix)
                match_rec.append(result['record'])
                pass
            pass
        return np.array(match_idx), pd.concat(match_rec)
        #return matches
        # pre-compute a working subset of the Gaia catalog:
        #user_dec_hi = stars[de_col].max() + 10.0 * tol_deg
        #user_dec_lo = stars[de_col].min() - 10.0 * tol_deg
        #nearby_bidx = (user_dec_lo <= stars[de_col]) \
        #            & (stars[de_col] <= user_dec_hi)
        #gaia_subset = self._srcdata[nearby_bidx]
        #return

    # TIMING DATA:
    # %timeit test_matches = gm.multipar_gaia_matches(stars, 
    #           ra_col='dra', de_col='dde', tol_arcsec=match_tol_arcsec)
    # 2.19 s ± 15.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    #def find_gaia_matches(stars, tol_arcsec, ra_col='dra', de_col='dde',
    #        xx_col='x', yy_col='y'):
    #    tol_deg = tol_arcsec / 3600.0
    #    matches = []
    #    for target in stars:
    #        sra, sde = target[ra_col], target[de_col]
    #        sxx, syy = target[xx_col], target[yy_col]
    #        result = gm.nearest_star(sra, sde, tol_deg)
    #        if result['match']:
    #            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
    #            matches.append((sxx, syy, sra, sde, *gcoords))
    #            pass
    #        pass
    #    return matches
    # TIMING INFO:
    # %timeit gaia_matches = find_gaia_matches(stars, match_tol_arcsec,
    #                                           xx_col='xrel', yy_col='yrel')
    # 924 ms ± 3.33 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    # ---------------------------------------
    # Batch (multi-object) matching routines
    # ---------------------------------------

    def twoway_gaia_matches(self, trial_ra, trial_de, tol_arcsec, debug=False):
        """
        Match a list of RA, DE positions against the Gaia catalog. This
        routine relies on the single-target matching method nearest_star.
        Matching is performed both forwards and backwards to eliminate
        duplicate matches as follows:
        1) First, we iterate over the the input RA/DE arrays and look for
        matching Gaia sources. The Gaia star nearest the trial position
        (within the specified tolerance) is taken to be the match. For each
        match, we set aside matching elements from both the input coordinates
        and the Gaia catalog.
        2) We deduplicate and concatenate the matched input coordinates and
        Gaia sources from step 1. The resulting input RA/DE arrays are the
        coordinates of all trial sources that MAY have matches in Gaia. The
        Gaia subset is deduplicated and contains the set of Gaia sources 
        to which input sources may match.
        3) We iterate over the deduplicated Gaia subset and look for matches
        from among the input array. The nearest match within the tolerance
        is taken to be real and set aside.

        This procedure eliminates the possibility of the same Gaia star
        matching to multiple trial positions. Similarly, no trial position
        can match to multiple Gaia sources. This does not guarantee that
        matches are correct, only that duplicates will not be present.

        Results are ordered by *input* coordinate index.

        The case of no matches results in zero-length arrays returned.

        Params:
        -------
        trial_ra    -- numpy array of R.A. values in decimal degrees
        trial_de    -- numpy array of Dec values in decimal degrees
        tol_arcsec  -- maximum matching distance in arcseconds
 
        Returns four arrays containing:
        -------------------------------
        idx         -- index of match in the input RA/DE arrays
        gaia_ra     -- Gaia RA of matched source
        gaia_dec    -- Gaia DE of matched source
        source_id   -- Gaia source_id of matched source (for record lookup)
        """

        tol_deg  = tol_arcsec / 3600.0
        star_ids = []       #  array indices of possible matches
        gaia_ids = []       # Gaia source_id of possible matches

        # iterate over input RA/DE and look for matches:
        for idx,(sra, sde) in enumerate(zip(trial_ra, trial_de)):
            result = self.nearest_star(sra, sde, tol_deg)
            if result['match']:
                gaia_ids.append(int(result['record']['source_id']))
                star_ids.append(idx)
                pass
            pass

        if debug:
            sys.stderr.write("After trial -> Gaia matching, have:\n")
            sys.stderr.write("--> len(star_ids) = %d\n" % len(star_ids)) 
            sys.stderr.write("--> len(gaia_ids) = %d\n" % len(gaia_ids)) 

        # Make deduplicated subset of possibly-matching Gaia sources:
        use_gaia = self._srcdata[self._srcdata.source_id.isin(gaia_ids)]

        # Make subset of possibly-matching input coordinates:
        trial_ra_subset = trial_ra[star_ids]
        trial_de_subset = trial_de[star_ids]

        # Iterate over Gaia possibles and select matches from trial data:
        matches = []
        for gi,(gix, gsrc) in enumerate(use_gaia.iterrows(), 1):
            sep_deg = angle.dAngSep(gsrc.ra, gsrc.dec,
                                        trial_ra_subset, trial_de_subset)
            midx = sep_deg.argmin()     # index of match in SUBSET
            sidx = star_ids[midx]       # index of match in trial arrays
            matches.append((sidx, gsrc.ra, gsrc.dec, gsrc.source_id))
            pass

        # Handle case of no matches:
        if not matches:
            #sys.stderr.write("NO MATCHES!\n")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Re-sort results according to trial array:
        idx, gra, gde, gid = zip(*matches)
        iorder = np.argsort(idx)
        return (np.array(idx)[iorder],
                np.array(gra)[iorder],
                np.array(gde)[iorder],
                np.array(gid)[iorder])

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (gaia_match.py):
#---------------------------------------------------------------------
#
#  2019-09-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created gaia_match.py.
#

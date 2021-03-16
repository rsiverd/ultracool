#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# First cut at astrometry fitting for UCD project.
#
# Rob Siverd
# Created:       2020-02-07
# Last modified: 2021-03-16
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

## Modules:
import os
import sys
import time
import numpy as np
from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import theil_sen as ts

## Useful stats routines:
def calc_ls_med_MAD(a, axis=None):
    """Return median and median absolute deviation of *a* (scaled to normal)."""
    med_val = np.median(a, axis=axis)
    sig_hat = (1.482602218 * np.median(np.abs(a - med_val), axis=axis))
    return (med_val, sig_hat)

##--------------------------------------------------------------------------##
##------------------       Astrometry Fitting (5-par)       ----------------##
##--------------------------------------------------------------------------##

_ARCSEC_PER_RADIAN = 180. * 3600.0 / np.pi
_MAS_PER_RADIAN = _ARCSEC_PER_RADIAN * 1e3
class AstFit(object):

    _need_eph_keys = ['jdtdb', 'x', 'y', 'z']
    _asec_per_rad  = _ARCSEC_PER_RADIAN
    _mas_per_rad   = _MAS_PER_RADIAN

    def __init__(self):
        self._jd_tdb = None
        self._dt_yrs = None
        self.obs_eph = None
        self.ref_tdb = None
        self.inliers = None
        self.rweight = None
        self._is_set = False
        self._chiexp = 2
        return

    def set_exponent(self, exponent=2):
        """
        Choose exponent used in penalty function (N below). The solver seeks
        to minimize the sum over data points of:
        ((obs - model) / err)**N
        """
        #Setting N=2 behaves like Chi-squared. Setting N=1 minimizes total
        #absolute deviation
        self._chiexp = exponent
        return

    def setup(self, jd_tdb_ref, RA_deg, DE_deg, obs_eph, 
            RA_err=None, DE_err=None):
        self._is_rdy = False
        if not all([isinstance(obs_eph[x], np.ndarray) \
                for x in self._need_eph_keys]):
            sys.stderr.write("Incomplete ephemeris data!\n") 
            sys.stderr.write("Required columns include:\n")
            sys.stderr.write("--> %s\n" % str(self._need_eph_keys))
            return False
        #if (len(jd_tdb_vec) != len(obs_eph)):
        #    sys.stderr.write("Error: mismatched input dimensions!\n")
        #    sys.stderr.write("len(jd_tdb_vec): %d\n" % len(jd_tdb_vec))
        #    sys.stderr.write("len(obs_eph)     %d\n" % len(obs_eph))
        #    return False
        self.inliers = np.ones_like(RA_deg, dtype='bool')
        self.rweight = np.ones_like(RA_deg)
        self.obs_eph = self._augmented_eph(obs_eph)
        self.ref_tdb = jd_tdb_ref
        self._dt_yrs = (self.obs_eph['jdtdb'] - self.ref_tdb) / 365.25
        self._RA_rad = np.radians(RA_deg)
        self._DE_rad = np.radians(DE_deg)
        self._RA_med, self._RA_MAD = calc_ls_med_MAD(self._RA_rad)
        self._DE_med, self._DE_MAD = calc_ls_med_MAD(self._DE_rad)
        #self._RA_MAD *= np.cos(self._DE_med)
        if isinstance(RA_err, np.ndarray):
            self._RA_err = np.radians(RA_err)
        else:
            self._RA_err = self._RA_MAD
        if isinstance(DE_err, np.ndarray):
            self._DE_err = np.radians(DE_err)
        else:
            self._DE_err = self._DE_MAD
        #self._DE_err = np.radians(DE_err) if DE_err else self._DE_MAD
        self._is_set = True
        return True

    @staticmethod
    def _augmented_eph(obs_eph):
        twopi = 2.0 * np.pi
        anom = np.arctan2(obs_eph['y'], obs_eph['x']) % twopi
        return append_fields(obs_eph, 'anom', anom, usemask=False)


    #def set_ref_time(self, t_ref):
    #    self.ref_time = t_ref
    #    return

    @staticmethod
    def _calc_parallax_factors(RA_rad, DE_rad, X_au, Y_au, Z_au):
        sinRA, cosRA = np.sin(RA_rad), np.cos(RA_rad)
        sinDE, cosDE = np.sin(DE_rad), np.cos(DE_rad)
        ra_factor = (X_au * sinRA - Y_au * cosRA) / cosDE
        de_factor =  X_au * cosRA * sinDE \
                  +  Y_au * sinRA * sinDE \
                  -  Z_au * cosDE
        return ra_factor, de_factor

    #def ts_fit_coord(self, time_vals, coo_vals):
    @staticmethod
    def ts_fit_radec_pm(t_yrs, RA_rad, DE_rad, plx_as=0, weighted=False):
        ts_ra_model = ts.linefit(t_yrs, RA_rad, weighted=weighted)
        ts_de_model = ts.linefit(t_yrs, DE_rad, weighted=weighted)
        return np.array([ts_ra_model[0], ts_de_model[0],
                ts_ra_model[1], ts_de_model[1], plx_as])

    def apparent_radec(self, t_ref, astrom_pars, eph_obs):
        """
        t_ref       --  chosen reference epoch
        astrom_pars --  five astrometric parameters specified at the
                        reference epoch: meanRA (rad), meanDE (rad),
                        pmRA*cos(DE), pmDE, and parallax
        eph_obs     --  dict with x,y,z,t elements describing the times
                        and places of observations (numpy arrays)
        FOR NOW, assume
                    [t_ref] = JD (TDB)
                    [t]     = JD (TDB)
                    [pars]  = rad, rad, arcsec/yr, arcsec/yr, arcsec
                                       *no cos(d)*
        """
    
        rra, rde, pmra, pmde, prlx = astrom_pars
    
        t_diff_yr = (eph_obs['t'] - t_ref) / 365.25     # units of years
    
        pfra, pfde = self._calc_parallax_factors(rra, rde,
                eph_obs['x'], eph_obs['y'], eph_obs['z'])
    
        #delta_ra = (t_diff_yr * pmra / _ARCSEC_PER_RADIAN) + (prlx * pfra)
        #delta_de = (t_diff_yr * pmde / _ARCSEC_PER_RADIAN) + (prlx * pfde)
        #delta_ra = (t_diff_yr * pmra + prlx * pfra) / _ARCSEC_PER_RADIAN
        #delta_de = (t_diff_yr * pmde + prlx * pfde) / _ARCSEC_PER_RADIAN
        #delta_ra = (t_diff_yr * pmra + prlx * pfra) / _MAS_PER_RADIAN
        #delta_de = (t_diff_yr * pmde + prlx * pfde) / _MAS_PER_RADIAN
        delta_ra = (t_diff_yr * pmra + prlx * pfra)
        delta_de = (t_diff_yr * pmde + prlx * pfde)
    
        return (rra + delta_ra, rde + delta_de)

    def eval_model(self, params):
        rra, rde, pmra, pmde, prlx = params
        pfra, pfde = self._calc_parallax_factors(rra, rde,
                self.obs_eph['x'], self.obs_eph['y'], self.obs_eph['z'])
        delta_ra = self._dt_yrs * pmra + prlx * pfra
        delta_de = self._dt_yrs * pmde + prlx * pfde
        return (rra + delta_ra, rde + delta_de)

    def _solver_eval(self, params):
        rra, rde, pmra, pmde, prlx = params
        pfra, pfde = self._calc_parallax_factors(rra, rde,
                self.obs_eph['x'], self.obs_eph['y'], self.obs_eph['z'])
        delta_ra = self._dt_yrs * pmra + prlx * pfra
        delta_de = self._dt_yrs * pmde + prlx * pfde
        return (rra + delta_ra, rde + delta_de)

    def _calc_radec_residuals(self, params):
        model_RA, model_DE = self._solver_eval(params)
        return (self._RA_rad - model_RA, self._DE_rad - model_DE)

    def _calc_radec_residuals_sigma(self, params):
        model_RA, model_DE = self._solver_eval(params)
        rsigs_RA = (self._RA_rad - model_RA) / self._RA_err
        rsigs_DE = (self._DE_rad - model_DE) / self._DE_err
        return rsigs_RA, rsigs_DE

    def _calc_total_residuals_sigma(self, params):
        return np.hypot(*self._calc_radec_residuals_sigma(params))

    def _calc_chi_square(self, params, negplxhit=100.):
        model_ra, model_de = self._solver_eval(params)
        #resid_ra = (model_ra - self._RA_rad) #/ np.cos(model_de)
        #resid_de = (model_de - self._DE_rad)
        resid_ra = (self._RA_rad - model_ra) #/ np.cos(model_de)
        resid_de = (self._DE_rad - model_de)
        #resid_ra = (model_ra - self._RA_rad) / self._RA_err
        #resid_de = (model_de - self._DE_rad) / self._DE_err
        if isinstance(self._RA_err, np.ndarray):
            resid_ra /= self._RA_err
        if isinstance(self._DE_err, np.ndarray):
            resid_de /= self._DE_err
        #return np.sum(np.hypot(resid_ra, resid_de))
        #return np.sum(np.hypot(resid_ra, resid_de)**2)
        resid_tot = np.hypot(resid_ra, resid_de)[self.inliers]
        if (params[4] < 0.0):
            resid_tot *= negplxhit
        return np.sum(resid_tot**self._chiexp)
        #return np.sum(np.hypot(resid_ra, resid_de)**self._chiexp)
        #return np.sum(np.abs(resid_ra * resid_de)**self._chiexp)

    # Driver routine for 5-parameter astrometric fitting:
    def fit_bestpars(self, sigcut=5):
        if not self._is_set:
            sys.stderr.write("Error: data not OK for fitting!\n")
            sys.stderr.write("Run setup() first and retry ...\n")
            return False

        # robust initial guess with Theil-Sen:
        uguess = self.ts_fit_radec_pm(self._dt_yrs, self._RA_rad, self._DE_rad)
        wguess = self.ts_fit_radec_pm(self._dt_yrs, self._RA_rad, self._DE_rad,
                weighted=True)
        #sys.stderr.write("Initial guess: %s\n" % str(guess))
        sys.stderr.write("Initial guess (unweighted):\n")
        sys.stderr.write("==> %s\n" % str(self.nice_units(uguess)))
        sys.stderr.write("\n")
        sys.stderr.write("Initial guess (weighted):\n")
        sys.stderr.write("==> %s\n" % str(self.nice_units(wguess)))
        sys.stderr.write("\n")
        guess = uguess  # adopt unweighted for now

        # check whether anything looks really bad:
        self._par_guess = guess
        #rsig_tot = np.hypot(*self._calc_radec_residuals_sigma(guess))
        rsig_tot = self._calc_total_residuals_sigma(guess)
        #sys.stderr.write("rsig_tot: %s\n" % str(rsig_tot))
        self.inliers = (rsig_tot < sigcut)
        ndropped = self.inliers.size - np.sum(self.inliers)
        sys.stderr.write("Dropped %d point(s) beyond %.2f-sigma.\n"
                % (ndropped, sigcut))
        #sys.stderr.write("ra_res: %s\n" % str(ra_res))
        #sys.stderr.write("de_res: %s\n" % str(de_res))
        #sys.stderr.write("ra_sig: %s\n" % str(ra_sig))
        #sys.stderr.write("de_sig: %s\n" % str(de_sig))
 

        # find minimum:
        self.result = opti.fmin(self._calc_chi_square, guess, 
                xtol=1e-9, ftol=1e-9, full_output=True)

        sys.stderr.write("Found minimum:\n")
        sys.stderr.write("==> %s\n" % str(self.nice_units(self.result[0])))
        return self.result[0]

    def nice_units(self, params):
        result = np.degrees(params)
        result[2:5] *= 3.6e6                # into milliarcsec
        result[2] *= np.cos(params[1])      # cos(dec) for pmRA
        return result

    def list_resid_sigmas(self, params):
        rsig_RA, rsig_DE = self._calc_radec_residuals_sigma(params)
        rsig_tot = np.hypot(rsig_RA, rsig_DE)
        #sys.stderr.write("%15s %15s\n")
        for ii,point in enumerate(zip(rsig_RA, rsig_DE, rsig_tot), 0):
            sys.stderr.write("> %10.5f %10.5f (%10.5f)\n" % point)
        return


######################################################################
# CHANGELOG (astrom_test.py):
#---------------------------------------------------------------------
#
#  2020-02-07:
#     -- Increased __version__ to 0.1.0.
#     -- First created astrom_test.py.
#

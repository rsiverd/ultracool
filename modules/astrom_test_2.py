#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Second cut at astrometry fitting for UCD project.
#
# Rob Siverd
# Created:       2021-08-30
# Last modified: 2021-08-30
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

## Median absolute residual:
def calc_MAR(residuals, scalefactor=1.482602218):
    """Return median absolute residual (MAR) of input array. By default,
    the result is scaled to the normal distribution."""
    return scalefactor * np.median(np.abs(residuals))

##--------------------------------------------------------------------------##
##------------------       Astrometry Fitting (5-par)       ----------------##
##--------------------------------------------------------------------------##

_ARCSEC_PER_RADIAN = 180. * 3600.0 / np.pi
_MAS_PER_RADIAN = _ARCSEC_PER_RADIAN * 1e3
class AstFit(object):

    """
    This module provides astrometric fitting capability. Internally, a
    5-parameter model is maintained in a numpy array. Its contents are:
        * RA (radians) at reference epoch
        * DE (radians) at reference epoch
        * pmRA (radians / yr). [this is pmRA* / cos(dec)]
        * pmDE (radians / yr)
        * parallax (radians)
    """

    _need_eph_keys = ['jdtdb', 'x', 'y', 'z']
    _need_data_keys = ['jdtdb', 'dra', 'dde', 'obs_x', 'obs_y', 'obs_z']
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
        self._can_iterate = False
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

    #def setup(self, jd_tdb_ref, RA_deg, DE_deg, obs_eph, 
    def setup(self, data, reject_outliers=True,
            jd_tdb_ref=None, RA_err=None, DE_err=None):
        self._is_rdy = False
        if not all([isinstance(data[x], np.ndarray) \
                for x in self._need_data_keys]):
            sys.stderr.write("Incomplete data set!\n") 
            sys.stderr.write("Required columns include:\n")
            sys.stderr.write("--> %s\n" % str(self._need_data_keys))
            return False
        self._outrej = reject_outliers
        #if not all([isinstance(obs_eph[x], np.ndarray) \
        #        for x in self._need_eph_keys]):
        #    sys.stderr.write("Incomplete ephemeris data!\n") 
        #    sys.stderr.write("Required columns include:\n")
        #    sys.stderr.write("--> %s\n" % str(self._need_eph_keys))
        #    return False
        
        #self.inliers = np.ones_like(RA_deg, dtype='bool')
        #self.rweight = np.ones_like(RA_deg)
        self.inliers = np.ones(len(data), dtype='bool')
        self.rweight = np.ones(len(data), dtype='float')
        #self.obs_eph = self._augmented_eph(obs_eph)
        self.dataset = np.copy(data)
        if jd_tdb_ref:
            self.ref_tdb = jd_tdb_ref
        else:
            self.ref_tdb = data['jdtdb'][0]
        #self.ref_tdb = jd_tdb_ref
        
        self._dt_yrs = (self.dataset['jdtdb'] - self.ref_tdb) / 365.25
        #self._RA_rad = np.radians(RA_deg)
        #self._DE_rad = np.radians(DE_deg)
        self._RA_rad = np.radians(self.dataset['dra'])
        self._DE_rad = np.radians(self.dataset['dde'])
        #self._RA_med, self._RA_MAD = calc_ls_med_MAD(self._RA_rad)
        #self._DE_med, self._DE_MAD = calc_ls_med_MAD(self._DE_rad)
        #self._RA_MAD *= np.cos(self._DE_med)

        self._RA_err = RA_err
        self._DE_err = DE_err
        self._need_resid_errors = False
        if not isinstance(RA_err, np.ndarray):
            sys.stderr.write("WARNING: RA_err not given, using estimated\n")
            self._need_resid_errors = True
        if not isinstance(DE_err, np.ndarray):
            sys.stderr.write("WARNING: DE_err not given, using estimated\n")
            self._need_resid_errors = True
        
        #if isinstance(RA_err, np.ndarray):
        #    self._RA_err = np.radians(RA_err)
        #else:
        #    self._RA_err = self._RA_MAD
        #if isinstance(DE_err, np.ndarray):
        #    self._DE_err = np.radians(DE_err)
        #else:
        #    self._DE_err = self._DE_MAD
        #self._DE_err = np.radians(DE_err) if DE_err else self._DE_MAD
        self._is_set = True
        self._can_iterate = False
        return True

    #def set_ref_time(self, t_ref):
    #    self.ref_time = t_ref
    #    return

    @staticmethod
    def _calc_parallax_factors(RA_rad, DE_rad, X_au, Y_au, Z_au):
        """Compute parallax factors in arcseconds. The RA component has 
        been divided by cos(dec) so that it can be used directly for
        residual minimization."""
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
    
        delta_ra = (t_diff_yr * pmra + prlx * pfra)
        delta_de = (t_diff_yr * pmde + prlx * pfde)
    
        return (rra + delta_ra, rde + delta_de)

    def eval_model(self, params):
        return self._solver_eval(params)
    #def eval_model(self, params):
    #    rra, rde, pmra, pmde, prlx = params
    #    pfra, pfde = self._calc_parallax_factors(rra, rde,
    #            self.dataset['obs_x'], self.dataset['obs_y'],
    #            self.dataset['obs_z'])
    #    delta_ra = self._dt_yrs * pmra + prlx * pfra
    #    delta_de = self._dt_yrs * pmde + prlx * pfde
    #    return (rra + delta_ra, rde + delta_de)

    def _solver_eval(self, params):
        rra, rde, pmra, pmde, prlx = params
        pfra, pfde = self._calc_parallax_factors(rra, rde,
                self.dataset['obs_x'], self.dataset['obs_y'],
                self.dataset['obs_z'])
        delta_ra = self._dt_yrs * pmra + prlx * pfra
        delta_de = self._dt_yrs * pmde + prlx * pfde
        #delta_ra = self._dt_yrs * pmra - prlx * pfra
        #delta_de = self._dt_yrs * pmde - prlx * pfde
        return (rra + delta_ra, rde + delta_de)

    def _calc_radec_residuals(self, params):
        model_RA, model_DE = self._solver_eval(params)
        return (self._RA_rad - model_RA, self._DE_rad - model_DE)

    def _calc_radec_residuals_sigma(self, params):
        model_RA, model_DE = self._solver_eval(params)
        #rsigs_RA = (self._RA_rad - model_RA) / self._RA_err
        #rsigs_DE = (self._DE_rad - model_DE) / self._DE_err
        rsigs_RA = (self._RA_rad - model_RA) / self._use_RA_err
        rsigs_DE = (self._DE_rad - model_DE) / self._use_DE_err
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
        #if isinstance(self._RA_err, np.ndarray):
        #    resid_ra /= self._RA_err
        #if isinstance(self._DE_err, np.ndarray):
        #    resid_de /= self._DE_err
        if isinstance(self._use_RA_err, np.ndarray):
            resid_ra /= self._use_RA_err
        if isinstance(self._use_DE_err, np.ndarray):
            resid_de /= self._use_DE_err
        #return np.sum(np.hypot(resid_ra, resid_de))
        #return np.sum(np.hypot(resid_ra, resid_de)**2)
        resid_tot = np.hypot(resid_ra, resid_de)[self.inliers]
        if (params[4] < 0.0):
            resid_tot *= negplxhit
        return np.sum(resid_tot**self._chiexp)
        #return np.sum(np.hypot(resid_ra, resid_de)**self._chiexp)
        #return np.sum(np.abs(resid_ra * resid_de)**self._chiexp)

    def _calc_initial_parallax(self, params):
        rra_resid, rde_resid = self._calc_radec_residuals(params)
        mar_ra_rad = calc_MAR(rra_resid)
        mar_ra_mas = _MAS_PER_RADIAN * mar_ra_rad
        sys.stderr.write("mar_ra_rad: %f\n" % mar_ra_rad)
        sys.stderr.write("mar_ra_mas: %f\n" % mar_ra_mas)
        pfra, pfde = self._calc_parallax_factors(
                self._RA_rad, self._DE_rad, self.dataset['obs_x'],
                self.dataset['obs_y'], self.dataset['obs_z'])
        #sys.stderr.write("pfra_arcsec: %s\n" % str(pfra_arcsec))
        #pfra_rad   = pfra_arcsec / _ARCSEC_PER_RADIAN
        adjustment_arcsec = ts.linefit(pfra, _ARCSEC_PER_RADIAN * rra_resid)
        sys.stderr.write("adjustment (arcsec): %s\n" % str(adjustment_arcsec))
        return adjustment_arcsec

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
        #guess[4] = 1000. / _MAS_PER_RADIAN

        # initial crack at parallax and zero-point:
        woohoo = self._calc_initial_parallax(guess)
        sys.stderr.write("woohoo: %s\n" % str(woohoo))
        self.woohoo = woohoo
        ra_nudge_rad, plx_rad = woohoo / _ARCSEC_PER_RADIAN
        guess[0] += ra_nudge_rad
        guess[4] = plx_rad

        # estimate RA,Dec uncertainty from residuals if not known a prior:
        if self._need_resid_errors:
            rra_resid, rde_resid = self._calc_radec_residuals(guess)
            rra_scatter = calc_MAR(rra_resid)
            rde_scatter = calc_MAR(rde_resid)
            mra_scatter = _MAS_PER_RADIAN * rra_scatter
            mde_scatter = _MAS_PER_RADIAN * rde_scatter
            #sys.stderr.write("rra_resid: %s\n" % str(rra_resid))
            #sys.stderr.write("rde_resid: %s\n" % str(rde_resid))
            sys.stderr.write("rra_scatter: %e (rad)\n" % rra_scatter)
            sys.stderr.write("rde_scatter: %e (rad)\n" % rde_scatter)
            sys.stderr.write("mra_scatter: %10.5f (mas)\n" % mra_scatter)
            sys.stderr.write("mde_scatter: %10.5f (mas)\n" % mde_scatter)
            self._RA_err = np.ones_like(self._RA_rad) * rra_scatter
            self._DE_err = np.ones_like(self._DE_rad) * rde_scatter
        self._use_RA_err = np.copy(self._RA_err)
        self._use_DE_err = np.copy(self._DE_err)

        # check whether anything looks really bad:
        self._par_guess = guess
        #rsig_tot = np.hypot(*self._calc_radec_residuals_sigma(guess))
        rsig_tot = self._calc_total_residuals_sigma(guess)
        #sys.stderr.write("rsig_tot:\n")
        #sys.stderr.write("%s\n" % str(rsig_tot))
        sys.stderr.write("typical rsig_tot: %8.3f\n" % np.median(rsig_tot))
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
        self.full_result = opti.fmin(self._calc_chi_square, guess, 
                xtol=1e-7, ftol=1e-7, full_output=True)
                #xtol=1e-9, ftol=1e-9, full_output=True)
        self.result = self.full_result[0]

        # brute-force minimum:
        #ra_fudge = np.median(self._RA_err)
        #de_fudge = np.median(self._DE_err)
        #pm_fudge = 0.2
        #px_fudge = 4.0
        #ranges = [(guess[0] - ra_fudge, guess[0] + ra_fudge),   # RA
        #          (guess[1] - de_fudge, guess[1] + de_fudge),   # DE
        #          (guess[2] / pm_fudge, guess[2] * pm_fudge),   # pmRA
        #          (guess[3] / pm_fudge, guess[3] * pm_fudge),   # pmRA
        #          (guess[4] / px_fudge, guess[3] * px_fudge),   # parallax
        #          ]
        #npts = 10
        #self.result = opti.brute(self._calc_chi_square, ranges, Ns=npts)

        sys.stderr.write("Found minimum:\n")
        sys.stderr.write("==> %s\n" % str(self.nice_units(self.result)))
        self._can_iterate = True
        return self.result
 
    # ----------------------------------------------------------------------- 
    def _calc_huber_rweights(self, residuals, sigma):
        _k_sig = 1.34 * sigma
        res_devs = np.abs(residuals / _k_sig)
        rweights = np.ones_like(res_devs)
        distants = (res_devs > 1.0)
        rweights[distants] = 1.0 / res_devs[distants]
        return rweights

    def iter_update_bestpars(self, params):
        """Perform an IRLS iteration."""

        # calculate residuals:
        rra_resid, rde_resid = self._calc_radec_residuals(params)
        #sys.stderr.write("rra_resid: %s\n" % str(rra_resid))
        #sys.stderr.write("rde_resid: %s\n" % str(rde_resid))
        rra_scatter = calc_MAR(rra_resid)
        rde_scatter = calc_MAR(rde_resid)
        #sys.stderr.write("rra_scatter: %e (rad)\n" % rra_scatter)
        #sys.stderr.write("rde_scatter: %e (rad)\n" % rde_scatter)

        ra_rweights = self._calc_huber_rweights(rra_resid, rra_scatter)
        #self._use_RA_err = ra_rweights * self._RA_err
        self._use_RA_err = self._RA_err / ra_rweights
        de_rweights = self._calc_huber_rweights(rde_resid, rde_scatter)
        self._use_DE_err = self._DE_err / de_rweights

        # find minimum:
        self.iresult = opti.fmin(self._calc_chi_square, params ,
                xtol=1e-7, ftol=1e-7, full_output=True)

        sys.stderr.write("Found IRLS minimum:\n")
        sys.stderr.write("==> %s\n" % str(self.nice_units(self.iresult[0])))
        self._can_iterate = True
        return self.iresult[0]


    # ----------------------------------------------------------------------- 
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
# CHANGELOG (astrom_test_2.py):
#---------------------------------------------------------------------
#
#  2020-02-07:
#     -- Increased __version__ to 0.1.0.
#     -- First created astrom_test_2.py.
#

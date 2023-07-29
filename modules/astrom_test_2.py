#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Second cut at astrometry fitting for UCD project.
#
# Rob Siverd
# Created:       2021-08-30
# Last modified: 2023-07-28
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
__version__ = "0.1.3"

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

    def __init__(self, vlevel=1):
        self._stream = sys.stderr
        self._vlevel = vlevel
        self._chiexp = 2
        self._reset()
        return

    def set_vlevel(self, vlevel):
        self._vlevel = vlevel

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

    def _reset(self):
        self._jd_tdb = None
        self._dt_yrs = None
        self.obs_eph = None
        self.ref_tdb = None
        self.inliers = None
        self.rweight = None
        self._is_set = False
        self._is_rdy = False

        # solver states:
        self._can_iterate  = False
        self._is_converged = False

        # solutions:
        self.full_result  = None
        self.iresult      = None
        self.iresult_prev = None
        self._have_result = False
        self._latest_pars = None
        return

    #def setup(self, jd_tdb_ref, RA_deg, DE_deg, obs_eph, 
    def setup(self, data, reject_outliers=True, jd_tdb_ref=None,
            #RA_err_key=None, DE_err_key=None,
            RA_err=None, DE_err=None,
            ra_key='dra', de_key='dde'):
        #self._is_rdy = False
        self._reset()
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
        #self._RA_rad = np.radians(self.dataset['dra'])
        #self._DE_rad = np.radians(self.dataset['dde'])
        self._RA_rad = np.radians(self.dataset[ra_key])
        self._DE_rad = np.radians(self.dataset[de_key])
        self._cosdec = np.cos(self._DE_rad)
        #self._RA_med, self._RA_MAD = calc_ls_med_MAD(self._RA_rad)
        #self._DE_med, self._DE_MAD = calc_ls_med_MAD(self._DE_rad)
        #self._RA_MAD *= np.cos(self._DE_med)

        # Initialize error vectors as needed:
        #self._RA_err = None
        #self._DE_err = None
        #if RA_err_key:
        #    self._RA_err = self.dataset[RA_err_key] 
        #self._RA_err = self.dataset[RA_err_key] if RA_err_key else None
        #self._DE_err = self.dataset[DE_err_key] if DE_err_key else None

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

    # ------------------------------- #
    #    Some Getters and Setters     #
    # ------------------------------- #

    # Retrieve some useful output when done:
    def collect_result_dataset(self):
        # first, identify best solution:
        if not self._have_result:
            sys.stderr.write("No solution to collect yet!\n")
            return None
        # calculate residuals (RADIANS, with cos(dec)):
        ra_res_rad, de_res_rad = \
                self._calc_radec_residuals_tru(self._latest_pars)
        ra_res_mas = self._mas_per_rad * ra_res_rad 
        de_res_mas = self._mas_per_rad * de_res_rad 
        signal = self.dataset['flux'] * self.dataset['exptime']
        rdata = append_fields(self.dataset,
                ('fit_resid_ra_mas', 'fit_resid_de_mas', 'signal', 'inliers'),
                (ra_res_mas, de_res_mas, signal, self.inliers),
                usemask=False)

        # include inlier/outlier flags:
        return rdata

    def get_latest_params(self):
        return self._latest_pars.copy()

    def get_bestfit_prmot_rad(self):
        return self._prmot_eval(self._latest_pars)

    def get_bestfit_prmot_deg(self):
        pmra_rad, pmde_rad = self._prmot_eval(self._latest_pars)
        return np.degrees(pmra_rad), np.degrees(pmde_rad)

    # Calculate RA/DE residuals w.r.t. best-fit proper motion (RADIANS):
    def get_radec_minus_prmot_rad(self, usecosdec=False):
        rra_model, rde_model = self._prmot_eval(self._latest_pars)
        rra_delta = self._RA_rad - rra_model
        rde_delta = self._DE_rad - rde_model
        if cosdec:
            rra_delta *= self._cosdec
        return rra_delta, rde_delta

    # Calculate RA/DE residuals w.r.t. best-fit proper motion (DEGREES):
    def get_radec_minus_prmot_deg(self, usecosdec=False):
        rra_delta, rde_delta = self.get_radec_minus_prmot_rad(usecosdec)
        return np.degrees(rra_delta), np.degrees(rde_delta)

    # Calculate RA/DE residuals w.r.t. best-fit proper motion (DEGREES):
    def get_radec_minus_prmot_mas(self, usecosdec=False):
        dra_delta, dde_delta = self.get_radec_minus_prmot_deg(usecosdec)
        return 3600.0*dra_delta, 3600.0*dde_delta

    # -----------------------------------------------------------------------

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

    # Evaluate a set of astrometric parameters using the
    # timestamps loaded into the solver. The stored parameters
    # are entirely in radians and coordinates
    # NOTES:
    #   * self._dt_yrs = Julian years (TDB), specifically,
    #                  = (JD_TDB - JD_TDB_ref) / 365.25
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
    
    # Evaluate just the proper motion component from a set of parameters:
    def _prmot_eval(self, params):
        rra, rde, pmra, pmde, prlx = params
        delta_ra = self._dt_yrs * pmra
        delta_de = self._dt_yrs * pmde
        return (rra + delta_ra, rde + delta_de)

    # RA/DE residuals in true units (cos(dec) corrected):
    def _calc_radec_residuals_tru(self, params, inliers=False):
        """This is the 'coordinate' version of the residual
        calculator. The RA component IS CORRECTED by cos(dec).
        The residuals returned by this routine are the differences
        between RA, Dec data and model in proper spherical units.
        """
        model_RA, model_DE = self._solver_eval(params)
        resid_RA = (self._RA_rad - model_RA) * np.cos(model_DE)
        resid_DE = (self._DE_rad - model_DE)
        if inliers:
            return resid_RA[self.inliers], resid_DE[self.inliers]
        else:
            return resid_RA, resid_DE

    # RA/DE residuals in coordinate units (no cos(dec) correction):
    def _calc_radec_residuals_coo(self, params, inliers=False):
        """This is the 'coordinate' version of the residual
        calculator. The RA component is NOT corrected by cos(dec).
        The residuals returned by this routine are the actual coordinate
        differences between RA, Dec data and model."""
        model_RA, model_DE = self._solver_eval(params)
        resid_RA = self._RA_rad - model_RA
        resid_DE = self._DE_rad - model_DE
        if inliers:
            return resid_RA[self.inliers], resid_DE[self.inliers]
        else:
            return resid_RA, resid_DE

    def _calc_radec_residuals(self, params, inliers=False):
        """This is the OLD AND DEPRECATED 'coordinate' version of the residual
        calculator. The RA component is NOT corrected by cos(dec).
        The residuals returned by this routine are the actual differences
        in RA, Dec coordinates between model and data. This version of
        the routine (with ambiguous name) is DEPRECATED in favor of
        the identical '_coo' variant above. This should help avoid
        confusion in the long term."""
        sys.stderr.write("_calc_radec_residuals() is DEPRECATED!\n")
        model_RA, model_DE = self._solver_eval(params)
        resid_RA = self._RA_rad - model_RA
        resid_DE = self._DE_rad - model_DE
        if inliers:
            return resid_RA[self.inliers], resid_DE[self.inliers]
        else:
            return resid_RA, resid_DE

    def _calc_radec_residuals_sigma(self, params):
        model_RA, model_DE = self._solver_eval(params)
        #rsigs_RA = (self._RA_rad - model_RA) / self._RA_err
        #rsigs_DE = (self._DE_rad - model_DE) / self._DE_err
        rsigs_RA = (self._RA_rad - model_RA) / self._use_RA_err
        rsigs_DE = (self._DE_rad - model_DE) / self._use_DE_err
        return rsigs_RA, rsigs_DE

    ## FIXME: the following implies that the sigma-based residuals
    ## are in matched units of arcseconds (cos(dec) correction applied).
    ## I don't think this is the case, meaning the residual calculation
    ## will exhibit bad behavior at high Dec. EEEEK!
    def _calc_total_residuals_sigma(self, params):
        return np.hypot(*self._calc_radec_residuals_sigma(params))

    def _calc_chi_square(self, params, negplxhit=100.):
        #model_ra, model_de = self._solver_eval(params)
        ##resid_ra = (model_ra - self._RA_rad) #/ np.cos(model_de)
        ##resid_de = (model_de - self._DE_rad)
        #resid_ra = (self._RA_rad - model_ra) #/ np.cos(model_de)
        #resid_de = (self._DE_rad - model_de)
        resid_ra, resid_de = self._calc_radec_residuals_coo(params)
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
        rra_resid, rde_resid = self._calc_radec_residuals_coo(params)
        mar_ra_rad = calc_MAR(rra_resid)
        mar_ra_mas = _MAS_PER_RADIAN * mar_ra_rad
        self._vlwrite("mar_ra_rad: %f\n" % mar_ra_rad, 2)
        self._vlwrite("mar_ra_mas: %f\n" % mar_ra_mas, 2)
        pfra, pfde = self._calc_parallax_factors(
                self._RA_rad, self._DE_rad, self.dataset['obs_x'],
                self.dataset['obs_y'], self.dataset['obs_z'])
        #sys.stderr.write("pfra_arcsec: %s\n" % str(pfra_arcsec))
        #pfra_rad   = pfra_arcsec / _ARCSEC_PER_RADIAN
        adjustment_arcsec = ts.linefit(pfra, _ARCSEC_PER_RADIAN * rra_resid)
        #sys.stderr.write("adjustment (arcsec): %s\n" % str(adjustment_arcsec))
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
        if self._vlevel >= 2:
            sys.stderr.write("Initial guess (unweighted):\n")
            sys.stderr.write("==> %s\n" % str(self.nice_units(uguess)))
            sys.stderr.write("\n")
            sys.stderr.write("Initial guess (weighted):\n")
            sys.stderr.write("==> %s\n" % str(self.nice_units(wguess)))
            sys.stderr.write("\n")
        guess = uguess  # adopt unweighted for now
        #guess[4] = 1000. / _MAS_PER_RADIAN

        # initial crack at parallax and zero-point:
        self._plx0 = self._calc_initial_parallax(guess)
        self._vlwrite("plx0: %s\n" % str(self._plx0), 2)
        ra_nudge_rad, plx_rad = self._plx0 / _ARCSEC_PER_RADIAN
        guess[0] += ra_nudge_rad
        guess[4] = plx_rad

        # estimate RA,Dec uncertainty from residuals if not known a priori:
        if self._need_resid_errors:
            rra_resid, rde_resid = self._calc_radec_residuals_coo(guess)
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
        #sys.stderr.write("typical rsig_tot: %8.3f\n" % np.median(rsig_tot))
        #sys.stderr.write("rsig_tot: %s\n" % str(rsig_tot))
        self.inliers = (rsig_tot < sigcut)
        ndropped = self.inliers.size - np.sum(self.inliers)
        message = "Dropped %d point(s) beyond %.2f-sigma.\n" % (ndropped, sigcut)
        self._vlwrite(message, 2)
        #sys.stderr.write("ra_res: %s\n" % str(ra_res))
        #sys.stderr.write("de_res: %s\n" % str(de_res))
        #sys.stderr.write("ra_sig: %s\n" % str(ra_sig))
        #sys.stderr.write("de_sig: %s\n" % str(de_sig))
 

        # find minimum:
        spamming = True if self._vlevel >= 2 else False
        self.full_result = opti.fmin(self._calc_chi_square, guess, 
                xtol=1e-7, ftol=1e-7, full_output=True, disp=spamming)
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
        self._have_result = True
        self._latest_pars = self.result
        return self.result
 
    # ----------------------------------------------------------------------- 
    def _calc_huber_rweights(self, residuals, sigma):
        #_k_sig = 1.00 * sigma
        _k_sig = 1.34 * sigma
        #_k_sig = 2.00 * sigma
        res_devs = np.abs(residuals / _k_sig)
        rweights = np.ones_like(res_devs)
        distants = (res_devs > 1.0)
        rweights[distants] = 1.0 / res_devs[distants]
        return rweights

    # perform one more iteration of parameter updates:
    def iter_update_bestpars(self, params):
        """Perform an IRLS iteration."""

        if not self._can_iterate:
            sys.stderr.write("Iteration not possible, solve first!\n")
            return None

        if self._is_converged:
            sys.stderr.write("Iteration has already converged!\n")
            return self.iresult[0]

        # calculate residuals:
        rra_resid, rde_resid = self._calc_radec_residuals_coo(params)
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
        self.iresult_prev = self.iresult    # save previous result
        spamming = True if self._vlevel >= 2 else False
        self.iresult = opti.fmin(self._calc_chi_square, params,
                xtol=1e-7, ftol=1e-7, full_output=True, disp=spamming)

        sys.stderr.write("Found IRLS minimum:\n")
        sys.stderr.write("==> %s\n" % str(self.nice_units(self.iresult[0])))

        # check for convergence:
        self._is_converged = self._convergence_check()

        self._can_iterate = True
        self._have_result = True
        self._latest_pars = self.iresult[0]
        return self.iresult[0]

    def _convergence_check(self):
        if self.iresult_prev == None:
            sys.stderr.write("First iteration, check skipped!\n")
            return False
        _params_match = np.all(self.iresult_prev[0] == self.iresult[0])
        _others_match = (self.iresult_prev[1:] == self.iresult[1:])
        #sys.stderr.write("_params_match: %s\n" % _params_match)
        #sys.stderr.write("_others_match: %s\n" % _others_match)
        return (_params_match and _others_match)

    def is_converged(self):
        return self._is_converged

    # ----------------------------------------------------------------------- 
    # ------------------------------- #
    #    Verbosity and Format Help    #
    # ------------------------------- #

    # vlevel-conscious messaging:
    def _vlwrite(self, msgtxt, vlmin):
        if (self._vlevel >= vlmin):
            self._stream.write(msgtxt)
        return


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

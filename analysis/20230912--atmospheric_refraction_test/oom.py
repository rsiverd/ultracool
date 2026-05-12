#!/usr/bin/env python3
#
# Order-of-magnitude estimate of differential refraction magnitude.
#
# NOTES:
# * https://en.wikipedia.org/wiki/Atmospheric_refraction
# * using Comstock formula from Wikipedia
# * also try bennett
#
# Filter info:
# http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?id=CFHT/Wircam.H2

import math
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

# WIRCam info and miscellany:
fov_diam_arcmin = 10.3
half_diam_deg   = 0.5 * fov_diam_arcmin / 60.0
wircam_pixscale = 0.305
_arcsec_per_deg = 3600.0
J_wlen          = 1248.13
H2_wlen         = 2129.84
nominal_P_mbar  =  618.5    # 618.5 +/- 1.25
nominal_temp_C  =    2.7    #   2.7 +/- 1.6
nominal_RH_pct  =   18.0

# Refraction constants:
p0 = 101325.            # pressure, Pa
tab_zz = np.array([0.0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70])
tab_fz = np.array([0.0,  0,  0,  0,  0,  2,  6, 12, 21, 34, 56, 97]) * 1e-4
tab_gz = np.array([4.0,  4,  4,  5,  5,  5,  6,  6,  7,  8, 10, 13]) * 1e-4

def Fcalc(zd_obs_deg, temp_C):
    fzvals = np.interp(zd_obs_deg, tab_zz, tab_fz)
    return (1.0 - 0.003592 * (temp_C - 15.) 
            - 0.0000055 * (temp_C - 7.5)**2) * (1 + fzvals)

def Gcalc(zd_obs_deg, temp_C, pressure_mbar):
    p_diff = (100. * pressure_mbar) - p0
    gzvals = np.interp(zd_obs_deg, tab_zz, tab_gz)
    return (1.0 + 0.9430e-5 * p_diff - 0.78e-10 * p_diff**2) * (1 + gzvals)

def calc_refr_laplace(zd_obs_deg):
    """This routine accepts a zenith distance in DEGREES and returns the
    refraction angle in ARCSECONDS using the Laplace formula."""
    tanz = np.tan(np.radians(zd_obs_deg))
    return 60.236*tanz - 0.0675*tanz**3

def lam_corr(lam_nm):
    return 0.98282 + 0.005981 / lam_nm**2

# Water vapor correction is a polynomial function of water vapor
# partial pressure. For background:
# https://www.engineeringtoolbox.com/relative-humidity-air-d_687.html
# * 1 mbar == 100 Pa
# * rel_hum = vapor_partial_pressure / saturation_vapor_partial_pressure
# * vapor_partial_pressure = rel_hum * saturation_vapor_partial_pressure
# * saturation vapor pressure ~~ e'_w   in Buck (1981)
# * abbreviation WVP == "water vapor pressure"
def h2o_corr(temp_C, pressure_mbar, rel_hum_pct):
    lhs = 1.0007 + (3.46e-6 * pressure_mbar)
    rhs = 6.1121 * math.exp(17.502 * temp_C / (240.97 + temp_C))
    saturation_WVP_mbar = lhs * rhs
    #partial_WVP_mbar = (rel_hum_pct / 100.) * saturation_WVP_mbar
    partial_WVP_Pa   = (rel_hum_pct / 100.) * saturation_WVP_mbar * 100
    return 1.0 - (0.152e-5 * partial_WVP_Pa) - (0.55e-9 * partial_WVP_Pa**2)

def calc_refr_tp(zd_obs_deg, temp_C, pressure_mbar):
    t_ratio = 1.0552126 / (1.0 + 0.00368084 * temp_C)
    p_ratio = 100. * pressure_mbar / p0
    F = Fcalc(zd_obs_deg, temp_C)
    G = Gcalc(zd_obs_deg, temp_C, pressure_mbar)
    R_0 = calc_refr_laplace(zd_obs_deg)
    return R_0 * p_ratio * t_ratio * F * G

def calc_refr_tplam(zd_obs_deg, temp_C, pressure_mbar, lam_nm):
    return calc_refr_tp(zd_obs_deg, temp_C, pressure_mbar) * lam_corr(lam_nm)

# Refraction(temperature, pressure, wavelength, humidity):
def calc_refr_TPWH(zd_obs_deg, temp_C, pressure_mbar, lam_nm, rel_hum_pct):
    R = calc_refr_tp(zd_obs_deg, temp_C, pressure_mbar)     # just T, P
    R *= lam_corr(lam_nm)                                   # T, P, wlen
    R *= h2o_corr(temp_C, pressure_mbar, rel_hum_pct)       # T, P, wlen, hum
    return R

#def calc_refr_comstock(pressure_mmhg, temp_C, alt_app):
#    """Returns refraction in arcseconds."""
#    numer = 21.5 * pressure_mmhg

def refr_multiplier(pressure_mbar=1010.0, temp_c=10.):
    return pressure_mbar * 0.2802 / (273. + temp_c)

def calc_refr_bennett(apparent_alt_deg, pressure_mbar=1010.0, temp_c=10.):
    """Returns refraction angle in degrees. This correction is SUBTRACTED
    from the apparent altitude to obtain the true altitude."""
    #if (apparent_alt_deg < -1.0) or (apparent_alt_deg > 89.9):
    #    return 0.0 * apparent_alt_deg
    #R_arcmin = np.zeros_like(apparent_alt_deg)
    tan_arg = apparent_alt_deg + (7.31 / (apparent_alt_deg + 4.4))
    #refr_arcmin = apparent_alt_deg + (7.31 / (apparent_alt_deg + 4.4))
    #R_arcmin = 1.0 / np.tan(np.radians(apparent_alt_deg + (7.31 / (alt_app_deg + 4.4))
    # (P_mbar / 1010.) * (283 / (273 + T_C))
    # 283 / 1010. =~ 0.2802
    multiplier = pressure_mbar * 0.2802 / (273. + temp_c)
    R_arcmin   = multiplier / np.tan(np.radians(tan_arg))
    if isinstance(apparent_alt_deg, np.ndarray):
        R_arcmin[apparent_alt_deg < -1.0] = 0.0
        R_arcmin[apparent_alt_deg > 90.0] = 0.0
    return R_arcmin / 60.0

# Saemundsson inverse refraction formula given true altitude:
def calc_refr_inverse_saem(true_alt_deg, pressure_mbar=1010.0, temp_c=10.):
    """Returns refraction angle in degrees. This correction is ADDED to
    the true altitude to obtain the apparent altitude."""
    tan_arg = true_alt_deg + (10.3 / (true_alt_deg + 5.11))
    multiplier = pressure_mbar * 0.2802 / (273. + temp_c)
    R_arcmin   = multiplier * 1.02 / np.tan(np.radians(tan_arg))
    if isinstance(true_alt_deg, np.ndarray):
        R_arcmin[true_alt_deg < -1.0] = 0.0
        R_arcmin[true_alt_deg > 90.0] = 0.0
    return R_arcmin / 60.0

# Test altitudes (APPARENT, degrees):
#trial_alt = np.arange(90)
trial_app_alt_top = np.linspace(10, 90, 200)
trial_app_alt_mid = trial_app_alt_top - half_diam_deg
trial_app_alt_bot = trial_app_alt_mid - half_diam_deg


# Fancy version:
refr_mid_fancy_J = calc_refr_TPWH(90-trial_app_alt_mid, 
        nominal_temp_C, nominal_P_mbar, J_wlen, nominal_RH_pct)
refr_mid_fancy_H = calc_refr_TPWH(90-trial_app_alt_mid, 
        nominal_temp_C, nominal_P_mbar, H2_wlen, nominal_RH_pct)

# Top/mod/bot refraction (arcmin):
refr_top_deg = calc_refr_bennett(trial_app_alt_top)
refr_mid_deg = calc_refr_bennett(trial_app_alt_mid)
refr_bot_deg = calc_refr_bennett(trial_app_alt_bot)

# ACTUAL altitude would be:
true_alt_deg = trial_app_alt_mid - refr_mid_deg


# Differential refraction (arcsec and pixels)
rdif_top_arcsec = 3600*(refr_top_deg - refr_mid_deg)
rdif_bot_arcsec = 3600*(refr_mid_deg - refr_bot_deg)
rdif_top_pixels = rdif_top_arcsec / wircam_pixscale
rdif_bot_pixels = rdif_bot_arcsec / wircam_pixscale
rel_refr_arcsec = rdif_top_arcsec - rdif_bot_arcsec
rel_refr_pixels = rel_refr_arcsec / wircam_pixscale

# Change in refraction-per-degree as a function of altitude:
trial_app_alt_midpoints = 0.5*(trial_app_alt_mid[:-1] + trial_app_alt_mid[1:])
delta_refr_mid = np.diff(3600*refr_mid_deg) / np.diff(trial_app_alt_mid)


# Draw stuff:
fig = plt.figure(1, figsize=(14,7))
fig.clf()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.grid(True)
ax2.grid(True)
ax1.plot(trial_app_alt_mid, 60*refr_mid_deg)
ax1.set_xlabel('Altitude [deg]')
ax1.set_ylabel('Refraction [arcmin]')
ax2.plot(trial_app_alt_mid, rdif_top_pixels, label='sensor top-mid')
ax2.plot(trial_app_alt_mid, rdif_bot_pixels, label='sensor mid-bot')
ax2.set_ylabel('Refraction offset [pix]')
ax2.legend(loc='upper right')

fig.tight_layout()


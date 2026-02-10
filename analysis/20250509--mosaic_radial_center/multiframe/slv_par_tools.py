#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Helper module that provides parameter manipulation and inspection tools.
#
# Rob Siverd
# Created:       2026-01-27
# Last modified: 2026-01-27
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
#import argparse
#import shutil
#import resource
#import signal
#import glob
import gc
import os
import sys
import copy
import time
import pprint
#import pickle
#import vaex
#import calendar
#import ephem
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
#import scipy.stats as scst
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
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

## Helpers for this investigation:
import helpers
reload(helpers)

##--------------------------------------------------------------------------##
## GLOBAL CONFIG:
_quads = ('NE', 'NW', 'SE', 'SW')
sensor_order = ['NE', 'NW', 'SE', 'SW']

## Key catalog columns:
_xcol, _ycol = 'x', 'y'
_racol, _decol = 'calc_ra', 'calc_de'

##--------------------------------------------------------------------------##
## New-style string formatting (more at https://pyformat.info/):

## Easy-to-read parameter printout with
## * CDxx in [mas]
## * CRPIXn in [pix]
## * CRVALn in [deg]
def parprint(params, stream=sys.stderr):
    sfpar = sift_params(params)
    sfpar['cdmat'] = {qq:vv*3.6e6 for qq,vv in sfpar['cdmat'].items()}
    pprint.pprint(sfpar)

## ----------------------------------------------------------------------- ##
## Parameter-parser for solving and diagnostics:
def sift_params(params):
    parsleft = params.copy()

    # Peel CDxx, CRPIXx from the front:
    cdmcrpix = parsleft[:24].reshape(-1, 6)
    parsleft = parsleft[24:]
    #cdmat_4s = cdmcrpix[:, :4]
    test_cdm_calc = {qq:vv for qq,vv in zip(sensor_order, cdmcrpix[:, :4])}
    test_sensor_crpix = {qq:vv for qq,vv in zip(sensor_order, cdmcrpix[:, 4:])}

    # Peel CRVAL1, CRVAL2 from the front next:
    cv1, cv2 = parsleft[:2]
    #crvals12 = parsleft[:2]
    #parsleft = parsleft[2:]

    # Remaining parameters are the distortion model:
    rdist_pars = parsleft[2:]

    return {'cdmat':test_cdm_calc, 'crpix':test_sensor_crpix,
            'crval':[cv1, cv2], 'rpars':rdist_pars}

## Inverse operation:
def unsift_params(sifted):
    parvec = []
    # CD matrix and CRPIX (6 pars per sensor) go first:
    for qq in sensor_order:
        parvec.extend(sifted['cdmat'][qq])
        parvec.extend(sifted['crpix'][qq])
    parvec.extend(sifted['crval'])          # CRVALs go next
    parvec.extend(sifted['rpars'])          # distortion model last
    return np.array(parvec)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Calculate sensor-to-sensor gutters implied by various CRPIX values:
def describe_gutters(sifted):
    sensor_crpix = sifted['crpix']
    ne_crp1, ne_crp2 = sensor_crpix['NE']
    nw_crp1, nw_crp2 = sensor_crpix['NW']
    se_crp1, se_crp2 = sensor_crpix['SE']
    sw_crp1, sw_crp2 = sensor_crpix['SW']
    ne_nw = ne_crp1 - nw_crp1 - 2048.
    se_sw = se_crp1 - sw_crp1 - 2048.
    ne_se = se_crp2 - ne_crp2 - 2048.
    nw_sw = sw_crp2 - nw_crp2 - 2048.
    sys.stderr.write("NE <--> NW (upper): %.2f\n" % ne_nw)
    sys.stderr.write("SE <--> SW (lower): %.2f\n" % se_sw)
    sys.stderr.write("NE <--> SE ( left): %.2f\n" % ne_se)
    sys.stderr.write("NW <--> SW (right): %.2f\n" % nw_sw)
    return


def describe_rotations(sifted):
    _sord = sensor_order
    _smap = sensor_qqmap
    sensor_cdmat = sifted['cdmat']
    _data = np.array([helpers.analyze(sensor_cdmat[x]) for x in _sord])
    #sensor_padeg = _data[:, 0]
    sensor_padeg = {qq:vv for qq,vv in zip(_sord, _data[:, :1])}
    sensor_scale = {qq:vv for qq,vv in zip(_sord, _data[:, 1:]*3600)}
    #sensor_padeg, sensor_xscale, sensor_yscale = {}, {}, {}
    for pair in sensor_padeg.items():
        sys.stderr.write("Sensor %s PA: %.3f\n" % pair)
    for qq,psc in sensor_scale.items():
        sys.stderr.write("Sensor %s pixscales: %s\n" % (qq, str(psc)))
    # Relative PAs:
    for ii,jj in itt.combinations(range(len(_sord)), 2):
        qi, qj = _smap[ii], _smap[jj]
        pi, pj = sensor_padeg[qi], sensor_padeg[qj]
        sys.stderr.write("Between %s and %s: %.4f\n" % (qi, qj, pi-pj))
    return

def describe_answer(sifted):
    describe_gutters(sifted)
    return

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## This routine makes lower/upper bounds arrays for opti.least_squares. The
## user specifies which (if any) kinds of parameters should be held fixed.
## Available options for fixing are: cdmat, crpix, crval
## The return quantity is a lower,upper bounds tuple.
##
## This routine works by first producing a 'delta' array with infinities for
## parameters to vary and tiny values for parameters to hold fixed. This delta
## is then subbed from and added to the input params to produce the bounds.
def make_bounds_like(pars, *, fixed=[], fixtol=1e-8):
    #_quads = ('NE', 'NW', 'SE', 'SW')
    # by default, all parameters vary (delta = np.inf)
    #sifted_delta = spt.sift_params(np.zeros_like(pars)+np.inf)
    sifted_delta = sift_params(np.zeros_like(pars)+np.inf)
    if 'crpix' in fixed:
        sys.stderr.write("Fixing CRPIXs!\n")
        sifted_delta['crpix'] = \
                dict(zip(_quads, np.ones(8).reshape(4,2)*fixtol))
    if 'crval' in fixed:
        sys.stderr.write("Fixing CRVALs!\n")
        sifted_delta['crval'] = np.ones(2)*fixtol
    if 'cdmat' in fixed:
        sys.stderr.write("Fixing CD matrices!\n")
        sifted_delta['cdmat'] = \
                dict(zip(_quads, np.ones(16).reshape(4,4)*fixtol))
    #return sifted_delta
    #valids = [x for x in fixed if x in sifted_delta.keys()]
    #sys.stderr.write("valids: %s\n" % str(valids))
    #for ptype in valids:
    #    sys.stderr.write("ptype: %s\n" % ptype)
    #    #import pdb; pdb.set_trace()
    #plusorminus = np.abs(pars * spt.unsift_params(sifted_delta))
    plusorminus = np.abs(pars * unsift_params(sifted_delta))
    return (pars - plusorminus, pars + plusorminus)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Pretty good guess for the distortion:
guess_distmod = np.array([ 1.757279e-01,  1.175609e-03,  1.047979e-06,
                           5.059445e-11,  4.450758e-13, -1.039322e-16])

## Polynomial model. Hopefully this is a strictly positive value.
#def poly_eval(r, model):
#    #return model[0] + model[1]*r + model[2]*r*r
#    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r
#    return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r + model[4]*r*r*r*r
#    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r \
#    #        + model[4]*r*r*r*r + model[5]*r*r*r*r*r

def poly_eval2(r, c0, c1, c2):
    return c0 + r * (c1 + r * c2)

def poly_eval3(r, c0, c1, c2, c3):
    return c0 + r * (c1 + r * (c2 + r * c3))

def poly_eval4(r, c0, c1, c2, c3, c4):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r*c4)))

def poly_eval5(r, c0, c1, c2, c3, c4, c5):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r*c5))))

def poly_eval(r, model):
    return poly_eval5(r, *model)

## Radial distortion model X- and Y- corrections. With a strictly positive
## distortion magnitude, you need to *SUBTRACT* these from RA/DE-derived
## positions in order to compare with measured X,Y positions.
def calc_rdist_corrections(xrel, yrel, model):
    rdist = np.hypot(xrel, yrel)     # distance from CRPIX
    rcorr = poly_eval(rdist, model)  # total correction magnitude
    theta = np.arctan2(yrel, xrel)
    xcorr = rcorr * np.cos(theta)
    ycorr = rcorr * np.sin(theta)
    return xcorr, ycorr

## ----------------------------------------------------------------------- ##
## Misrotation calculator:
def calc_roterr_deg(ddata):
    xctr, yctr = 1024.5, 1024.5
    have_ang = np.arctan2(ddata['ymeas'] - yctr,
                          ddata['xmeas'] - xctr)
    want_ang = np.arctan2(ddata['ymeas'] - yctr + ddata['yerror'],
                          ddata['xmeas'] - xctr + ddata['xerror'])
    return np.median(np.degrees(want_ang - have_ang))

## Check for residual rotation in the diagnostic data:
def show_misrotations(diags):
    result = {qq:calc_roterr_deg(dd) for qq,dd in diags.items()}
    for qq in sensor_order:
        sys.stderr.write("Residual %s rotation: %9.4f deg\n" % (qq, result[qq]))
    return result


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## ----------------------------------------------------------------------- ##
## As noted above, this convention SUBTRACTS calculated x- and y-corrections
## from RA/DE-derived xrel/yrel coordinates before comparing to measured X,Y.
## THIS VERSION ALSO FITS RADIAL DISTORTION PARAMETERS
## This version expects per-sensor CD11, CD12, CD21, CD22, CRPIX1, CRPIX2 at
## the front of the parameters array.
def squared_residuals_foc2ccd_rdist(params, dataset, diags=False,
                    unsquared=False, snrweight=False):
    # parse parameters
    #sifted = spt.sift_params(params)
    sifted = sift_params(params)
    brute_cdmat = sifted['cdmat']
    brute_crpix = sifted['crpix']
    using_crval = sifted['crval']
    #rdist_pars  = sifted['rpars']

    #nextra = 5 - len(rdist_pars)
    #nextra = 6 - len(rdist_pars)
    #test_distmod = rdist_pars.tolist() + [0.0]*nextra
    test_distmod = guess_distmod

    # note average star count for normalization
    avg_nstars = np.average([len(x) for x in dataset.values()])
    typical_rdist = 1448.0     # 0.5 * np.hypot(2048, 2048)
    qxres, qyres = {}, {}
    xres, yres = [], []
    diag_data = {}
    for qq,gst in dataset.items():
        nstar_scale_factor = np.sqrt(avg_nstars / float(len(gst)))
        tcpx1, tcpx2 = brute_crpix.get(qq)
        #gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        gxx, gyy = gst['x'], gst['y']
        #cdmcrv = np.array(brute_crval.get(qq).tolist() + [test_crval1, test_crval2])
        cdmcrv = np.array(brute_cdmat.get(qq).tolist() + using_crval)
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    dataset[qq]['gra'], dataset[qq]['gde'])
        #import pdb; pdb.set_trace()
        #breakpoint()
        test_rrel = np.hypot(test_xrel, test_yrel)
        xnudge, ynudge = calc_rdist_corrections(test_xrel, test_yrel, test_distmod)
        test_xccd = test_xrel + xnudge + tcpx1
        test_yccd = test_yrel + ynudge + tcpx2
        x_error = test_xccd - gxx.values
        y_error = test_yccd - gyy.values
        scaled_xerr = x_error * nstar_scale_factor
        scaled_yerr = y_error * nstar_scale_factor
        if snrweight:
            scaled_xerr /= gst['realerr']
            scaled_yerr /= gst['realerr']
        #scaled_xerr *= test_rrel / typical_rdist    # more weight far away
        #scaled_yerr *= test_rrel / typical_rdist    # more weight far away
        #scaled_xerr *= np.sqrt(test_rrel / typical_rdist)    # more weight far away
        #scaled_yerr *= np.sqrt(test_rrel / typical_rdist)    # more weight far away
        #qxres[qq] = scaled_xerr
        #qyres[qq] = scaled_yerr
        xres.extend(scaled_xerr)
        yres.extend(scaled_yerr)
        #xres.extend(xres)
        if diags:
            diag_data[qq] = {   "gid":gst['gid'],
                              "xmeas":gxx,
                              "ymeas":gyy,
                              "xcalc":test_xccd,
                              "ycalc":test_yccd,
                              'rdist':test_rrel,
                             'xnudge':xnudge,
                             'ynudge':ynudge,
                             'xerror':x_error,
                             'yerror':y_error,
                             'rerror':np.hypot(x_error, y_error),
                             'scaled_xerror':scaled_xerr,
                             'scaled_yerror':scaled_yerr,
                             'scaled_rerror':np.hypot(scaled_xerr, scaled_yerr),
                               'flux':gst['flux'],
                               'fwhm':gst['fwhm'],
                              'flags':gst['flag'],
                            'dumbsnr':gst['dumbsnr'],
                            'realerr':gst['realerr'],
                            }

        pass
    #return qxres, qyres
    #return xres, yres
    if diags:
        return diag_data
    if unsquared:
        return np.concatenate((xres, yres))
    return np.concatenate((xres, yres))**2

def fmin_squared_residuals_foc2ccd_rdist(params, dataset, **kwargs):
    return np.sum(squared_residuals_foc2ccd_rdist(params, dataset, **kwargs))

##--------------------------------------------------------------------------##
##------------------     Residual Check & Param Tuning      ----------------##
##--------------------------------------------------------------------------##

def derotate_fit_parameters(fit_params, dataset):
    # Extract diagnostics:
    sifted_pars = sift_params(fit_params)
    diag_data = squared_residuals_foc2ccd_rdist(fit_params, 
                                                dataset=dataset, diags=True)

    # Estimate and correct residual rotation. Note that this effectively
    # scrambles CRPIX (which is corrected below).
    #rot_error = spt.show_misrotations(diag_data)
    rot_error = show_misrotations(diag_data)
    derot_sifted = copy.deepcopy(sifted_pars)
    for qq,resid in rot_error.items():
        rmat = helpers.rotation_matrix(-1.0 * np.radians(resid))
        _new_cdm = np.dot(rmat, derot_sifted['cdmat'][qq].reshape(2, 2))
        derot_sifted['cdmat'][qq] = _new_cdm.flatten()
        pass
    #derot_params = spt.unsift_params(derot_sifted)
    derot_params = unsift_params(derot_sifted)

    # Re-evaluate residuals with de-rotated parameters.
	#derot_diags = squared_residuals_foc2ccd_rdist(derot_params, diags=True)
    #derot_diags = spt.squared_residuals_foc2ccd_rdist(derot_params,
    derot_diags = squared_residuals_foc2ccd_rdist(derot_params,
                                            dataset=dataset, diags=True)
    diag_data = derot_diags

    # Modify CRPIX to soak up the offset(s) introduced by de-rotation:
    fixed_sifted = copy.deepcopy(derot_sifted)
    for qq,ddata in derot_diags.items():
        fixed_sifted['crpix'][qq][0] -= np.median(ddata['xerror'])
        fixed_sifted['crpix'][qq][1] -= np.median(ddata['yerror'])
    #fixed_params = spt.unsift_params(fixed_sifted)
    fixed_params = unsift_params(fixed_sifted)

    # Re-evaluate residuals with de-rotated and de-shifted parameters:
    #fixed_diags = spt.squared_residuals_foc2ccd_rdist(fixed_params,
    fixed_diags = squared_residuals_foc2ccd_rdist(fixed_params,
                                                      dataset=dataset, diags=True)
    #diag_data = fixed_diags

    # Return the updated parameter sets:
    return derot_params, fixed_params



##--------------------------------------------------------------------------##
##------------------     Post-Solution RA/DE Update         ----------------##
##--------------------------------------------------------------------------##

## Calculate the current X,Y distortion offset:
def relpix_diff(xmeas, ymeas, xproj, yproj, model=guess_distmod):
    #xcorr, ycorr = spt.calc_rdist_corrections(xproj, yproj, model)
    xcorr, ycorr = calc_rdist_corrections(xproj, yproj, model)
    xdiff = xmeas - (xproj + xcorr)
    ydiff = ymeas - (yproj + ycorr)
    return xdiff, ydiff

def qrms(vec):
    return np.sqrt(np.sum(vec**2))

## Iterate to convergence:
def iter_calc_true_xyrel(xrel_dist, yrel_dist, distmod,
                         xytol=1e-12, itermax=20, verbose=False):
    xnew, ynew = xrel_dist.copy(), yrel_dist.copy()
    for i in range(itermax):
        xerr, yerr = relpix_diff(xrel_dist, yrel_dist, xnew, ynew, distmod)
        xrms, yrms = qrms(xerr), qrms(yerr)
        if verbose:
            sys.stderr.write("Current RMS (%d): %.5e, %.5e\n" % (i, xrms, yrms))
        if max(np.max(np.abs(xerr)), np.max(np.abs(yerr))) < xytol:
            if verbose:
                sys.stderr.write("Converged!\n")
            break
        else:
            xnew += xerr
            ynew += yerr
    return xnew, ynew   # undistorted xrel, yrel

### Validate (SUCCESSFUL):
#xcorr, ycorr = spt.calc_rdist_corrections(xrel_true, yrel_true,
#                                                spt.guess_distmod)
#xcalc, ycalc = xrel_true + xcorr + crpix1, yrel_true + ycorr + crpix2
#sys.stderr.write("X std: %.8ee\n" % np.std(ss['x'] - xcalc))
#sys.stderr.write("Y std: %.8ee\n" % np.std(ss['y'] - ycalc))
#
#xerr, yerr = relpix_diff(xmeas, ymeas, xnew, ynew)
#xvar, yvar = np.sum(xerr**2), np.sum(yerr**2)
#sys.stderr.write("Current vars: %.3f, %.3f\n" % (xvar, yvar))

## Recalculate catalog RA/DE using latest parameters. This routine
## expects the original 'stars' array and the latest solution parameters.
def inplace_update_catalog_radec(data, new_params):
    sparams = sift_params(new_params)
    for qq,ss in data.items():
        #sparams['cdmat'][qq]
        crpix1, crpix2 = sparams['crpix'][qq]
        cdmcrv = np.array(sparams['cdmat'][qq].tolist() + sparams['crval'])
        xrel_dist = ss[_xcol] - crpix1   # includes distortion
        yrel_dist = ss[_ycol] - crpix2   # includes distortion
        # Compute undistorted relative X,Y positions. We need these in order
        # to accurately recalculate RA/DE. The distortion model is defined in
        # terms of undistorted coordinates, so we need to iterate to a solution.
        xrel_true, yrel_true = \
                iter_calc_true_xyrel(xrel_dist, yrel_dist, guess_distmod)
                #iter_calc_true_xyrel(xrel_dist, yrel_dist, spt.guess_distmod)
        newra, newde = helpers.eval_cdmcrv(cdmcrv, xrel_true, yrel_true)
        ss[_racol] = newra % 360.0
        ss[_decol] = newde
        pass
    return data


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##





######################################################################
# CHANGELOG (slv_par_tools.py):
#---------------------------------------------------------------------
#
#  2026-01-27:
#     -- Increased __version__ to 0.0.1.
#     -- First created slv_par_tools.py.
#

#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This is a WCS solution helper module that handles loading of Gaia data
# and fcat object catalogs from disk.
#
# Rob Siverd
# Created:       2026-01-27
# Last modified: 2026-01-27
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Modules:
#import shutil
#import glob
import gc
import os
import sys
import time
#import pprint
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
import pandas as pd
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
from importlib import reload

## Gaia catalog matching:
import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()

### Assert magnitude limit for matching:
#gmag_limit = 19.0
#gm.set_Gmag_limit(gmag_limit)


## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ecl = extended_catalog.ExtendedCatalog()
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Helpers for this investigation:
import helpers
reload(helpers)

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

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## By default, the output 'stream' is stderr but this can be modified
## by the calling routine/application.
#stream = sys.stderr

## Key catalog columns:
_xcol, _ycol = 'x', 'y'
_racol, _decol = 'calc_ra', 'calc_de'

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Embed approximate SNR/uncertainty. Assume:
## * sky + rdnoise^2 =~ 1000.0     (RDNOISE ~ 30)
## * sigma =~ FWHM / SNR
## NOTE: to calibrate, I used known-decent parameters and evaluated the
## following: median(sqrt((resid / asterr)**2))  ==>  2.64
## That factor of 2.64 is the factor by which the errors are underestimated
## This value is consistent with the ~0.05 pixel RMS seen previously.
_rescale = 2.64
_wirgain = 3.8                  # electrons per ADU
#_fluxcol = 'FLUX_ISO'
_fluxcol = 'flux'


## Load catalogs from 'cpath' dictionary. Once loaded, extract observation
## time and adjust Gaia matcher epoch accordingly. Finally, promote data to
## pandas DataFrame and insert added value columns.
def load_fcats(cpath, stream=sys.stderr):
    # Load ExtendedCatalog data:
    stream.write("Loading catalogs ... ")
    cdata = {}
    chdrs = {}
    for qq,fpath in cpath.items():
        ecl.load_from_fits(fpath)
        cdata[qq] = ecl.get_catalog()
        chdrs[qq] = ecl.get_header()

    # Set Gaia epoch:
    stream.write("Gaia epoch ... ")
    obs_time = helpers.wircam_timestamp_from_header(chdrs['NE'])
    gm.set_epoch(obs_time)

    # Promote to DataFrame:
    stream.write("df promotion ... ")
    stars = {qq:pd.DataFrame.from_records(tt) for qq,tt in cdata.items()}

    # Embed approximate SNR (see comments above):
    stream.write("SNR ... ") 
    for ss in stars.values():
        ## FOR FWHM SEE https://github.com/sep-developers/sep/issues/34
        calc_fwhm = 2.0 * np.sqrt(np.log(2.) * (ss['a']**2 + ss['b']**2))
        src_ele = _wirgain * ss[_fluxcol]
        src_snr = src_ele / np.sqrt(src_ele + 1000.0)
        #ast_err = ss['FWHM_IMAGE'] / src_snr            # a
        ast_err = calc_fwhm / src_snr            # a
        ss[   'fwhm'] = calc_fwhm
        ss['dumbsnr'] = src_snr
        ss['dumberr'] = ast_err
        ss['realerr'] = ast_err * _rescale
    stream.write("done.\n")

    # Return everything:
    return cdata, chdrs, stars

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Make a uniform match tolerance dictionary:
def uniform_sensor_mtol(tol):
    return {'NE':tol, 'NW':tol, 'SE':tol, 'SW':tol}

## Match to Gaia with adjustable tolerance. Return copy of input catalogs
## augmented with Gaia matches (leave originals unchanged).
def register_catalogs_to_gaia(data, tol_arcsec, stream=sys.stderr):
    stream.write("Gaia matching ...\n")
    #_xcol, _ycol = 'x', 'y'
    sensor_mtol = uniform_sensor_mtol(tol_arcsec)
    stars = {kk:vv.copy() for kk,vv in data.items()}    # duplicate

    for qq,ss in stars.items():
        #csln = imwcs.get(qq)
        xpos, ypos = ss[_xcol], ss[_ycol]
        #sra, sde = imwcs[qq].all_pix2world(xpos, ypos, 1, ra_dec_order=True)
        #sra, sde = ss['calc_ra'], ss['calc_de']
        sra, sde = ss[_racol], ss[_decol]
        #ss['anet_ra'], ss['anet_de'] = sra, sde
        #matches = gm.twoway_gaia_matches(sra, sde, mtol_arcsec)
        #matches = gm.twoway_gaia_matches(sra, sde, sensor_mtol.get(qq))
        #matches = slvh.gm.twoway_gaia_matches(sra, sde, sensor_mtol.get(qq))
        matches = gm.twoway_gaia_matches(sra, sde, sensor_mtol.get(qq))
        idx, gra, gde, gid = matches
        gcosdec = np.cos(np.radians(gde))
        #mismatch =
        delta_ra_arcsec = 3600.0 * (gra - sra[idx]) * gcosdec
        delta_de_arcsec = 3600.0 * (gde - sde[idx])
        med_delta_ra, sig_delta_ra = rs.calc_ls_med_MAD(delta_ra_arcsec)
        med_delta_de, sig_delta_de = rs.calc_ls_med_MAD(delta_de_arcsec)
        sys.stderr.write("%s | RA~ %.3f +/- %.3f | DE~ %.3f +/- %.3f\n"
                % (qq, med_delta_ra, sig_delta_ra, med_delta_de, sig_delta_de))
    
        # also embed Gaia info:
        big_gid = np.zeros(len(xpos), dtype=gid.dtype)
        big_gra = np.zeros_like(sra) * np.nan
        big_gde = np.zeros_like(sde) * np.nan
        big_gid[idx] = gid
        big_gra[idx] = gra
        big_gde[idx] = gde
        ss['gid'] = big_gid
        ss['gra'] = big_gra
        ss['gde'] = big_gde
        #break
        pass
    sys.stderr.write("done.\n")
    return stars
 




##--------------------------------------------------------------------------##





######################################################################
# CHANGELOG (slv_helper.py):
#---------------------------------------------------------------------
#
#  2026-01-27:
#     -- Increased __version__ to 0.0.1.
#     -- First created slv_helper.py.
#

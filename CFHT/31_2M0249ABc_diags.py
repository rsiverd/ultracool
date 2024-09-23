#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Quick diagnostics to see if the data came out OK.
#
# Rob Siverd
# Created:       2024-06-03
# Last modified: 2024-06-03
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
#import argparse
#import shutil
import signal
#import glob
import gc
import os
import sys
import time
import pickle
#import vaex
#import calendar
#import ephem
import numpy as np
from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
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

## Angular math:
import angle
reload(angle)

## Astrometry fitting module:
import astrom_test_2
reload(astrom_test_2)
at2 = astrom_test_2
#af  = at2.AstFit()  # used for target
#afn = at2.AstFit()  # used for neighbors
#afA = at2.AstFit()  # for G125-24A
#afB = at2.AstFit()  # for G125-24B

## Detrending facility:
import detrending
reload(detrending)

##--------------------------------------------------------------------------##
## Projections with cartopy:
#try:
#    import cartopy.crs as ccrs
#    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#    from cartopy.feature.nightshade import Nightshade
#    #from cartopy import config as cartoconfig
#except ImportError:
#    sys.stderr.write("Error: cartopy module not found!\n")
#    sys.exit(1)

##--------------------------------------------------------------------------##
## Disable buffering on stdout/stderr:
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##

## Pickle store routine:
def stash_as_pickle(filename, thing):
    with open(filename, 'wb') as sapf:
        pickle.dump(thing, sapf)
    return

## Pickle load routine:
def load_pickled_object(filename):
    with open(filename, 'rb') as lpof:
        thing = pickle.load(lpof)
    return thing

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

## Home-brew KDE:
#try:
#    import my_kde
#    reload(my_kde)
#    mk = my_kde
#except ImportError:
#    logger.error("module my_kde not found!  Install and retry.")
#    sys.stderr.write("\nError!  my_kde module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

## Fast FITS I/O:
#try:
#    import fitsio
#except ImportError:
#    logger.error("fitsio module not found!  Install and retry.")
#    sys.stderr.write("\nError: fitsio module not found!\n")
#    sys.exit(1)

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

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80


##--------------------------------------------------------------------------##
## This third version allows the user to specify keyword choice/order if
## desired but defaults to the keys provided in the first dictionary.
def recarray_from_dicts(list_of_dicts, use_keys=None):
    keys = use_keys if use_keys else list_of_dicts[0].keys()
    data = [np.array([d[k] for d in list_of_dicts]) for k in keys]
    return np.core.records.fromarrays(data, names=','.join(keys))

## Concatenate record arrays field-by-field. This assumes that both record
## arrays have identical field names. It also assumes that arrays is a
## non-empty iterable containing structured arrays.
def concat_recs_by_field(arrays):
    if len(arrays) == 1:
        return np.concatenate(arrays)
    fld_list = arrays[0].dtype.names
    vec_list = []
    for kk in fld_list:
        vec_list.append(np.concatenate([x[kk] for x in arrays]))
    #vec_list = [np.concatenate([x[kk] for kk in fld_list for x in arrays])]
    return np.core.records.fromarrays(vec_list, names=','.join(fld_list))

def jdtdb_sorted(array):
    order = np.argsort(array['jdtdb'])
    return array[order]

##--------------------------------------------------------------------------##
## Magnitude conversions:

## Magnitude to flux conversion:
def kadu(mag, zeropt=25.0):
    return 10.0**(0.4 * (zeropt - mag))

## Flux to magnitude conversion:
def kmag(adu, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(adu))


##--------------------------------------------------------------------------##
## FIXME: consider including the detectino list in the big data pickle!

## The master detection list:
det_list = 'process/2M0249ABc_detections_J.txt'
sys.stderr.write("Loading detections list ... ")
gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
gftkw.update({'names':True, 'autostrip':True})
try:
    det_data = np.genfromtxt(det_list, dtype=None, **gftkw)
except:
    sys.stderr.write("FAILED!\n")
    sys.stderr.write("Missing or empty file?\n")
    sys.stderr.write("--> %s\n" % context.det_list)
    sys.exit(1)
sys.stderr.write("done.\n")
every_srcid = det_data['srcid']
src2ra = dict(zip(det_data['srcid'], det_data['dra']))
src2de = dict(zip(det_data['srcid'], det_data['dde']))


## Promote to DataFrame:
#det_data = pd.DataFrame(det_data)

##--------------------------------------------------------------------------##

## Pickled results of cross-matches:
data_file_J = 'process/2M0249ABc_Jdet_J_fcat.pickle'
data_file_H2 = 'process/2M0249ABc_Jdet_H2_fcat.pickle'

sci_names = ['2M0249AB', '2M0249c']

sys.stderr.write("Loading data ... ")
targets_J  = load_pickled_object(data_file_J)
targets_H2 = load_pickled_object(data_file_H2)
sys.stderr.write("done.\n")

## Merge stuff:
srcs_H2   = set(targets_H2.keys())
srcs_J    = set(targets_J.keys())
srcs_both = srcs_H2.intersection(srcs_J)
srcs_all  = set(targets_J.keys()).union(targets_H2.keys())

## The 'stationaries' are objects from the master detection list that
## had enough data to survive the matching stage.
stationaries = srcs_all.intersection(set(every_srcid))

## The 'movers' are objects with srcids not in the master list:
movers = srcs_all - stationaries

##--------------------------------------------------------------------------##
## Some global config:
_use_ra_key = 'calc_ra'
_use_de_key = 'calc_de'     

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##



## Data seen in BOTH channels:
sys.stderr.write("Concatenating and repackaging ... ")
targets_hold = {tid:[] for tid in srcs_all}
for tt in (targets_J, targets_H2):
    for kk,vv in tt.items():
        targets_hold[kk].append(vv)
#targets_both = {np.concatenate(srcs_H2[id]}
targets_all = {tt:jdtdb_sorted(concat_recs_by_field(vv)) \
        for tt,vv in targets_hold.items()}
#for tt,vv in targets_hold.items():
#    targets_all[tt] = np.concatenate(vv)
#targets_all = {tt:np.concatenate(vv) for tt,vv in targets_hold.items()}
sys.stderr.write("done.\n")

targets = targets_H2
#targets = targets_J
#targets = targets_all

#fast_A  = targets['2M0249AB']
#fast_B  = targets['2M0249c']
#ref_jdtdb = np.median(fast_A['jdtdb'])

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Check that the movers are not also present among the stationaries:
stationary_id = list(stationaries)
stationary_ra = np.array([src2ra[x] for x in stationary_id])
stationary_de = np.array([src2de[x] for x in stationary_id])

## Check each mover against those positions:
dupe_tol_arcsec = 0.5   # ~1.5 pixels
for targ in movers:
    #sys.stderr.write("targ: %s\n" % targ)
    data = targets[targ]
    npts = len(data)
    med_ra = np.median(data[_use_ra_key])
    med_de = np.median(data[_use_de_key])
    all_sep_deg = angle.dAngSep(med_ra, med_de, stationary_ra, stationary_de)
    idx_nearest = all_sep_deg.argmin()
    min_sep_deg = all_sep_deg[idx_nearest]    # degrees
    min_sep_arcsec = 3600.0 * min_sep_deg
    sys.stderr.write("%15s min_sep_arcsec: %8.2f\n" % (targ, min_sep_arcsec))
    if min_sep_arcsec <= dupe_tol_arcsec:
        dupe_srcid = stationary_id[idx_nearest]
        ndupe = len(targets[dupe_srcid])
        sys.stderr.write("DUPE DETECTED: %s\n" % dupe_srcid)
        sys.stderr.write("npts in %s -- %d\n" % (targ, npts))
        sys.stderr.write("npts in %s -- %d\n" % (dupe_srcid, ndupe))
        targets.pop(dupe_srcid)
        sys.stderr.write("Duplicate popped!\n")
        pass
    pass


#sys.exit(0)

## Source counts for all associated targets:
n_sources = {kk:len(vv) for kk,vv in targets.items()}
#huge_srcs = max(n_sources.values())
want_dets = max(n_sources.values()) / 2
proc_objs = [sid for sid,nn in n_sources.items() if nn > want_dets]
proc_data = {sid:targets[sid] for sid in proc_objs}
proc_objs_full = [x for x in proc_objs]

## Some typical target info:
imag_avgs = {}
imag_stds = {}
xpix_avgs = {}
ypix_avgs = {}
for targ in proc_objs:
    #signal = proc_data[targ]['flux'] * proc_data[targ]['exptime']
    signal = proc_data[targ]['flux']
    instmag = kmag(signal)
    med_mag, iqr_mag = rs.calc_ls_med_IQR(instmag)
    imag_avgs[targ] = med_mag
    imag_stds[targ] = iqr_mag
    xpix_avgs[targ] = np.median(proc_data[targ]['x'])
    ypix_avgs[targ] = np.median(proc_data[targ]['y'])

## Optionally limit my proc_objs list to N per mag bin:
def imag2bin(mag, lower, bsize):
    return int((mag - lower) / bsize)



#sys.exit(0)

##--------------------------------------------------------------------------##
## Initialize fitters:
fitters = {sid:at2.AstFit() for sid in proc_objs}

## Run processing:
num_todo = 0
sig_thresh = 3
save_fitters  = {}
save_bestpars = {}
maxiters = 30
for targ in proc_objs:
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Initial fit of: %s\n" % targ)
    afn = at2.AstFit()
    afn.setup(proc_data[targ], ra_key=_use_ra_key, de_key=_use_de_key)
    bestpars = afn.fit_bestpars(sigcut=sig_thresh)
    iterpars = bestpars.copy()
    for i in range(maxiters):
        sys.stderr.write("Iteration %d ...\n" % i)
        iterpars = afn.iter_update_bestpars(iterpars)
        if afn.is_converged():
            sys.stderr.write("Converged!  Iteration: %d\n" % i)
            break
    save_bestpars[targ] = afn.nice_units(iterpars)
    save_fitters[targ] = afn
    pass

##--------------------------------------------------------------------------##
## Pick some trend stars:
max_mag_stddev = 0.25
max_ast_stddev = 100.0  # mas
trend_targlist = []

## Snag one from lower-left:
trcount = {'UL':0, 'UR':0, 'LL':0, 'LR':0}
tr_qmax = 2
for targ in proc_objs:
    # Lower-left:
    if (trcount['LL'] < tr_qmax):
        if (xpix_avgs[targ] <  500.0) and (ypix_avgs[targ] <  500.0):
            trend_targlist.append(targ)
            trcount['LL'] += 1
    # Lower-right:
    if (trcount['LR'] < tr_qmax):
        if (xpix_avgs[targ] > 1000.0) and (ypix_avgs[targ] <  500.0):
            trend_targlist.append(targ)
            trcount['LR'] += 1
    # Upper-left:
    if (trcount['UL'] < tr_qmax):
        if (xpix_avgs[targ] <  500.0) and (ypix_avgs[targ] > 1000.0):
            trend_targlist.append(targ)
            trcount['UL'] += 1
    # Upper-right:
    if (trcount['UR'] < tr_qmax):
        if (xpix_avgs[targ] > 1000.0) and (ypix_avgs[targ] > 1000.0):
            trend_targlist.append(targ)
            trcount['UR'] += 1

## Ensure the list is unique:
trend_targlist = list(set(trend_targlist))

## Collect residual vectors from trend targets:
trend_resid_vecs = {}   # JDTDB, RA, DE, instr
for targ in trend_targlist:
    this_fit = save_fitters[targ]
    this_jdtdb = this_fit.dataset['jdtdb']
    this_instr = this_fit.dataset['instrument']
    ra_errs_mas, de_errs_mas = \
            this_fit.get_radec_minus_model_mas(cos_dec_mult=True)
    trend_resid_vecs[targ] = \
            (this_jdtdb, ra_errs_mas, de_errs_mas, this_instr)
    pass

## Detrend residuals:
#ICD_RA = detrending.InstCooDetrend()
#ICD_DE = detrending.InstCooDetrend()
save_dtr_ra = {}
save_dtr_de = {}
for targ in proc_objs:
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Target: %s\n" % targ)
    this_ICD_RA = detrending.InstCooDetrend()
    this_ICD_DE = detrending.InstCooDetrend()
    this_ICD_RA.reset()
    this_ICD_DE.reset()
    others = [x for x in trend_targlist if x!= targ]

    # load object data into detrender:
    this_fit = save_fitters[targ]
    this_jdtdb = this_fit.dataset['jdtdb']
    this_instr = this_fit.dataset['instrument']
    ra_errs_mas, de_errs_mas = \
            this_fit.get_radec_minus_model_mas(cos_dec_mult=True)
    this_ICD_RA.set_data(this_jdtdb, ra_errs_mas, this_instr)
    this_ICD_DE.set_data(this_jdtdb, de_errs_mas, this_instr)
    # load trend data into detrender:
    for trtarg in others:
        tr_jdtdb, tr_ra_errs, tr_de_errs, tr_inst = trend_resid_vecs[trtarg]
        this_ICD_RA.add_trend(trtarg, tr_jdtdb, tr_ra_errs, tr_inst)
        this_ICD_DE.add_trend(trtarg, tr_jdtdb, tr_de_errs, tr_inst)
        pass
    # clean it up:
    this_ICD_RA.detrend()
    this_ICD_DE.detrend()
    # save for later:
    save_dtr_ra[targ] = this_ICD_RA
    save_dtr_de[targ] = this_ICD_DE

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## FWHM estimation using detection a,b:
def estimate_fwhm(aa, bb):
    _ln2 = np.log(2.0)
    return 2.0 * np.sqrt(_ln2 * (aa*aa + bb*bb))


## Apply adjustments:
_nom_instmag = 13.9
sys.stderr.write("Detrending input data sets ... ")
_deg_per_mas  = 1.0 / 3.6e6
save_datasets = {}
for targ in proc_objs:
    #sys.stderr.write("-------------------------\n")
    #sys.stderr.write("targ: %s\n" % targ)
    _data  = save_fitters[targ].dataset
    #sys.stderr.write("len(_data): %d\n" % len(_data))
    dtr_ra = save_dtr_ra[targ]
    dtr_de = save_dtr_de[targ]
    ra_answer = dtr_ra.get_results_all()
    de_answer = dtr_de.get_results_all()
    if not np.all(ra_answer['time'] == de_answer['time']):
        sys.stderr.write("Coordinates got mismatched! Fail ...\n")
        sys.exit(1)
    inst_vals = ra_answer.get('inst')
    #ra_resid_raw = dtr_ra._cdtr_objs
    cos_dec  = np.cos(np.radians(_data[_use_de_key]))
    #sys.stderr.write("len(cos_dec): %d\n" % len(cos_dec))
    add_fwhm = estimate_fwhm(_data['a'], _data['b'])
    fixed_ra = np.zeros_like(_data[_use_ra_key])
    fixed_de = np.zeros_like(_data[_use_de_key])
    # fill the corrected RA/DE instrument by instrument:
    for iii in np.unique(inst_vals):
        dtr_which = (inst_vals == iii)
        raw_which = (_data['instrument'] == iii)
        #sys.stderr.write("np.sum(dtr_which): %d\n" % np.sum(dtr_which))
        #sys.stderr.write("np.sum(raw_which): %d\n" % np.sum(raw_which))
        i_cos_dec = cos_dec[raw_which]
        if np.sum(dtr_which) != np.sum(raw_which):
            sys.stderr.write("MISALIGNMENT! FIX ME ...\n")
        de_adj = de_answer['filt'][dtr_which] * _deg_per_mas
        ra_adj = ra_answer['filt'][dtr_which] * _deg_per_mas / i_cos_dec
        fixed_ra[raw_which] = _data[_use_ra_key][raw_which] - ra_adj
        fixed_de[raw_which] = _data[_use_de_key][raw_which] - de_adj
    # attach the corrected RA/DE to the original dataset:
    new_data = append_fields(_data, ('fwhm', 'dtr_ra', 'dtr_de'), 
            (add_fwhm, fixed_ra, fixed_de), usemask=False)
    save_datasets[targ] = new_data
    pass
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
## Make slightly-more-sensible astrometric uncertainties:
_nom_astrsig =  20.     # mas, at median FWHM
sciAB = save_datasets['2M0249AB']
medFW = np.median(sciAB['fwhm'])
asterr = _nom_astrsig * sciAB['fwhm'] / medFW
cosdec = np.cos(np.radians(sciAB['dtr_de']))
de_err_mas = asterr
ra_err_mas = asterr / cosdec

## Save to file:
save_sciab = 'wircam_2M0249AB.csv'
grab_these = ['jdtdb', 'dtr_ra', 'dtr_de', 'obs_x', 'obs_y', 'obs_z']
with open(save_sciab, 'w') as sf:
    sf.write("jdtdb,de_deg,de_deg,err_ra_mas,err_de_mas,obsx_au,obsy_au,obsz_au\n")
    got_cols = [sciAB[x] for x in grab_these]
    ord_cols = got_cols[:3] + [ra_err_mas, de_err_mas] + got_cols[-3:]
    #for stuff in zip([sciAB[x] for x in grab_these]):
    for stuff in zip(*ord_cols):
        vals = ','.join([str(x) for x in stuff])
        sf.write("%s\n" % vals)
    pass

##--------------------------------------------------------------------------##
## Initialize NEW fitters for detrended data:
dtr_fitters = {sid:at2.AstFit() for sid in proc_objs}

## Run processing:
_dtr_ra_key = 'dtr_ra'
_dtr_de_key = 'dtr_de'
num_todo = 0
sig_thresh = 3
save_dtr_fitters  = {}
save_dtr_bestpars = {}
maxiters = 30
for targ in proc_objs:
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Initial fit of: %s\n" % targ)
    afn = at2.AstFit()
    afn.setup(save_datasets[targ], ra_key=_dtr_ra_key, de_key=_dtr_de_key)
    bestpars = afn.fit_bestpars(sigcut=sig_thresh)
    iterpars = bestpars.copy()
    for i in range(maxiters):
        sys.stderr.write("Iteration %d ...\n" % i)
        iterpars = afn.iter_update_bestpars(iterpars)
        if afn.is_converged():
            sys.stderr.write("Converged!  Iteration: %d\n" % i)
            break
    save_dtr_bestpars[targ] = afn.nice_units(iterpars)
    save_dtr_fitters[targ] = afn
    pass


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Announce bestpars for science targets:
for targ in movers:
    #data = save_fitters[targ].dataset
    data = save_dtr_fitters[targ].dataset
    inst_vec  = data['instrument']
    inst_list = np.unique(inst_vec)
    counts = {x:np.sum(inst_vec == x) for x in inst_list}
    pars = save_bestpars[targ]
    sys.stderr.write("%15s: %s\n" % (targ, str(counts)))
    sys.stderr.write("%15s %s\n" % ('-->', str(pars)))

### Raw+clean RA residuals for the target:
#sdss0805_dtr_ra = save_dtr_ra['SDSS0805']
#sdss0805_ra_resid_raw = sdss0805_dtr_ra._cdtr_objs['wircam_J']._raw_vals
#sdss0805_ra_resid_cln = sdss0805_dtr_ra.get_results()[1]
#sdss0805_ra_delta_mas = sdss0805_ra_resid_cln - sdss0805_ra_resid_raw
#
### Raw+clean DE residuals for the target:
#sdss0805_dtr_de = save_dtr_de['SDSS0805']
#sdss0805_de_resid_raw = sdss0805_dtr_de._cdtr_objs['wircam_J']._raw_vals
#sdss0805_de_resid_cln = sdss0805_dtr_de.get_results()[1]
#sdss0805_de_delta_mas = sdss0805_de_resid_cln - sdss0805_de_resid_raw




##--------------------------------------------------------------------------##

sys.exit(0)
sys.exit(0)
sys.exit(0)
sys.exit(0)
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.grid(True)
#ax1.axis('off')

## Polar scatter:
#skw = {'lw':0, 's':15}
#ax1.scatter(azm_rad, zdist_deg, **skw)

## For polar axes:
#ax1.set_rmin( 0.0)                  # if using altitude in degrees
#ax1.set_rmax(90.0)                  # if using altitude in degrees
#ax1.set_theta_direction(-1)         # counterclockwise
#ax1.set_theta_zero_location("N")    # North-up
#ax1.set_rlabel_position(-30.0)      # move labels 30 degrees

## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)

#ax1.plot(kde_pnts, kde_vals)

#ax1.pcolormesh(xx, yy, ivals)

#blurb = "some text"
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes)
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes,
#      va='top', ha='left', bbox=dict(facecolor='white', pad=10.0))
#      fontdict={'family':'monospace'}) # fixed-width
#      fontdict={'fontsize':24}) # larger typeface

#colors = cm.rainbow(np.linspace(0, 1, len(plot_list)))
#for camid, c in zip(plot_list, colors):
#    cam_data = subsets[camid]
#    xvalue = cam_data['CCDATEMP']
#    yvalue = cam_data['PIX_MED']
#    yvalue = cam_data['IMEAN']
#    ax1.scatter(xvalue, yvalue, color=c, lw=0, label=camid)

#mtickpos = [2,5,7]
#ndecades = 1.0   # for symlog, set width of linear portion in units of dex
#nonposx='mask' | nonposx='clip' | nonposy='mask' | nonposy='clip'
#ax1.set_xscale('log', basex=10, nonposx='mask', subsx=mtickpos)
#ax1.set_xscale('log', nonposx='clip', subsx=[3])
#ax1.set_yscale('symlog', basey=10, linthreshy=0.1, linscaley=ndecades)
#ax1.xaxis.set_major_formatter(formatter) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
#ax1.set_xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#for label in ax1.get_xticklabels():
#    label.set_rotation(30)
#    label.set_fontsize(14) 

#ax1.xaxis.label.set_fontsize(18)
#ax1.yaxis.label.set_fontsize(18)

#ax1.set_xlim(nice_limits(xvec, pctiles=[1,99], pad=1.2))
#ax1.set_ylim(nice_limits(yvec, pctiles=[1,99], pad=1.2))

#ax1.legend(loc='best', prop={'size':24})

#spts = ax1.scatter(x, y, lw=0, s=5)
##cbar = fig.colorbar(spts, orientation='vertical')   # old way
#cbnorm = mplcolors.Normalize(*spts.get_clim())
#scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#scm.set_array([])
#cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (31_2M0249ABc_diags.py):
#---------------------------------------------------------------------
#
#  2024-06-03:
#     -- Increased __version__ to 0.0.1.
#     -- First created 31_2M0249ABc_diags.py.
#

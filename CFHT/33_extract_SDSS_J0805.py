#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Select the J0805 data and do some detrending against neighbors to get an
# initial dataset for fit testing.
#
# Rob Siverd
# Created:       2024-04-23
# Last modified: 2024-04-23
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
#import resource
#import signal
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
#import theil_sen as ts
#import window_filter as wf
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

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

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

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

## Magnitude to flux conversion:
def kadu(mag, zeropt=25.0):
    return 10.0**(0.4 * (zeropt - mag))

## Flux to magnitude conversion:
def kmag(adu, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(adu))

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
##--------------------------------------------------------------------------##

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

##--------------------------------------------------------------------------##
## Load J detections for this field (no other filters available):

data_file_J = 'process/sdss0805_Jdet_J_fcat.pickle'

targets_J = load_pickled_object(data_file_J)

targets = targets_J

mover = targets_J['SDSS0805']

##--------------------------------------------------------------------------##
## Source counts for all associated targets:
n_sources = {kk:len(vv) for kk,vv in targets_J.items()}

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
    afn.setup(proc_data[targ], ra_key='calc_ra', de_key='calc_de')
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

## Include the fast-movers if they are not saturated:
#for targ in ['G12524A', 'G12524B']:
#    if imag_stds[targ] < max_mag_stddev:
#        trend_targlist.append(targ)

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

## Raw+clean RA residuals for the target:
sdss0805_dtr_ra = save_dtr_ra['SDSS0805']
sdss0805_ra_resid_raw = sdss0805_dtr_ra._cdtr_objs['wircam_J']._raw_vals
sdss0805_ra_resid_cln = sdss0805_dtr_ra.get_results()[1]
sdss0805_ra_delta_mas = sdss0805_ra_resid_cln - sdss0805_ra_resid_raw

## Raw+clean DE residuals for the target:
sdss0805_dtr_de = save_dtr_de['SDSS0805']
sdss0805_de_resid_raw = sdss0805_dtr_de._cdtr_objs['wircam_J']._raw_vals
sdss0805_de_resid_cln = sdss0805_dtr_de.get_results()[1]
sdss0805_de_delta_mas = sdss0805_de_resid_cln - sdss0805_de_resid_raw

uncertainty_mas = 0.025

deg_per_mas = 1.0 / 3.6e6
cosdec_val = np.cos(np.radians(mover['calc_de']))
keep_jdtdb = mover['jdtdb']
keep_radeg = mover['calc_ra'] \
        + (sdss0805_ra_delta_mas * deg_per_mas / cosdec_val)
keep_dedeg = mover['calc_de'] \
        + (sdss0805_de_delta_mas * deg_per_mas)
base_uncvec = np.ones_like(keep_jdtdb) * uncertainty_mas
#keep_err_de = base_uncvec * deg_per_mas
#keep_err_ra = base_uncvec * deg_per_mas / cosdec_val
keep_err_de = base_uncvec
keep_err_ra = base_uncvec / cosdec_val

save_file = 'clean_J0805.csv'
with open(save_file, 'w') as sf:
    sf.write("jdtdb,ra_deg,ra_err_mas,de_deg,de_err_mas\n")
    for things in zip(keep_jdtdb, keep_radeg, keep_err_ra,
                                    keep_dedeg, keep_err_de):
        dline = ','.join(['%.7f'%x for x in things])
        sf.write("%s\n" % dline)
        pass

save_data = 'clean_J0805.pickle'
sci = save_fitters['SDSS0805']
sci_data = sci.collect_result_dataset()
sci_data_dtr = append_fields(sci_data,
        ('dtr_dra', 'dtr_dde'), 
        (keep_radeg, keep_dedeg),
        usemask=False)
stash_as_pickle(save_data, sci_data_dtr)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Gather data for an RMS plot:
targ_instmag  = {}
targ_ast_rms  = {}
targ_mag_rms  = {}
targ_dra_rms_raw = {}
targ_dde_rms_raw = {}
targ_dra_rms_cln = {}
targ_dde_rms_cln = {}
for targ in proc_objs:
    #signal = proc_data[targ]['flux'] * proc_data[targ]['exptime']
    signal = proc_data[targ]['flux']
    instmag = kmag(signal)
    med_mag, iqr_mag = rs.calc_ls_med_IQR(instmag)
    targ_instmag[targ] = med_mag
    targ_mag_rms[targ] = iqr_mag
    # 'raw' astrometric scatter:
    this_fit = save_fitters[targ]
    ra_errs_mas, de_errs_mas = \
            this_fit.get_radec_minus_model_mas(cos_dec_mult=True)
    tot_ast_mas = np.hypot(ra_errs_mas, de_errs_mas)
    med_ast_err, iqr_ast_err = rs.calc_ls_med_IQR(tot_ast_mas)
    targ_ast_rms[targ] = np.median(tot_ast_mas)
    med_dra_err, sig_dra_err = rs.calc_ls_med_MAD(ra_errs_mas)
    med_dde_err, sig_dde_err = rs.calc_ls_med_MAD(de_errs_mas)
    targ_dra_rms_raw[targ] = sig_dra_err
    targ_dde_rms_raw[targ] = sig_dde_err
    # 'cln' astrometric scatter:
    this_dtr_ra = save_dtr_ra[targ]
    this_dtr_de = save_dtr_de[targ]
    clean_ra_resids = this_dtr_ra.get_results()[1]
    clean_de_resids = this_dtr_de.get_results()[1]
    med_cra_err, sig_cra_err = rs.calc_ls_med_MAD(clean_ra_resids)
    med_cde_err, sig_cde_err = rs.calc_ls_med_MAD(clean_de_resids)
    targ_dra_rms_cln[targ] = sig_cra_err
    targ_dde_rms_cln[targ] = sig_cde_err

p_instmag = [targ_instmag[x] for x in proc_objs]
p_ast_rms = [targ_ast_rms[x] for x in proc_objs]

p_dra_rms_raw = [targ_dra_rms_raw[x] for x in proc_objs]
p_dde_rms_raw = [targ_dde_rms_raw[x] for x in proc_objs]
p_dra_rms_cln = [targ_dra_rms_cln[x] for x in proc_objs]
p_dde_rms_cln = [targ_dde_rms_cln[x] for x in proc_objs]
#t_layout() # adjust boundaries sensibly, matplotlib v1.1+
#        plt.draw()
#        n proc_objs]

sys.exit(0)
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
ax2.grid(True)
#ax1.axis('off')

skw = {'lw':0, 's':25}
#ax1.scatter(p_instmag, p_ast_rms)
ax1.scatter(p_instmag, p_dra_rms_raw, label='raw', **skw)
ax1.scatter(p_instmag, p_dra_rms_cln, label='cln', **skw)
ax2.scatter(p_instmag, p_dde_rms_raw, label='raw', **skw)
ax2.scatter(p_instmag, p_dde_rms_cln, label='cln', **skw)
ax1.set_ylabel('RA RMS (mas)')
ax2.set_ylabel('DE RMS (mas)')
ax2.set_xlabel("Instrumental Mag")
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_ylim(5, 500)
ax2.set_ylim(5, 500)
ax1.legend(loc='lower right')
ax2.legend(loc='lower right')

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()





######################################################################
# CHANGELOG (33_extract_SDSS_J0805.py):
#---------------------------------------------------------------------
#
#  2024-04-23:
#     -- Increased __version__ to 0.0.1.
#     -- First created 33_extract_SDSS_J0805.py.
#

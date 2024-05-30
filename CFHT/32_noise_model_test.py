#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Stand-alone RMS plot and noise model examination script.
#
# Rob Siverd
# Created:       2023-12-19
# Last modified: 2023-12-19
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
#import glob
import gc
import os
import sys
import time
#import vaex
#import calendar
#import ephem
import numpy as np
import pickle
#from numpy.lib.recfunctions import append_fields
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
import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Astrometry fitting module:
import astrom_test_2
reload(astrom_test_2)
at2 = astrom_test_2
#af  = at2.AstFit()  # used for target
afn = at2.AstFit()  # used for neighbors
afA = at2.AstFit()  # for G125-24A
afB = at2.AstFit()  # for G125-24B

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
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Parse arguments and run script:
#class MyParser(argparse.ArgumentParser):
#    def error(self, message):
#        sys.stderr.write('error: %s\n' % message)
#        self.print_help()
#        sys.exit(2)
#
### Enable raw text AND display of defaults:
#class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
#                        argparse.RawDescriptionHelpFormatter):
#    pass
#
### Parse the command line:
#if __name__ == '__main__':
#
#    # ------------------------------------------------------------------
#    prog_name = os.path.basename(__file__)
#    descr_txt = """
#    PUT DESCRIPTION HERE.
#    
#    Version: %s
#    """ % __version__
#    parser = argparse.ArgumentParser(
#            prog='PROGRAM_NAME_HERE',
#            prog=os.path.basename(__file__),
#            #formatter_class=argparse.RawTextHelpFormatter)
#            description='PUT DESCRIPTION HERE.')
#            #description=descr_txt)
#    parser = MyParser(prog=prog_name, description=descr_txt)
#                          #formatter_class=argparse.RawTextHelpFormatter)
#    # ------------------------------------------------------------------
#    parser.set_defaults(thing1='value1', thing2='value2')
#    # ------------------------------------------------------------------
#    parser.add_argument('firstpos', help='first positional argument')
#    parser.add_argument('-w', '--whatever', required=False, default=5.0,
#            help='some option with default [def: %(default)s]', type=float)
#    parser.add_argument('-s', '--site',
#            help='Site to retrieve data for', required=True)
#    parser.add_argument('-n', '--number_of_days', default=1,
#            help='Number of days of data to retrieve.')
#    parser.add_argument('-o', '--output_file', 
#            default='observations.csv', help='Output filename.')
#    parser.add_argument('--start', type=str, default=None, 
#            help="Start time for date range query.")
#    parser.add_argument('--end', type=str, default=None,
#            help="End time for date range query.")
#    parser.add_argument('-d', '--dayshift', required=False, default=0,
#            help='Switch between days (1=tom, 0=today, -1=yest', type=int)
#    parser.add_argument('-e', '--encl', nargs=1, required=False,
#            help='Encl to make URL for', choices=all_encls, default=all_encls)
#    parser.add_argument('-s', '--site', nargs=1, required=False,
#            help='Site to make URL for', choices=all_sites, default=all_sites)
#    parser.add_argument('remainder', help='other stuff', nargs='*')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    #iogroup = parser.add_argument_group('File I/O')
#    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
#    #        help='Output filename', type=str)
#    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
#    #        help='KELT image with WCS')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    ofgroup = parser.add_argument_group('Output format')
#    fmtparse = ofgroup.add_mutually_exclusive_group()
#    fmtparse.add_argument('--python', required=False, dest='output_mode',
#            help='Return Python dictionary with results [default]',
#            default='pydict', action='store_const', const='pydict')
#    bash_var = 'ARRAY_NAME'
#    bash_msg = 'output Bash code snippet (use with eval) to declare '
#    bash_msg += 'an associative array %s containing results' % bash_var
#    fmtparse.add_argument('--bash', required=False, default=None,
#            help=bash_msg, dest='bash_array', metavar=bash_var)
#    fmtparse.set_defaults(output_mode='pydict')
#    # ------------------------------------------------------------------
#    # Miscellany:
#    miscgroup = parser.add_argument_group('Miscellany')
#    miscgroup.add_argument('--debug', dest='debug', default=False,
#            help='Enable extra debugging messages', action='store_true')
#    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
#            help='less progress/status reporting')
#    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
#            help='more progress/status reporting')
#    # ------------------------------------------------------------------
#
#    context = parser.parse_args()
#    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
#    context.prog_name = prog_name
#
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Load data for quick analysis:

#data_file_H2 = 'process/calib1_H2_fcat.pickle'
#data_file_J  = 'process/calib1_J_fcat.pickle'
data_file_H2 = 'process/calib1_Jdet_H2_fcat.pickle'
data_file_J  = 'process/calib1_Jdet_J_fcat.pickle'
#targets = {}

sys.stderr.write("Loading data ... ")
with open(data_file_H2, 'rb') as pp:
    targets_H2 = pickle.load(pp)
with open(data_file_J, 'rb') as pp:
    targets_J = pickle.load(pp)
sys.stderr.write("done.\n")

## Merge stuff:
srcs_H2   = set(targets_H2.keys())
srcs_J    = set(targets_J.keys())
srcs_both = srcs_H2.intersection(srcs_J)
srcs_all  = set(targets_J.keys()).union(targets_H2.keys())

### Data seen in BOTH channels:
#sys.stderr.write("Concatenating and repackaging ... ")
#targets_hold = {tid:[] for tid in srcs_all}
#for tt in (targets_J, targets_H2):
#    for kk,vv in tt.items():
#        targets_hold[kk].append(vv)
##targets_both = {np.concatenate(srcs_H2[id]}
#targets_all = {tt:jdtdb_sorted(concat_recs_by_field(vv)) \
#        for tt,vv in targets_hold.items()}
##for tt,vv in targets_hold.items():
##    targets_all[tt] = np.concatenate(vv)
##targets_all = {tt:np.concatenate(vv) for tt,vv in targets_hold.items()}
#sys.stderr.write("done.\n")

targets = targets_H2
#targets = targets_J

speedies = ['G12524A', 'G12524B']

fast_A = targets['G12524A']
fast_B = targets['G12524B']


ref_jdtdb = np.median(fast_A['jdtdb'])

##--------------------------------------------------------------------------##
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

shorter_list = []
shorter_list.extend(speedies)
mag_lower = 10.
mag_upper = 20.
n_bins = 20
bin_width = (mag_upper - mag_lower) / float(n_bins)
pts_per_bin = np.zeros(n_bins)
max_per_bin = 30
max_per_bin = 5
for targ in proc_objs:
    this_mag = imag_avgs[targ]
    if this_mag >= mag_upper:
        continue    # out of range
    this_bin = imag2bin(this_mag, mag_lower, bin_width)
    if pts_per_bin[this_bin] < max_per_bin:
        shorter_list.append(targ)
        pts_per_bin[this_bin] += 1
        pass
    pass

proc_objs = shorter_list

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

## Detrend everything:
#afA = save_fitters['G12524A']
#afB = save_fitters['G12524B']

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
for targ in ['G12524A', 'G12524B']:
    if imag_stds[targ] < max_mag_stddev:
        trend_targlist.append(targ)

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


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

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

##--------------------------------------------------------------------------##

#tgtA_ICDRA = detrending.InstCooDetrend()
#tgtA_ICDDE = detrending.InstCooDetrend()
#tgtB_ICDRA = detrending.InstCooDetrend()
#tgtB_ICDDE = detrending.InstCooDetrend()
#
#data_A = afA.dataset
#data_B = afB.dataset
#jdtdb_A = data_A['jdtdb']
#jdtdb_B = data_B['jdtdb']
#pars_A = afA.get_latest_params()
#pars_B = afB.get_latest_params()
#
#resids_A = afA._calc_radec_residuals_coo(pars_A)
#resids_B = afB._calc_radec_residuals_coo(pars_B)
#
#tgtA_ICDRA.set_data(jdtdb_A, resids_A[0], data_A['instrument'])
#tgtA_ICDDE.set_data(jdtdb_A, resids_A[1], data_A['instrument'])
#tgtB_ICDRA.set_data(jdtdb_B, resids_B[0], data_B['instrument'])
#tgtB_ICDDE.set_data(jdtdb_B, resids_B[1], data_B['instrument'])
#
#tgtA_ICDRA.add_trend('B', jdtdb_B, resids_B[0], data_B['instrument'])
#tgtA_ICDDE.add_trend('B', jdtdb_B, resids_B[1], data_B['instrument'])
#tgtA_ICDRA.detrend()
#tgtA_ICDDE.detrend()
#
#tgtB_ICDRA.add_trend('A', jdtdb_A, resids_A[0], data_A['instrument'])
#tgtB_ICDDE.add_trend('A', jdtdb_A, resids_A[1], data_A['instrument'])
#tgtB_ICDRA.detrend()
#tgtB_ICDDE.detrend()
#
#
#tgtA_tvec_ra, tgtA_clean_resid_ra, tgtA_ivec_ra = tgtA_ICDRA.get_results()
#tgtA_tvec_de, tgtA_clean_resid_de, tgtA_ivec_de = tgtA_ICDDE.get_results()
#
#tgtB_tvec_ra, tgtB_clean_resid_ra, tgtB_ivec_ra = tgtB_ICDRA.get_results()
#tgtB_tvec_de, tgtB_clean_resid_de, tgtB_ivec_de = tgtB_ICDDE.get_results()



##--------------------------------------------------------------------------##
## Plot config:

# gridspec examples:
# https://matplotlib.org/users/gridspec.html

#gs1 = gridspec.GridSpec(4, 4)
#gs1.update(wspace=0.025, hspace=0.05)  # set axis spacing

#ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3) # top-left + center + right
#ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2) # mid-left + mid-center
#ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) # mid-right + bot-right
#ax4 = plt.subplot2grid((3, 3), (2, 0))            # bot-left
#ax5 = plt.subplot2grid((3, 3), (2, 1))            # bot-center


##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
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

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (32_noise_model_test.py):
#---------------------------------------------------------------------
#
#  2023-12-19:
#     -- Increased __version__ to 0.0.1.
#     -- First created 32_noise_model_test.py.
#

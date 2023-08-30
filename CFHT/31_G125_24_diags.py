#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Some quick diagnostics using G125-24A and G125-24B.
#
# Rob Siverd
# Created:       2023-07-27
# Last modified: 2023-07-27
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
#import resource
#import signal
#import glob
import gc
import os
import ast
import sys
import time
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
#import seaborn as sns
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
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
## Load data for quick analysis:

data_file_H2 = 'process/calib1_H2_fcat.pickle'
data_file_J  = 'process/calib1_J_fcat.pickle'
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

## Initialize fitters:
fitters = {sid:at2.AstFit() for sid in proc_objs}

## Run processing:
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
## Settings:
#sig_thresh = 3
#
### Try an astrometric fit for A component:
#afA.setup(fast_A, ra_key='calc_ra', de_key='calc_de')
#bestpars_A = afA.fit_bestpars(sigcut=sig_thresh)
#initpars_A = bestpars_A.copy()
#
### Iterate to an improved solution:
#iterpars_A = afA.iter_update_bestpars(bestpars_A)
#for i in range(30):
#    sys.stderr.write("Iteration %d ...\n" % i)
#    iterpars_A = afA.iter_update_bestpars(iterpars_A)
#    if afA.is_converged():
#        sys.stderr.write("Converged!  Iteration: %d\n" % i)
#        break
#bestpars_A = iterpars_A
#pa_jd = afA.dataset['jdtdb']
#pa_ra, pa_de = np.degrees(afA.eval_model(bestpars_A))
#
#
### Try an astrometric fit for B component:
#afB.setup(fast_B, ra_key='calc_ra', de_key='calc_de')
#bestpars_B = afB.fit_bestpars(sigcut=sig_thresh)
#initpars_B = bestpars_B.copy()
#
### Iterate to an improved solution:
#iterpars_B = afB.iter_update_bestpars(bestpars_B)
#for i in range(30):
#    sys.stderr.write("Iteration %d ...\n" % i)
#    iterpars_B = afB.iter_update_bestpars(iterpars_B)
#    if afB.is_converged():
#        sys.stderr.write("Converged!  Iteration: %d\n" % i)
#        break
#bestpars_B = iterpars_B
#
## for plotting:
#pb_jd = afB.dataset['jdtdb']
#pb_ra, pb_de = np.degrees(afB.eval_model(bestpars_B))

##--------------------------------------------------------------------------##
## Misc:
#def log_10_product(x, pos):
#   """The two args are the value and tick position.
#   Label ticks with the product of the exponentiation."""
#   return '%.2f' % (x)  # floating-point
#
#formatter = plt.FuncFormatter(log_10_product) # wrap function for use

## Convenient, percentile-based plot limits:
def nice_limits(vec, pctiles=[1,99], pad=1.2):
    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
    middle = np.average(ends)
    return (middle + pad * (ends - middle))

## Convenient plot limits for datetime/astropy.Time content:
#def nice_time_limits(tvec, buffer=0.05):
#    lower = tvec.min()
#    upper = tvec.max()
#    ndays = upper - lower
#    return ((lower - 0.05*ndays).datetime, (upper + 0.05*ndays).datetime)

##--------------------------------------------------------------------------##
## Single-object plot generator:
#def plot_results(targfit, fignum=1, ref_jdtdb=None):
#    data = targfit.dataset
#    fmap = {'J':'b', 'H2':'g'}
#    fig, axs = plt.subplots(3, 2, sharex=True, num=fignum, clear=True)
#    ax1, ax2 = axs[0]
#    p_colors = [fmap[x] for x in data['filter']]
#    skw = {'lw':0, 's':5, 'c':fA_colors}
#    jd_ref = ref_jdtdb if ref_jdtdb else data['jdtdb'][0]
#    rel_jd = data['jdtdb'] - jd_ref
#
#    for ax in axs.flatten():
#        ax.yaxis.get_major_formatter().set_useOffset(False)
#    axs[0, 0].scatter(rel_jd, data['calc_ra'], **skw)
#    #fA_colors = [fmap[x] for x in fast_A['filter']]
#    #fB_colors = [fmap[x] for x in fast_B['filter']]
#    return

afA = save_fitters['G12524A']
afB = save_fitters['G12524B']

## Bring up the plotting window before scanning through everything:
fig_dims = (17, 9)
fig = plt.figure(1, figsize=fig_dims)
plt.show()

import result_plotter
reload(result_plotter)
mkplot = result_plotter.plot_ast_summary
#mkplot(afA)
#mkplot(afA, mas_lims=(-250,250))

#mkplot(afB, mas_lims=(-250,250))

mas_lims = (-250, 250)
#mas_lims = None
for targ in proc_objs:
    sys.stderr.write("Plotting results for target: %s\n" % targ)
    afobj = save_fitters[targ]
    mkplot(afobj, mas_lims=mas_lims)
    sys.stderr.write("Plotting done, press ENTER to continue, 'q' to stop\n")
    derp = input()
    if derp == 'q':
        break

#stuff = save_fitters['-65.3135466+35.2829209'].dataset

ra_res, de_res = save_fitters['-65.3135466+35.2829209'].get_radec_minus_model_mas()
verybad = ~rs.pick_inliers(ra_res, 3)

#badpts = stuff[verybad]

tgtA_ICDRA = detrending.InstCooDetrend()
tgtA_ICDDE = detrending.InstCooDetrend()
tgtB_ICDRA = detrending.InstCooDetrend()
tgtB_ICDDE = detrending.InstCooDetrend()

data_A = afA.dataset
data_B = afB.dataset
jdtdb_A = data_A['jdtdb']
jdtdb_B = data_B['jdtdb']
pars_A = afA.get_latest_params()
pars_B = afB.get_latest_params()

resids_A = afA._calc_radec_residuals_coo(pars_A)
resids_B = afB._calc_radec_residuals_coo(pars_B)

tgtA_ICDRA.set_data(jdtdb_A, resids_A[0], data_A['instrument'])
tgtA_ICDDE.set_data(jdtdb_A, resids_A[1], data_A['instrument'])
tgtB_ICDRA.set_data(jdtdb_B, resids_B[0], data_B['instrument'])
tgtB_ICDDE.set_data(jdtdb_B, resids_B[1], data_B['instrument'])

tgtA_ICDRA.add_trend('B', jdtdb_B, resids_B[0], data_B['instrument'])
tgtA_ICDDE.add_trend('B', jdtdb_B, resids_B[1], data_B['instrument'])
tgtA_ICDRA.detrend()
tgtA_ICDDE.detrend()

tgtB_ICDRA.add_trend('A', jdtdb_A, resids_A[0], data_A['instrument'])
tgtB_ICDDE.add_trend('A', jdtdb_A, resids_A[1], data_A['instrument'])
tgtB_ICDRA.detrend()
tgtB_ICDDE.detrend()


tgtA_tvec_ra, tgtA_clean_resid_ra, tgtA_ivec_ra = tgtA_ICDRA.get_results()
tgtA_tvec_de, tgtA_clean_resid_de, tgtA_ivec_de = tgtA_ICDDE.get_results()

tgtB_tvec_ra, tgtB_clean_resid_ra, tgtB_ivec_ra = tgtB_ICDRA.get_results()
tgtB_tvec_de, tgtB_clean_resid_de, tgtB_ivec_de = tgtB_ICDDE.get_results()

sys.exit(0)

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (17, 9)
fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1)
ax3 = fig.add_subplot(223, sharex=ax1)
ax4 = fig.add_subplot(224, sharex=ax1)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax_list = [ax1, ax2, ax3, ax4]

## Figure out some sensible limits:
#np.percentile(fast_A['calc_ra'], [2, 98])

fmap = {'J':'b', 'H2':'g'}
fA_colors = [fmap[x] for x in fast_A['filter']]
fB_colors = [fmap[x] for x in fast_B['filter']]

skw = {'lw':0, 's':5, 'c':fA_colors}
ax1.scatter(fast_A['jdtdb'] - ref_jdtdb, fast_A['calc_ra'], **skw)
ax1.plot(pa_jd - ref_jdtdb, pa_ra, c='r')
ax1.set_ylim(nice_limits(fast_A['calc_ra'], [5, 95], 2.0))
ax2.scatter(fast_A['jdtdb'] - ref_jdtdb, fast_A['calc_de'], **skw)
ax2.plot(pa_jd - ref_jdtdb, pa_de, c='r')

skw['c'] = fB_colors
ax3.scatter(fast_B['jdtdb'] - ref_jdtdb, fast_B['calc_ra'], **skw)
ax3.plot(pb_jd - ref_jdtdb, pb_ra, c='r')
ax3.set_ylim(nice_limits(fast_B['calc_ra'], [5, 95], 2.0))
ax4.scatter(fast_B['jdtdb'] - ref_jdtdb, fast_B['calc_de'], **skw)
ax4.plot(pb_jd - ref_jdtdb, pb_de, c='r')


for ax in ax_list:
    ax.yaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)
#ax1.grid(True)
#ax1.axis('off')

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



######################################################################
# CHANGELOG (31_G125_24_diags.py):
#---------------------------------------------------------------------
#
#  2023-07-27:
#     -- Increased __version__ to 0.0.1.
#     -- First created 31_G125_24_diags.py.
#

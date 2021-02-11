#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Look at residuals from my first attempt at fitting.
#
# Rob Siverd
# Created:       2020-02-10
# Last modified: 2020-02-10
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

## Optional matplotlib control:
#from matplotlib import use, rc, rcParams
#from matplotlib import use
#from matplotlib import rc
#from matplotlib import rcParams
#use('GTKAgg')  # use GTK with Anti-Grain Geometry engine
#use('agg')     # use Anti-Grain Geometry engine (file only)
#use('ps')      # use PostScript engine for graphics (file only)
#use('cairo')   # use Cairo (pretty, file only)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('font',**{'sans-serif':'Arial','family':'sans-serif'})
#rc('text', usetex=True) # enables text rendering with LaTeX (slow!)
#rcParams['axes.formatter.useoffset'] = False   # v. 1.4 and later
#rcParams['agg.path.chunksize'] = 10000
#rcParams['font.size'] = 10

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
import pickle
import gc
import os
import sys
import time
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
import matplotlib.pyplot as plt
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

import fluxmag
reload(fluxmag)

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
## Colors for fancy terminal output:
NRED    = '\033[0;31m'   ;  BRED    = '\033[1;31m'
NGREEN  = '\033[0;32m'   ;  BGREEN  = '\033[1;32m'
NYELLOW = '\033[0;33m'   ;  BYELLOW = '\033[1;33m'
NBLUE   = '\033[0;34m'   ;  BBLUE   = '\033[1;34m'
NMAG    = '\033[0;35m'   ;  BMAG    = '\033[1;35m'
NCYAN   = '\033[0;36m'   ;  BCYAN   = '\033[1;36m'
NWHITE  = '\033[0;37m'   ;  BWHITE  = '\033[1;37m'
ENDC    = '\033[0m'

## Suppress colors in cron jobs:
if (os.getenv('FUNCDEF') == '--nocolors'):
    NRED    = ''   ;  BRED    = ''
    NGREEN  = ''   ;  BGREEN  = ''
    NYELLOW = ''   ;  BYELLOW = ''
    NBLUE   = ''   ;  BBLUE   = ''
    NMAG    = ''   ;  BMAG    = ''
    NCYAN   = ''   ;  BCYAN   = ''
    NWHITE  = ''   ;  BWHITE  = ''
    ENDC    = ''

## Fancy text:
degree_sign = u'\N{DEGREE SIGN}'

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

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
## Load pickled data:
export_file = 'resid_and_gse.pickle'
target_file = 'target_data.pickle'
if os.path.isfile(export_file):
    with open(export_file, 'rb') as ef:
        res_data, gse_data, centroid_method = pickle.load(ef)
    sys.stderr.write("Centroid method: %s\n" % centroid_method) 
else:
    sys.stderr.write("File not found: '%s'\n" % export_file)

## Load target info if available:
if os.path.isfile(target_file):
    with open(target_file, 'rb') as tf:
        tgt_ccat = pickle.load(tf)
else:
    sys.stderr.write("File not found: '%s'\n" % target_file)

## Separate target from riffraff:
#if 'tgt' in res_data.keys():
#    targ_res = res_data.pop('tgt')

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

ccurve_kind = 'gapped'
ccurve_kind = 'filled'
ccurve_kind = 'clipped'

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

def replace_outliers(residuals, cutoff, fillval=0.0):
    result = residuals.copy()
    crappy = (residuals < -1.0*cutoff) | (cutoff < residuals)
    result[crappy] = fillval
    return result


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
_mas_per_radian = 180. * 3.6e6 / np.pi

## Find all distinct JD (TDB):
unique_tdb = np.unique(np.concatenate([x['jdtdb'] for x in res_data.values()]))
tdb_index = np.arange(unique_tdb.size)
idx2tdb = dict(zip(tdb_index, unique_tdb))
tdb2idx = dict(zip(unique_tdb, tdb_index))

gid_list = list(res_data.keys())
if 'tgt' in gid_list:
    gid_list.remove('tgt')
    gid_list = ['tgt'] + gid_list

## Add a fictitious magnitude to everything:
for thing in gid_list:
    avgflx = np.average(res_data[thing]['flux'])
    avgmag = fluxmag.kmag(avgflx)
    sys.stderr.write("avgmag: %f\n" % avgmag)
    res_data[thing]['flx'] = avgflx
    res_data[thing]['mag'] = avgmag



vec_sizes = {gg:rr['jdtdb'].size for gg,rr in res_data.items()}
ra_MAD, de_MAD = {}, {}

p_res_RA, p_res_DE, p_jd_idx = {}, {}, {}
f_res_RA, f_res_DE, f_jd_idx = {}, {}, {}
c_res_RA, c_res_DE, c_jd_idx = {}, {}, {}

sig_cut = 3
for gg,res in res_data.items():
    res_ra_med, res_ra_mad = rs.calc_ls_med_MAD(res['resid_ra'])
    res_de_med, res_de_mad = rs.calc_ls_med_MAD(res['resid_de'])
    ra_MAD[gg] = res_ra_mad * _mas_per_radian
    de_MAD[gg] = res_de_mad * _mas_per_radian
    p_res_RA[gg] = (res['resid_ra'] - res_ra_med) * _mas_per_radian
    p_res_DE[gg] = (res['resid_de'] - res_de_med) * _mas_per_radian
    p_jd_idx[gg] = np.array([tdb2idx[t] for t in res['jdtdb']])

    # filled versions:
    f_res_RA[gg] = 0.0 * tdb_index
    f_res_DE[gg] = 0.0 * tdb_index
    f_jd_idx[gg] = tdb_index
    for ii,rr,dd in zip(p_jd_idx[gg], p_res_RA[gg], p_res_DE[gg]):
        f_res_RA[gg][ii] = rr
        f_res_DE[gg][ii] = dd

    # clipped versions:
    c_jd_idx[gg] = tdb_index
    c_res_RA[gg] = replace_outliers(f_res_RA[gg], sig_cut * ra_MAD[gg], 0.0)
    c_res_DE[gg] = replace_outliers(f_res_DE[gg], sig_cut * de_MAD[gg], 0.0)


fakemag = np.array([res_data[x]['mag'] for x in gid_list])
fakerms_ra = np.array([ra_MAD[x] for x in gid_list])
fakerms_de = np.array([de_MAD[x] for x in gid_list])
fakerms = np.hypot(fakerms_ra, fakerms_de)

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (12, 9)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(211)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)

ax2 = fig.add_subplot(212)
ax2.grid(True)

#filling = True

# UNFILLED:
plot_tag = ccurve_kind
if (ccurve_kind == 'gapped'):
    spacing = 400.0     # spacing between curves
    pkw = {'lw':0, 's':25}
    for ii,gid in enumerate(gid_list):
        res = res_data[gid]
        offset = spacing * (ii + 0.5)
        xvals = res['jdtdb']
        xvals = p_jd_idx[gid]
        #ax1.scatter(xvals, offset + p_res_RA[gid], **pkw)
        #ax2.scatter(xvals, offset + p_res_DE[gid], **pkw)
        ax1.plot(xvals, offset + p_res_RA[gid])
        ax2.plot(xvals, offset + p_res_DE[gid])

if (ccurve_kind == 'filled'):
    spacing = 400.0     # spacing between curves
    pkw = {'lw':0, 's':25}
    for ii,gid in enumerate(gid_list):
        offset = spacing * (ii + 0.5)
        xvals = tdb_index
        #ax1.scatter(xvals, offset + p_res_RA[gid], **pkw)
        #ax2.scatter(xvals, offset + p_res_DE[gid], **pkw)
        ax1.plot(xvals, offset + f_res_RA[gid])
        ax2.plot(xvals, offset + f_res_DE[gid])

if (ccurve_kind == 'clipped'):
    plot_tag += '_%dsig' % sig_cut
    spacing = 400.0     # spacing between curves
    pkw = {'lw':0, 's':25}
    for ii,gid in enumerate(gid_list):
        offset = spacing * (ii + 0.5)
        xvals = tdb_index
        #ax1.scatter(xvals, offset + p_res_RA[gid], **pkw)
        #ax2.scatter(xvals, offset + p_res_DE[gid], **pkw)
        ax1.plot(xvals, offset + c_res_RA[gid])
        ax2.plot(xvals, offset + c_res_DE[gid])
    ax1.set_title("%d-sigma clipped" % sig_cut)

# FILLED:

ax1.set_ylim(-0.5*spacing, (len(gid_list)+0.5)*spacing)
ax2.set_ylim(-0.5*spacing, (len(gid_list)+0.5)*spacing)

ax1.set_ylabel('RA residual (mas)')
ax2.set_ylabel('DE residual (mas)')
ax2.set_xlabel('Data Point Number')

## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)

#ax1.plot(kde_pnts, kde_vals)

#blurb = "some text"
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes)
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes,
#      va='top', ha='left', bbox=dict(facecolor='white', pad=10.0))
#      fontdict={'family':'monospace'}) # fixed-width

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

#spts = ax1.scatter(x, y, lw=0, s=5)
##cbar = fig.colorbar(spts, orientation='vertical')   # old way
#cbnorm = mplcolors.Normalize(*spts.get_clim())
#sm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#sm.set_array([])
#cbar = fig.colorbar(sm, orientation='vertical')
#cbar = fig.colorbar(sm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

plot_name = 'correl_resid_%s_%s.png' % (centroid_method, plot_tag)
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')


######################################################################
# CHANGELOG (inspect_residuals.py):
#---------------------------------------------------------------------
#
#  2020-02-10:
#     -- Increased __version__ to 0.0.1.
#     -- First created inspect_residuals.py.
#

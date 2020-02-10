#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Illustrate data and fitting for this object.
#
# Rob Siverd
# Created:       2020-02-09
# Last modified: 2020-02-09
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
import scipy.optimize as opti
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
import theil_sen as ts
#import itertools as itt
import pickle
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import angle
reload(angle)

import astrom_test
reload(astrom_test)
af = astrom_test.AstFit()
eee = astrom_test.SSTEph()


##--------------------------------------------------------------------------##
## Projections with cartopy:
#try:
#    import cartopy.crs as ccrs
#    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#    #from cartopy.feature.nightshade import Nightshade
#    #from cartopy import config as cartoconfig
#except ImportError:
#    sys.stderr.write("Error: cartopy module not found!\n")
#    sys.exit(1)

##--------------------------------------------------------------------------##

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

##--------------------------------------------------------------------------##
## Reload pickled data:
savefile = 'GSE_tuple.pickle'
if not os.path.isfile(savefile):
    sys.stderr.write("Save-file not found: '%s'\n" % savefile)
    sys.exit(1)

with open(savefile, 'rb') as f:
    gneat, sneat, use_eph = pickle.load(f)

_ra_key, _de_key =  'dra',  'dde'
use_epoch_tdb = 2456712.3421157757
use_epoch_tdb = 2457174.500000000

# ----------------
sys.stderr.write("Gaia info:\n\n")
sys.stderr.write("RA:       %15.7f\n" % gneat['ra'])
sys.stderr.write("DE:       %15.7f\n" % gneat['dec'])
sys.stderr.write("parallax: %10.4f +/- %8.4f\n"
        % (gneat.parallax, gneat.parallax_error))
sys.stderr.write("pmRA:     %10.4f +/- %8.4f\n"
        % (gneat.pmra, gneat.pmra_error))
sys.stderr.write("pmDE:     %10.4f +/- %8.4f\n"
        % (gneat.pmdec, gneat.pmdec_error))

##--------------------------------------------------------------------------##
## Theil-Sen fitting:
sjd_utc, sra, sde = sneat['jdutc'], sneat[_ra_key], sneat[_de_key]
syr = 2000.0 + ((sjd_utc - 2451544.5) / 365.25)
smonth = (syr % 1.0) * 12.0

ts_ra_model = ts.linefit(syr, sra)
ts_de_model = ts.linefit(syr, sde)
ts_pmra_masyr = ts_ra_model[1] * 3.6e6 / np.cos(np.radians(ts_de_model[0]))
ts_pmde_masyr = ts_de_model[1] * 3.6e6


# initial RA/Dec guess:
sys.stderr.write("%s\n" % fulldiv)
guess_ra = sra.mean()
guess_de = sde.mean()
sys.stderr.write("guess_ra:  %15.7f\n" % guess_ra)
sys.stderr.write("guess_de:  %15.7f\n" % guess_de)

afpars = [guess_ra, guess_de, ts_pmra_masyr/1e3, ts_pmde_masyr/1e3, 1.0]
appcoo = af.apparent_radec(use_epoch_tdb, afpars, use_eph)

# proper fit:
design_matrix = np.column_stack((np.ones(syr.size), syr))
#de_design_matrix = np.column_stack((np.ones(syr.size), syr))
ra_ols_res = sm.OLS(sra, design_matrix).fit()
de_ols_res = sm.OLS(sde, design_matrix).fit()
ra_rlm_res = sm.RLM(sra, design_matrix).fit()
de_rlm_res = sm.RLM(sde, design_matrix).fit()
rlm_pmde_masyr = de_rlm_res.params[1] * 3.6e6
rlm_pmra_masyr = ra_rlm_res.params[1] * 3.6e6 \
        * np.cos(np.radians(de_rlm_res.params[0]))


sys.stderr.write("\nTheil-Sen intercepts:\n")
sys.stderr.write("RA:   %15.7f\n" % ts_ra_model[0])
sys.stderr.write("DE:   %15.7f\n" % ts_de_model[0])

sys.stderr.write("\nTheil-Sen proper motions:\n")
sys.stderr.write("RA:   %10.6f mas/yr\n" % ts_pmra_masyr)
sys.stderr.write("DE:   %10.6f mas/yr\n" % ts_pmde_masyr)

sys.stderr.write("\nRLM (Huber) proper motions:\n")
sys.stderr.write("RA:   %10.6f mas/yr\n" % rlm_pmra_masyr)
sys.stderr.write("DE:   %10.6f mas/yr\n" % rlm_pmde_masyr)

sys.stderr.write("\n")
sys.stderr.write("%s\n" % fulldiv)

bfde_path = de_rlm_res.params[0] + de_rlm_res.params[1]*syr
bfra_path = ra_rlm_res.params[0] + ra_rlm_res.params[1]*syr

##--------------------------------------------------------------------------##

af.setup(use_epoch_tdb, sra, sde, use_eph)
af.set_exponent(2.0)
winner = af.fit_bestpars(sigcut=5.0)
win_ra, win_de = af.eval_model(winner)
win_ra = np.degrees(win_ra)
win_de = np.degrees(win_de)

rsig_RA, rsig_DE = af._calc_radec_residuals_sigma(af._par_guess)

sys.stderr.write("\n")
sys.stderr.write("%s\n" % fulldiv)

# TO INSPECT:
# plotgsource(3192149715934444800); plt.plot(bfra_path, bfde_path)
sys.stderr.write("NOTE TO SELF: Gaia pmRA includes cos(dec)!\n")

##--------------------------------------------------------------------------##
## Plottable curves for the different proper motion fits:
padding = 1.1
syr_range = syr.max() - syr.min()
#syr_lspan = syr_range * 1.1
syr_plt_1 = syr.min() - padding * syr_range
syr_plt_2 = syr.min() + padding * syr_range
#syr_edges = np.array([syr_plt_1, syr_plt_2])
syr_edges = np.array([syr.min(), syr.max()])
p_ols_ra = ra_ols_res.params[0] + syr_edges * ra_ols_res.params[1]
p_ols_de = de_ols_res.params[0] + syr_edges * de_ols_res.params[1]
p_rlm_ra = ra_rlm_res.params[0] + syr_edges * ra_rlm_res.params[1]
p_rlm_de = de_rlm_res.params[0] + syr_edges * de_rlm_res.params[1]
p_sen_ra = ts_ra_model[0] + syr_edges * ts_ra_model[1]
p_sen_de = ts_de_model[0] + syr_edges * ts_de_model[1]

p_gaia_ra = gneat['ra'].values + syr_edges * gneat['pmra'].values
p_gaia_de = gneat['dec'].values + syr_edges * gneat['pmdec'].values

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

## Convenient limits for datetime objects:
#def dt_limits(vec, pad=0.1):
#    tstart, tstop = vec.min(), vec.max()
#    trange = (tstop - tstart).total_seconds()
#    tpad = dt.timedelta(seconds=pad*trange)
#    return (tstart - tpad, tstop + tpad)

##--------------------------------------------------------------------------##
## Solve prep:
#ny, nx = img_vals.shape
#x_list = (0.5 + np.arange(nx)) / nx - 0.5            # relative (centered)
#y_list = (0.5 + np.arange(ny)) / ny - 0.5            # relative (centered)
#xx, yy = np.meshgrid(x_list, y_list)                 # relative (centered)
#xx, yy = np.meshgrid(nx*x_list, ny*y_list)           # absolute (centered)
#xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))   # absolute
#yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij') # absolute
#yy, xx = np.nonzero(np.ones_like(img_vals))          # absolute
#yy, xx = np.mgrid[0:ny,   0:nx].astype('uint16')     # absolute (array)
#yy, xx = np.mgrid[1:ny+1, 1:nx+1].astype('uint16')   # absolute (pixel)

## 1-D vectors:
#x_pix, y_pix, ivals = xx.flatten(), yy.flatten(), img_vals.flatten()
#w_vec = np.ones_like(ivals)            # start with uniform weights
#design_matrix = np.column_stack((np.ones(x_pix.size), x_pix, y_pix))

## Image fitting (statsmodels etc.):
#data = sm.datasets.stackloss.load()
#ols_res = sm.OLS(ivals, design_matrix).fit()
#rlm_res = sm.RLM(ivals, design_matrix).fit()
#rlm_model = sm.RLM(ivals, design_matrix, M=sm.robust.norms.HuberT())
#rlm_res = rlm_model.fit()
#data = pd.DataFrame({'xpix':x_pix, 'ypix':y_pix})
#rlm_model = sm.RLM.from_formula("ivals ~ xpix + ypix", data)

##--------------------------------------------------------------------------##
## Theil-Sen line-fitting (linear):
#model = ts.linefit(xvals, yvals)
#icept, slope = ts.linefit(xvals, yvals)

## Theil-Sen line-fitting (loglog):
#xvals, yvals = np.log10(original_xvals), np.log10(original_yvals)
#xvals, yvals = np.log10(df['x'].values), np.log10(df['y'].values)
#llmodel = ts.linefit(np.log10(xvals), np.log10(yvals))
#icept, slope = ts.linefit(xvals, yvals)
#fit_exponent = slope
#fit_multiplier = 10**icept
#bestfit_x = np.arange(5000)
#bestfit_y = fit_multiplier * bestfit_x**fit_exponent

## Log-log evaluator:
#def loglog_eval(xvals, model):
#    icept, slope = model
#    return 10**icept * xvals**slope
#def loglog_eval(xvals, icept, slope):
#    return 10**icept * xvals**slope

##--------------------------------------------------------------------------##
## KDE:
#kde_pnts, kde_vals = mk.go(data_vec)

##--------------------------------------------------------------------------##
## Vaex plotting:
#ds = vaex.open('big_file.hdf5')
#ds = vaex.from_arrays(x=x, y=y)     # load from arrays
#ds = vaex.from_csv('mydata.csv')

## Stats:
#ds.mean("x"), ds.std("x"), ds.correlation("vx**2+vy**2+vz**2", "E")
#ds.plot(....)
#http://vaex.astro.rug.nl/latest/tutorial_ipython_notebook.html


##--------------------------------------------------------------------------##
#plain_crs = ccrs.PlateCarree()
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (12, 10)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111, aspect='equal', projection=plain_crs)
ax1 = fig.add_subplot(111, aspect='equal')
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
#ax1.axis('off')

ax1.scatter(sra, sde, lw=0, s=10)
afpars = [np.radians(guess_ra), np.radians(guess_de),
               ts_pmra_masyr, ts_pmde_masyr, 1.0]
pxvals = np.array([0.1, 1., 10., 100.])
pxvals = np.array([1., 10.])
#for plx in pxvals:
#    afpars[4] = plx
#    tra, tde = af.apparent_radec(use_epoch_tdb, afpars, use_eph)
#    dra, dde = np.degrees(tra), np.degrees(tde)
#    ax1.scatter(dra, dde, lw=0, s=15, label='%7.4f'%plx)
#ax1.legend(loc='best')

#afpars[4] = 0.0
#line_ra, line_de = af.apparent_radec(use_epoch_tdb, afpars, use_eph)
#icept, slope = ts.linefit(line_ra, line_de)
#ra_extrema = line_ra.min(), line_ra.max()
#ra_range = line_ra.max() - line_ra.min()
#plot_ra = np.array([line_ra.min() - 0.1*ra_range,
#                    line_ra.max() + 0.1*ra_range])
#plot_de = icept + slope * plot_ra
#ax1.plot(np.degrees(plot_ra), np.degrees(plot_de), c='r', ls='--', lw=1,
#        label='Theil-Sen')

ax1.plot(p_sen_ra, p_sen_de, c='r', ls='-', lw=1, label='Theil-Sen')
ax1.plot(p_rlm_ra, p_rlm_de, c='g', ls='-', lw=1, label='RLM (Huber)')
ax1.plot(p_ols_ra, p_ols_de, c='k', ls='-', lw=1, label='OLS')

ax1.set_xlim(nice_limits(sra, pctiles=[0,100], pad=1.2))
ax1.set_ylim(nice_limits(sde, pctiles=[0,100], pad=1.2))
#ax1.legend(loc='best')


ax1.plot(win_ra, win_de, c='c', lw=1, label='optimized')
ax1.legend(loc='lower left')


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

#spts = ax1.scatter(x, y, lw=0, s=5)
##cbar = fig.colorbar(spts, orientation='vertical')   # old way
#cbnorm = mplcolors.Normalize(*spts.get_clim())
#sm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#sm.set_array([])
#cbar = fig.colorbar(sm, orientation='vertical')
#cbar = fig.colorbar(sm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (illustrate.py):
#---------------------------------------------------------------------
#
#  2020-02-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created illustrate.py.
#

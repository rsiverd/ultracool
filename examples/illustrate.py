#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Illustrate data and fitting for this object.
#
# Rob Siverd
# Created:       2020-02-09
# Last modified: 2020-02-13
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
__version__ = "0.3.0"

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
import argparse
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
import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
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

sys.stderr.write("Loading hourly Spitzer ephemeris ... ")
import horiz_ephem
reload(horiz_ephem)
he = horiz_ephem.HorizEphem()
sst_hourly_file = '/home/rsiverd/Spitzer/timekeeping/hourly/concat_sst_hourly.txt'
he.load_ascii_ephemeris(sst_hourly_file)
sys.stderr.write("done.\n")

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
try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
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
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

## Enable raw text AND display of defaults:
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
    pass

## Parse the command line:
if __name__ == '__main__':

    # ------------------------------------------------------------------
    prog_name = os.path.basename(__file__)
    descr_txt = """
    Fit 5-parameter solution for a single object and inspect results.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    parser.set_defaults(pos_method='simple')
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    fitgroup = parser.add_argument_group('Fitting')
    fitgroup.add_argument('-W', '--window', required=False,
            dest='pos_method', action='store_const', const='window',
            help='use windowed position measurements')
    fitgroup.add_argument('-S', '--simple', required=False,
            dest='pos_method', action='store_const', const='simple',
            help='use simple (non-windowed) position measurements')
    fitgroup.add_argument('-P', '--pixphase', required=False,
            dest='pos_method', action='store_const', const='pp_fix',
            help='use pixel phase-corrected position (testing)')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-o', '--output_file', default='scatter_and_fits.png',
            help='output plot filename [def: %(default)s]')
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Miscellany:
    miscgroup = parser.add_argument_group('Miscellany')
    miscgroup.add_argument('--debug', dest='debug', default=False,
            help='Enable extra debugging messages', action='store_true')
    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
            help='less progress/status reporting')
    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
            help='more progress/status reporting')
    # ------------------------------------------------------------------

    context = parser.parse_args()
    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
    context.prog_name = prog_name

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## RA/DE coordinate keys for various methods:
centroid_colmap = {
        'simple'    :   ('dra', 'dde'),
        'window'    :   ('wdra', 'wdde'),
        'pp_fix'    :   ('ppdra', 'ppdde'),
        }

## Stop in case of method confusion:
if not context.pos_method in centroid_colmap.keys():
    sys.stderr.write("unrecognized method: '%s'\n" % context.pos_method)
    sys.exit(1)
_ra_key, _de_key = centroid_colmap.get(context.pos_method)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Reload pickled data:
savefile = 'GSE_tuple.pickle'
if not os.path.isfile(savefile):
    sys.stderr.write("Save-file not found: '%s'\n" % savefile)
    sys.exit(1)

with open(savefile, 'rb') as f:
    gneat, sneat, use_eph = pickle.load(f)

_have_gaia = True if isinstance(gneat, pd.DataFrame) else False

## Set coord keys:
#if context.pos_kind:
#    _ra_key, _de_key =  'wdra',  'wdde'
#else:
#    _ra_key, _de_key =  'dra',  'dde'
#use_epoch_tdb = 2456712.3421157757
#use_epoch_tdb = 2457174.500000000
use_epoch = astt.Time(2457174.500000000, scale='tdb', format='jd')
use_epoch_tdb = use_epoch.tdb.jd
tstamps = astt.Time(use_eph['jdtdb'], scale='tdb', format='jd')

# ----------------
if _have_gaia:
    sys.stderr.write("Gaia info:\n\n")
    sys.stderr.write("RA:       %15.7f\n" % gneat['ra'])
    sys.stderr.write("DE:       %15.7f\n" % gneat['dec'])
    sys.stderr.write("pmRA:     %10.4f +/- %8.4f\n"
            % (gneat.pmra, gneat.pmra_error))
    sys.stderr.write("pmDE:     %10.4f +/- %8.4f\n"
            % (gneat.pmdec, gneat.pmdec_error))
    sys.stderr.write("parallax: %10.4f +/- %8.4f\n"
            % (gneat.parallax, gneat.parallax_error))

##--------------------------------------------------------------------------##
## Split observer orbit into halves:
avgra = np.average(sneat['dra'])
twopi = 2.0 * np.pi
obs_anomaly = np.arctan2(use_eph['y'], use_eph['x']) % twopi
rel_phase = ((obs_anomaly - np.radians(avgra)) % twopi) / twopi
#phase_per_day = np.median(np.diff(rel_phase) / np.diff(tstamps.tdb.jd))
#orb_period = 1.0 / phase_per_day


def period_check(params):
    """params ~ (zeropoint_tdb, period_days)"""
    mod_phase = ((tstamps.tdb.jd - params[0]) / params[1]) % 1.0
    resid = rel_phase - mod_phase
    return np.sum(resid**2)
param0 = np.array([tstamps.tdb.jd[0], 365.25])
orb_start, orb_period = opti.fmin(period_check, param0)

#orb_period = 365.25
#orb_start = tstamps.tdb.jd[0] - rel_phase[0] / phase_per_day
#orb_start = tstamps.tdb.jd[0] - rel_phase[0] * orb_period
tspan_days = tstamps.tdb.jd.max() - tstamps.tdb.jd.min()
ncycles = np.ceil(tspan_days / orb_period)
orb_zeros = orb_start + np.arange(ncycles) * orb_period
cycle_ends_jd_tdb = np.column_stack((orb_zeros, orb_zeros + orb_period))
cycle_ends_yr_tdb = (cycle_ends_jd_tdb - use_epoch.tdb.jd) / 365.25

##--------------------------------------------------------------------------##
##------------------     Exposure Times Kludge (temp)       ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Kludgey exposure time retrieval ... ")
import astropy.io.fits as pf
ffpath = '/home/rsiverd/ucd_project/ucd_sha_data/legacy_symlinks' # all files
ipaths = np.array([os.path.join(ffpath, x) for x in sneat['iname']])
expsec = np.array([pf.getheader(x)['EXPTIME'] for x in ipaths])
relSNR = np.sqrt(expsec)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------           Theil-Sen Fitting            ----------------##
##--------------------------------------------------------------------------##

sjd_utc, sra, sde = sneat['jdutc'], sneat[_ra_key], sneat[_de_key]
#syr = 2000.0 + ((sjd_utc - 2451544.5) / 365.25)
#smonth = (syr % 1.0) * 12.0
syr = (tstamps.tdb.jd - use_epoch.tdb.jd) / 365.25

ts_ra_model = ts.linefit(syr, sra)
ts_de_model = ts.linefit(syr, sde)
#ts_ra_model = ts.linefit(dt_yrs, sra)
#ts_de_model = ts.linefit(dt_yrs, sde)
#ts_pmra_masyr = ts_ra_model[1] * 3.6e6 * np.cos(np.radians(ts_de_model[0]))
ts_pmra_masyr = ts_ra_model[1] * 3.6e6
ts_pmde_masyr = ts_de_model[1] * 3.6e6


# initial RA/Dec guess:
guess_ra = sra.mean()
guess_de = sde.mean()
if (context.vlevel >= 2):
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("guess_ra:  %15.7f\n" % guess_ra)
    sys.stderr.write("guess_de:  %15.7f\n" % guess_de)

#afpars = [np.radians(guess_ra), np.radians(guess_de), ts_pmra_masyr/1e3, ts_pmde_masyr/1e3, 1.0]
afpars = [np.radians(guess_ra), np.radians(guess_de), 
        np.radians(ts_ra_model[1]), np.radians(ts_de_model[1]), 1.0]
appcoo = af.apparent_radec(use_epoch.tdb.jd, afpars, use_eph)

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

if (context.vlevel >= 1):
    sys.stderr.write("%s\n" % fulldiv)
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

ast_err = 1e-4 / relSNR
sys.stderr.write("%s\n" % fulldiv)
af.setup(use_epoch.tdb.jd, sra, sde, use_eph)
#af.setup(use_epoch.tdb.jd, sra, sde, use_eph, RA_err=ast_err, DE_err=ast_err)
af.set_exponent(2.0)
winner = af.fit_bestpars(sigcut=5.0)
slv_ra, slv_de = af.eval_model(winner)
slv_ra = np.degrees(slv_ra)
slv_de = np.degrees(slv_de)

rsig_RA, rsig_DE = af._calc_radec_residuals_sigma(af._par_guess)

sys.stderr.write("\n")
sys.stderr.write("%s\n" % fulldiv)

# TO INSPECT:
# plotgsource(3192149715934444800); plt.plot(bfra_path, bfde_path)
sys.stderr.write("NOTE TO SELF: Gaia pmRA includes cos(dec)!\n")

##--------------------------------------------------------------------------##
## Plottable curves for the different proper motion fits:
padding = 0.0
syr_range = syr.max() - syr.min()
#syr_lspan = syr_range * 1.1
syr_plt_1 = syr.min() - padding * syr_range
syr_plt_2 = syr.max() + padding * syr_range
syr_edges = np.array([syr_plt_1, syr_plt_2])
#syr_edges = np.array([syr.min(), syr.max()])
p_ols_ra = ra_ols_res.params[0] + syr_edges * ra_ols_res.params[1]
p_ols_de = de_ols_res.params[0] + syr_edges * de_ols_res.params[1]
p_rlm_ra = ra_rlm_res.params[0] + syr_edges * ra_rlm_res.params[1]
p_rlm_de = de_rlm_res.params[0] + syr_edges * de_rlm_res.params[1]
p_sen_ra = ts_ra_model[0] + syr_edges * ts_ra_model[1]
p_sen_de = ts_de_model[0] + syr_edges * ts_de_model[1]

if _have_gaia:
    p_gaia_ra = gneat['ra'].values + syr_edges * gneat['pmra'].values
    p_gaia_de = gneat['dec'].values + syr_edges * gneat['pmdec'].values
else:
    p_gaia_ra, p_gaia_de = None, None

##--------------------------------------------------------------------------##
## Linear ephemeris approximation:

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
## RA/DE figures:
cfsize = (10, 5)
figcoo = plt.figure(4, figsize=(10,4))
figcoo.clf()
figRA = plt.figure(2, figsize=cfsize)
figDE = plt.figure(3, figsize=cfsize)
figRA.clf()
figDE.clf()

t_all = he._eph_data['JDTDB'][::12]
x_all = he._eph_data['X'][::12]
y_all = he._eph_data['Y'][::12]
z_all = he._eph_data['Z'][::12]
dense_eph = np.core.records.fromarrays((t_all, x_all, y_all, z_all),
        names='t,x,y,z')
dense_ra, dense_de = af.apparent_radec(use_epoch.tdb.jd, winner, dense_eph)

#axRA = figRA.add_subplot(111)
axRA = figcoo.add_subplot(121)
axRA.scatter(tstamps.tdb.jd, sra, lw=0, s=5)
axRA.plot(dense_eph['t'], np.degrees(dense_ra), c='k')
axRA.set_ylim(nice_limits(sra))


#axDE = figDE.add_subplot(111)
axDE = figcoo.add_subplot(122)
axDE.scatter(tstamps.tdb.jd, sde, lw=0, s=5)
axDE.plot(dense_eph['t'], np.degrees(dense_de), c='k')
axDE.set_ylim(nice_limits(sde, [2,98]))

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
ax1 = fig.add_subplot(211, aspect='equal')
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
#ax1.axis('off')

colorvec = use_eph['jdtdb']
colorvec = rel_phase
spts = ax1.scatter(sra, sde, lw=0, s=10, c=colorvec)
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
#ax1.set_xlim(nice_limits(p_sen_ra, pctiles=[0,100], pad=1.2))
#ax1.set_ylim(nice_limits(p_sen_de, pctiles=[0,100], pad=1.2))


ax1.plot(slv_ra, slv_de, c='c', lw=1, label='optimized')
#ax1.legend(loc='lower left')
ax1.legend(loc='best')


## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
ax1.yaxis.get_major_formatter().set_useOffset(False)

#cbnorm = mplcolors.Normalize(*spts.get_clim())
#scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#scm.set_array([])
##cbar = fig.colorbar(scm, orientation='horizontal')
#cbar = fig.colorbar(scm, orientation='vertical')

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

#ax2 = fig.add_subplot(212)
#ax2.grid(True)
#ax2.scatter(syr, rel_phase, lw=0, s=15, c=rel_phase)
#
##ax2.scatter(syr, rel_phase, lw=0, s=15, c=rel_phase)
#
#phase_ends = np.array([0, 1])
#for cyc_yrs in cycle_ends_yr_tdb:
#    ax2.plot(cyc_yrs, phase_ends, c='r', lw=0.75, ls=':')

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

plot_name = 'scatter_and_fits.png'
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (illustrate.py):
#---------------------------------------------------------------------
#
#  2020-02-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created illustrate.py.
#

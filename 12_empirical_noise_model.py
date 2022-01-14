#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This script estimates empirical noise model terms for a specific
# instrument using residuals of 4-parameter astrometric fits.
#
# Rob Siverd
# Created:       2021-06-03
# Last modified: 2021-06-03
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
import pickle
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
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

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
#    sys.stderr.write("\nError: astropy module not found!\n")
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

## Fancy text:
degree_sign = u'\N{DEGREE SIGN}'

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

##--------------------------------------------------------------------------##
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()

##--------------------------------------------------------------------------##

j2000_epoch = astt.Time('2000-01-01T12:00:00', scale='tt', format='isot')

def fit_4par(data):
    years = (data['jdtdb'] - j2000_epoch.tdb.jd) / 365.25
    ts_ra_model = ts.linefit(years, data[_ra_key])
    ts_de_model = ts.linefit(years, data[_de_key])
    return {'epoch_jdtdb'  :   j2000_epoch.tdb.jd,
                 'ra_deg'  :   ts_ra_model[0],
                 'de_deg'  :   ts_de_model[0],
             'pmra_degyr'  :   ts_ra_model[1],
             'pmde_degyr'  :   ts_de_model[1],
            }

def eval_4par(data, model):
    years = (data['jdtdb'] - model['epoch_jdtdb']) / 365.25
    calc_ra = model['ra_deg'] + years * model['pmra_degyr']
    calc_de = model['de_deg'] + years * model['pmde_degyr']
    return calc_ra, calc_de


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
    Measure parameters of empirical noise model using residuals from
    4-parameter astrometric fits.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    # ------------------------------------------------------------------
    parser.add_argument('-i', '--instrument', required=True, default=None,
            help='instrument name for plot', type=str)
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-s', '--site',
    #        help='Site to retrieve data for', required=True)
    #parser.add_argument('-n', '--number_of_days', default=1,
    #        help='Number of days of data to retrieve.')
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('--start', type=str, default=None, 
    #        help="Start time for date range query.")
    #parser.add_argument('--end', type=str, default=None,
    #        help="End time for date range query.")
    #parser.add_argument('-d', '--dayshift', required=False, default=0,
    #        help='Switch between days (1=tom, 0=today, -1=yest', type=int)
    #parser.add_argument('-e', '--encl', nargs=1, required=False,
    #        help='Encl to make URL for', choices=all_encls, default=all_encls)
    #parser.add_argument('-s', '--site', nargs=1, required=False,
    #        help='Site to make URL for', choices=all_sites, default=all_sites)
    parser.add_argument('data_files', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #iogroup = parser.add_argument_group('File I/O')
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
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

## Abort if no input files provided:
if not context.data_files:
    sys.stderr.write("No data files provided!\n\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------          catalog config (FIXME)        ----------------##
##--------------------------------------------------------------------------##

## RA/DE coordinate keys for various methods:
centroid_colmap = {
        'simple'    :   ('dra', 'dde'),
        'window'    :   ('wdra', 'wdde'),
        'pp_fix'    :   ('ppdra', 'ppdde'),
        }

centroid_method = 'simple'
#centroid_method = 'window'
#centroid_method = 'pp_fix'
_ra_key, _de_key = centroid_colmap[centroid_method]

##--------------------------------------------------------------------------##
## Ensure presence of input files:
sys.stderr.write("Checking input files ... ")
for dfile in context.data_files:
    if not os.path.isfile(dfile):
        sys.stderr.write("\nError: file missing: %s\n" % dfile)
        #logger.error("file not found: %s" % dfile)
        sys.exit(1)
sys.stderr.write("done.\n")

## Load data files:
sys.stderr.write("Loading data files ... ")
tdata, sdata, gdata = {}, {}, {}
for dfile in context.data_files:
    with open(dfile, 'rb') as pp:
        tdata[dfile], sdata[dfile], gdata[dfile] = pickle.load(pp)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------       Initial Dataset Inspection       ----------------##
##--------------------------------------------------------------------------##

## Tag data sets by filename:
tags = context.data_files

min_pts = 25

ds_npts, ds_large, results_4par = {}, {}, {}
for tag in context.data_files:
    ds_npts[tag] = {x:len(sdata[tag][x]) for x in sdata[tag].keys()}
    ds_large[tag] = [ss for ss,nn in ds_npts[tag].items() if nn>min_pts]
    results_4par[tag] = {x:fit_4par(sdata[tag][x]) for x in ds_large[tag]}


##--------------------------------------------------------------------------##
##------------------         Do 4-Parameter Fitting         ----------------##
##--------------------------------------------------------------------------##

#for tag in context.data_files:

## Merge residuals of all data sets:
lookie_snr = {x:[] for x in context.data_files}
inspection = {x:[] for x in context.data_files}

sys.stderr.write("Performing 4-parameter fits:\n")
for tag in context.data_files:
    sys.stderr.write("--> fitting data from %s ...\n" % tag)
    for sid in ds_large[tag]:
        stmp = sdata[tag][sid]
        model = results_4par[tag][sid]
        sra, sde = eval_4par(stmp, model)
        cos_dec = np.cos(np.radians(model['de_deg']))
        delta_ra_mas = 3.6e6 * (stmp[_ra_key] - sra) * cos_dec
        delta_de_mas = 3.6e6 * (stmp[_de_key] - sde)
        #for iname,xpix,ypix,expt,rmiss,dmiss in zip(stmp['iname'],
        #        stmp['wx'], stmp['wy'], stmp['exptime'],
        #        delta_ra_mas, delta_de_mas):
        #    resid_data_ch1[iname].append({'x':xpix, 'y':ypix,
        #        'ra_err':rmiss, 'de_err':dmiss, 'exptime':expt})
        delta_tot_mas = np.sqrt(delta_ra_mas**2 + delta_de_mas**2)
        med_flux, iqr_flux = rs.calc_ls_med_IQR(stmp['flux'])
        med_resd, iqr_resd = rs.calc_ls_med_IQR(delta_tot_mas)
        med_signal = med_flux * stmp['exptime']
        #snr_scale = med_flux * np.sqrt(stmp['exptime'])
        snr_scale = np.sqrt(med_signal)
        approx_fwhm = delta_tot_mas * snr_scale
        med_fwhm, iqr_fwhm = rs.calc_ls_med_IQR(approx_fwhm)
        npoints = len(stmp['flux'])
        non_neg = (delta_tot_mas > 0.0)
        #if np.any(delta_tot_mas == 0.0):
        #    sys.stderr.write("blank?!?!?!\n")
        #    sys.exit(1)
        lookie_snr[tag].append((med_flux, med_fwhm, npoints,
            snr_scale[0], med_resd))
        inspection[tag].extend(list(zip(delta_tot_mas[non_neg],
            np.abs(delta_ra_mas[non_neg]), np.abs(delta_de_mas[non_neg]),
            stmp['flux'][non_neg], (stmp['flux']*stmp['exptime'])[non_neg])))
        sys.stderr.write("approx_fwhm: %s\n" % str(approx_fwhm))

sys.stderr.write("Fitting complete.\n")

## Merge into single data set:
lookie_combined = []
inspection_comb = []
for tag in context.data_files:
    lookie_combined.extend(lookie_snr[tag])
    inspection_comb.extend(inspection[tag])

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
##------------------        Lin-Log Polynomial Fitting      ----------------##
##--------------------------------------------------------------------------##

def polyfit(x, y, deg):
    if (deg < 1):
        return np.average(y)
    nmat = np.ones_like(y)
    for expo in range(1, deg+1, 1):
        nmat = np.column_stack((nmat, x**expo))
    rckw = {'rcond':None} if (_have_np_vers >= 1.14) else {}
    return np.linalg.lstsq(nmat, y, **rckw)[0]

def polyval(x, mod):
    z = np.zeros_like(x)
    for i in range(mod.size):
        z += mod[i] * x**i
    return z

def iter_fit_logrms(mags, lrms, degs):
    # Initial fit of mag,RMSD, used to find outliers:
    ppmod = polyfit(mags, lrms, degs[0])
    resid = lrms - polyval(mags, ppmod)
    clean = (resid <= 0.5)

    # Make curve from 2nd-pass 'clean' fit:
    ppmod = polyfit(mags[clean], lrms[clean], degs[1])
    return ppmod

##--------------------------------------------------------------------------##

# Magnitude to flux conversion:
def kadu(mag, zeropt=25.0):
    return 10.0**(0.4 * (zeropt - mag))

# Flux to magnitude conversion:
def kmag(adu, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(adu))


import spitz_error_model
reload(spitz_error_model)
sem = spitz_error_model.SpitzErrorModel()

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Conversion factors:
fluxconv = 0.1257       # IRAC ch1
fluxconv = 0.1447       # IRAC ch2
sst_gain = 3.71
fluxconv = {'IRAC1':0.1257, 'IRAC2':0.1447}
sst_gain = {'IRAC1':3.70,   'IRAC2':3.71}

if not (context.instrument in fluxconv.keys()):
    sys.stderr.write("Instrument %s not recognized!\n" % context.instrument)
    sys.stderr.write("Cannot proceed with noise model creation.\n")
    sys.exit(1)

#def signal2counts(signal, fluxconv=0.1447, gain=3.71):
def signal2counts(signal):
    return signal * sst_gain[context.instrument] / fluxconv[context.instrument]

fig_dims = (10, 7)

def limify(ax):
    ax.set_xlim(0.2, 50.)
    ax.set_ylim(500., 2500.)

flx, mfwhm, npoints, msnr, mresd = np.array(lookie_combined).T
counts = msnr**2
instmag = kmag(counts)

#inspection_comb = np.array(inspection_comb)

every_delta_mas, every_ra_delta_mas, every_de_delta_mas, \
        every_flux, every_signal = np.array(inspection_comb).T
#every_instmag = kmag(every_signal * sst_gain / fluxconv)
every_counts  = signal2counts(every_signal)
every_instmag = kmag(every_counts)

## Binned values for plotting:
imag_limits = {'IRAC1':(11.0, 18.0), 'IRAC2':(12.0, 19.0)}
nbins = 14
#imag_limits = {'IRAC1':(10.0, 18.0), 'IRAC2':(10.0, 19.0)}
#nbins = 16
#imag_lower = {'IRAC1':11.0, 'IRAC2':12.0}
#imag_upper = {'IRAC1':18.0, 'IRAC2':19.0}
imag_lower, imag_upper = imag_limits[context.instrument]
#imag_lower, imag_upper = 11.0, 19.0
bwidth = (imag_upper - imag_lower) / float(nbins)
binned_imag, binned_delta_ra_mas, binned_delta_de_mas = [], [], []
for ii in range(nbins):
    b_lo = imag_lower + ii * bwidth
    b_hi = imag_lower + (ii+1) * bwidth
    bmid = 0.5 * (b_lo + b_hi)
    which = (b_lo < every_instmag) & (every_instmag <= b_hi)
    #sys.stderr.write("b_lo, b_hi: %.2f, %.2f\n" % (b_lo, b_hi))
    avg_instmag = np.average(every_instmag[which])
    #binned_imag.append(bmid)
    binned_imag.append(avg_instmag)
    sys.stderr.write("mid=%.3f, avg=%.3f\n" % (bmid, avg_instmag))
    binned_delta_ra_mas.append(np.median(every_ra_delta_mas[which]))
    binned_delta_de_mas.append(np.median(every_de_delta_mas[which]))

sys.stderr.write("Press ENTER to continue ... ")
asdf = input()

binned_imag = np.array(binned_imag)
binned_counts = kadu(binned_imag)
binned_delta_ra_mas = np.array(binned_delta_ra_mas)
binned_delta_de_mas = np.array(binned_delta_de_mas)

do_quantile_fits = False

if do_quantile_fits:
    fdata = pd.DataFrame(data=np.vstack((every_instmag, 
                            np.log10(every_delta_mas))).T,
                            columns=["fmag", "lrms"])
    mod = smf.quantreg('lrms ~ fmag + I(fmag**2)', fdata)
    qlist = [0.25, 0.5, 0.75]
    qmods = {}
    for qq in qlist:
        sys.stderr.write("\rFitting q=%.2f ... " % qq)
        res = mod.fit(q=qq)
        qmods[qq] = res.params.values
    #sys.stderr.write("done.\n")

use_imag = True

medmed_fwhm = np.median(mfwhm)
mmf_txt = 'median: %.1f mas' % medmed_fwhm
fig = plt.figure(3, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(111); ax1.grid(True)
ax1.set_title(context.instrument)
#spts = ax1.scatter(msnr, mresd, c=npoints)
#spts = ax1.scatter(instmag, mresd, c=npoints)
spts = ax1.scatter(every_instmag, every_delta_mas, lw=0, s=1)
ax1.set_ylabel('resid (mas)')
ax1.set_yscale('log')
ax1.set_ylim(30., 1100.)

if use_imag:
    ax1.set_xlabel('instrumental mag')
    ax1.set_xscale('linear')
    ax1.set_xlim(11, 23.)
else:
    ax1.set_xlabel('med_snr')
    ax1.set_xscale('log')
    ax1.set_xlim(0.9, 250.)
cbnorm = mplcolors.Normalize(*spts.get_clim())
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')

## overplot quantile fits:
if do_quantile_fits:
    #fmval = np.linspace(3, 200)
    fmval = np.linspace(12, 23)
    for qq,params in qmods.items():
        qlabel = 'q=%.2f quantile' % qq
        #qrmsd = polyval(np.log10(fmval), params)
        qrmsd = 10.**polyval(fmval, params)
        ax1.plot(fmval, qrmsd, label=qlabel)
    ax1.legend(loc='upper left')

fig.tight_layout()
plt.draw()

##--------------------------------------------------------------------------##
## Use these to fit raw data:
fittable_counts = every_counts
fittable_delta_ra = every_ra_delta_mas
fittable_delta_de = every_de_delta_mas

## Use these to fit binned data:
fittable_counts = binned_counts
fittable_delta_ra = binned_delta_ra_mas
fittable_delta_de = binned_delta_de_mas

### Use these to fit binned data jointly:
#fittable_counts = np.concatenate((binned_counts, binned_counts))
#fittable_delta_ra = np.concatenate((binned_delta_ra_mas, binned_delta_de_mas))
#fittable_delta_de = fittable_delta_ra

## Spitzer noise model thoughts:
## --> RONOISE = 7.5    / [ele] readout noise 
## --> npixels =~ pi/4 * FWHM^2     # from solutions
## --> SEP appears to use a variable box size. The minimum number of pixels
##  in a detection is 5 so read noise must contribute at least at this level.

## RMS model evaluator:
def trialrms(star_counts, fwhm, noise_floor=0, eff_gain=1.00, 
        bkg_ele=0.0, rdnoise_ele=7.5, npixels=5):
    star_ele = star_counts * eff_gain       # signal
    star_noi = np.sqrt(star_ele + npixels*bkg_ele + npixels*rdnoise_ele**2)
    star_snr = star_ele / star_noi
    star_rms = fwhm / star_snr
    return np.sqrt(star_rms**2 + noise_floor**2)

def ra_rms_fitme(params):
    fwhm, nfloor = params
    calc_rms_mas = trialrms(fittable_counts, fwhm, noise_floor=nfloor)
    #residuals = fittable_delta_ra - calc_rms_mas
    residuals = np.log10(fittable_delta_ra / calc_rms_mas)
    return np.sum(residuals*residuals)

def de_rms_fitme(params):
    fwhm, nfloor = params
    calc_rms_mas = trialrms(fittable_counts, fwhm, noise_floor=nfloor)
    #residuals = fittable_delta_de - calc_rms_mas
    residuals = np.log10(fittable_delta_de / calc_rms_mas)
    return np.sum(residuals*residuals)

ra_guess = np.array([1000, 30])
best_ra_pars = opti.fmin(ra_rms_fitme, ra_guess)
sys.stderr.write("best_ra_pars: %s\n" % str(best_ra_pars))

de_guess = np.array([1000, 30])
best_de_pars = opti.fmin(de_rms_fitme, de_guess)
sys.stderr.write("best_de_pars: %s\n" % str(best_de_pars))

use_pars = 0.5 * (best_ra_pars + best_de_pars)

sys.stderr.write("use_pars (%s): %s\n" % (context.instrument, str(use_pars)))

guess_imag = np.linspace(10, 20)
guess_iele = kadu(guess_imag)
#guess_rms = trialrms(guess_iele, 5000., noise_floor=30)
guess_rms = trialrms(guess_iele, use_pars[0], noise_floor=use_pars[1])
#guess_rms = trialrms(guess_iele, 3000, noise_floor=50)

## To get an error bar from flux:





## RMS in RA:
ptopts = {'lw':0, 's':1}
rmslims = (5, 2500)

rfig = plt.figure(4, figsize=fig_dims)
rfig.clf()
rax  = rfig.add_subplot(111)
rax.grid(True)
rax.set_yscale('log')
rax.scatter(every_instmag, every_ra_delta_mas, **ptopts)
rax.scatter(binned_imag, binned_delta_ra_mas, c='r')
rax.plot(guess_imag, guess_rms, c='k')
rax.set_xlabel('Instrumental Mag')
rax.set_ylabel('RA scatter (mas)')
rax.set_ylim(*rmslims)

rfig.tight_layout()
plt.draw()
rplot_file = 'RMS_RA_%s.png' % context.instrument
rfig.savefig(rplot_file)

## RMS in DE:
dfig = plt.figure(5, figsize=fig_dims)
dfig.clf()
dax  = dfig.add_subplot(111)
dax.grid(True)
dax.set_yscale('log')
dax.scatter(every_instmag, every_de_delta_mas, **ptopts)
dax.scatter(binned_imag, binned_delta_de_mas, c='r')
dax.plot(guess_imag, guess_rms, c='k')
dax.set_xlabel('Instrumental Mag')
dax.set_ylabel('DE scatter (mas)')
dax.set_ylim(*rmslims)

dfig.tight_layout()
plt.draw()
dplot_file = 'RMS_DE_%s.png' % context.instrument
dfig.savefig(dplot_file)

sys.exit(0)


##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (12, 10)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.grid(True)
#ax1.axis('off')

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
# CHANGELOG (12_empirical_noise_model.py):
#---------------------------------------------------------------------
#
#  2021-06-03:
#     -- Increased __version__ to 0.0.1.
#     -- First created 12_empirical_noise_model.py.
#

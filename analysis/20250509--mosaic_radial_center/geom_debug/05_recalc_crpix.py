#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Re-examine whether or not the linear adjustments of CRPIX among sensors
# is a sensible thing to do using newly obtained relative rotations that
# I believe are more accurate.
#
# Rob Siverd
# Created:       2026-01-13
# Last modified: 2026-01-13
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
import math
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

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#    warnings.simplefilter('error')    # halt on warnings
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
#class Unbuffered(object):
#   def __init__(self, stream):
#       self.stream = stream
#   def write(self, data):
#       self.stream.write(data)
#       self.stream.flush()
#   def __getattr__(self, attr):
#       return getattr(self.stream, attr)
#
#sys.stdout = Unbuffered(sys.stdout)
#sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##

### Process resources (see https://docs.python.org/3/library/resource.html)
#unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
#prettybig = (3e9, 6e9)    # 3 GB soft, 6 GB hard limit
#if (resource.getrlimit(resource.RLIMIT_DATA) == unlimited):
#    resource.setrlimit(resource.RLIMIT_DATA, prettybig)     # limit heap size
#if (resource.getrlimit(resource.RLIMIT_AS) == unlimited):
#    resource.setrlimit(resource.RLIMIT_AS, prettybig)       # address space

## Memory management:
#def get_memory():
#    with open('/proc/meminfo', 'r') as mem:
#        free_memory = 0
#        for i in mem:
#            sline = i.split()
#            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
#                free_memory += int(sline[1])
#    return free_memory
#
#def memory_limit():
#    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

### Measure memory used so far:
#def check_mem_usage_MB():
#    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#    return max_kb_used / 1000.0

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
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

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

## Star extraction:
#try:
#    import easy_sep
#    reload(easy_sep)
#except ImportError:
#    logger.error("easy_sep module not found!  Install and retry.")
#    sys.stderr.write("Error: easy_sep module not found!\n\n")
#    sys.exit(1)
#pse = easy_sep.EasySEP()

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
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array((c, -s, s, c)).reshape(2, 2)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## From the MCMC-driven fit (see recent_mcmc_fit.txt), we have CRPIX values:
## 2089.0361744 ,  -79.2044974
##  -95.08646925,  -70.37480337,
## 2092.99640046, 2113.22323721,
##  -97.71397795, 2125.95149377,
ne_crpix1, ne_crpix2 = 2089.0361744 ,  -79.2044974
nw_crpix1, nw_crpix2 =  -95.08646925,  -70.37480337
se_crpix1, se_crpix2 = 2092.99640046, 2113.22323721
sw_crpix1, sw_crpix2 =  -97.71397795, 2125.95149377
fitted_crpix = [np.array([2089.0361744 ,  -79.2044974]),
                np.array([ -95.08646925,  -70.37480337]),
                np.array([2092.99640046, 2113.22323721]),
                np.array([ -97.71397795, 2125.95149377]),]


## Relative rotations of the four sensors were fit from the four CDxx 
## matrices assuming a constant pixel scale. Those rotations (IN DEGREES):
jnt_posang = np.array([-0.04337314,  0.08228862,  0.03982677,  0.09518227])
avg_posang = np.average(jnt_posang)
relrot_deg = jnt_posang - jnt_posang[0]
rpa_ne, rpa_nw, rpa_se, rpa_sw = relrot_deg
#rel_rmats = [rotation_matrix(x) for x in np.radians(relrot_deg)]
rel_rmats = [rotation_matrix(-x) for x in np.radians(relrot_deg)]
rmat_ne, rmat_nw, rmat_se, rmat_sw = rel_rmats

##--------------------------------------------------------------------------##
## NE inner corner position (externally obtained and out-of-bounds) on each
## of the four sensors. This is the starting point for estimating CRPIX1
## in a joint fashion. Based on how well input positions are recovered, 
## these should be accurate to ~0.002 pixels.
nec_ne_x, nec_ne_y = 2048.000,    1.000
nec_nw_x, nec_nw_y = -134.580,    9.697
nec_se_x, nec_se_y = 2052.228, 2192.539
nec_sw_x, nec_sw_y = -137.002, 2203.682
corner_coords = [np.array([nec_ne_x, nec_ne_y]),
                 np.array([nec_nw_x, nec_nw_y]),
                 np.array([nec_se_x, nec_se_y]),
                 np.array([nec_sw_x, nec_sw_y]),]

## The X,Y delta seen on NE sensor:
ne_delta_x = ne_crpix1 - nec_ne_x
ne_delta_y = ne_crpix2 - nec_ne_y
ne_delta_xy = np.array([ne_delta_x, ne_delta_y])

##--------------------------------------------------------------------------##
## Compensate CRPIX X,Y delta for relative rotations:
sensor_deltas = [np.dot(rr, ne_delta_xy) for rr in rel_rmats]

## Final CRPIX:
estim_crpix = [x+y for x,y in zip(corner_coords, sensor_deltas)]

## Residuals:
resids = np.array(fitted_crpix) - np.array(estim_crpix)
sys.stderr.write("Residuals:\n%s\n" % str(resids))

##--------------------------------------------------------------------------##
## Quick FITS I/O:
#data_file = 'image.fits'
#img_vals = pf.getdata(data_file)
#hdr_keys = pf.getheader(data_file)
#img_vals, hdr_keys = pf.getdata(data_file, header=True)
#img_vals, hdr_keys = pf.getdata(data_file, header=True, uint=True) # USHORT
#img_vals, hdr_keys = fitsio.read(data_file, header=True)

#date_obs = hdr_keys['DATE-OBS']
#site_lat = hdr_keys['LATITUDE']
#site_lon = hdr_keys['LONGITUD']

## Initialize time:
#img_time = astt.Time(hdr_keys['DATE-OBS'], scale='utc', format='isot')
#img_time += astt.TimeDelta(0.5 * hdr_keys['EXPTIME'], format='sec')
#jd_image = img_time.jd

## Initialize location:
#observer = ephem.Observer()
#observer.lat = np.radians(site_lat)
#observer.lon = np.radians(site_lon)
#observer.date = img_time.datetime

#pf.writeto('new.fits', img_vals)
#qsave('new.fits', img_vals)
#qsave('new.fits', img_vals, header=hdr_keys)

## Star extraction:
#pse.set_image(img_vals, gain=3.6)
#objlist = pse.analyze(sigthresh=5.0)

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

## Theil-Sen line-fitting variants:
#linear_model = ts.linefit(xvals, yvals)
#loglog_model = ts.linefit(np.log10(xvals), np.log10(yvals))
#linlog_model = ts.linefit(xvals, np.log10(yvals))


## Evaluators:
#def loglog_eval(xvals, model):
#    icept, slope = model
#    return 10**icept * np.log10(xvals)**slope
#def loglog_eval(xvals, icept, slope):
#    return 10**icept * xvals**slope

## Better evaluators (FIXME: this should go in a modele):
#def linear_eval(xvals, icept, slope):
#    return icept + slope*xvals                  # for y vs x fit
#def loglog_eval(xvals, icept, slope):
#    return 10**(icept + slope*xvals)            # for log(y) vs x fit
#def loglog_eval(xvals, icept, slope):
#    return 10**(icept + slope*np.log10(xvals))  # for log(y) vs log(x) fit

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
#ax1.set_theta_direction(-1)         # clockwise
#ax1.set_theta_direction(+1)         # counterclockwise
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
#ax1.xaxis.set_major_formatter(fptformat) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
#ax1.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#ax1.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'], rotation=45)
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




######################################################################
# CHANGELOG (05_recalc_crpix.py):
#---------------------------------------------------------------------
#
#  2026-01-13:
#     -- Increased __version__ to 0.0.1.
#     -- First created 05_recalc_crpix.py.
#

#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This is a testbed to help clarify what functional form should be used
# for elliptical distortion in WIRCam. This distortion profile, if needed,
# would arise due to tip/tilt of the detector plane w.r.t. CFHT focal plane.
#
# More information about a useful mathematical formulation can be found in
# Tessore & Metcalf (2015):
# https://www.aanda.org/articles/aa/full_html/2015/08/aa26773-15/aa26773-15.html
# https://www.aanda.org/articles/aa/pdf/2015/08/aa26773-15.pdf
#
# The main idea is that the profile is likely circular / symmetric when
# sampled in a plane perpendicular to the chief ray. As a result, it makes
# sense to model the elliptical profile as essentially a tilted circular
# profile to accommodate actual tip/tilt of the detector mosaic plane. This
# testbed aims to determine what functional form might work well.
#
# Other URLs related to ellipse geometry:
# * https://www.researchgate.net/publication/228546868_A_real-time_FPGA_implementation_of_a_barrel_distortion_correction_algorithm_with_bilinear_interpolation/link/0c96051ef1ca6ee9ca000000/download
# * https://www.onemathematicalcat.org/Math/Precalculus_obj/getEllipseStretchShrinkCircle.htm
#
# Rob Siverd
# Created:       2026-05-26
# Last modified: 2026-06-02
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.1"

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
#import math
#import io
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
import scipy.optimize as opti
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
from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
np.set_printoptions(suppress=False, linewidth=120)
#import operator
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Solution parameter helpers:
import slv_par_tools
reload(slv_par_tools)
spt = slv_par_tools
quads = spt._quads

## Rotation matrix helper:
import rotation_matrix
reload(rotation_matrix)
frot = rotation_matrix.FrameRotation()

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
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
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
## New-style string formatting (more at https://pyformat.info/):

#oldway = '%s %s' % ('one', 'two')
#newway = '{} {}'.format('one', 'two')

#oldway = '%d %d' % (1, 2)
#newway = '{} {}'.format(1, 2)

# With padding:
#oldway = '%10s' % ('test',)        # right-justified
#newway = '{:>10}'.format('test')   # right-justified
#oldway = '%-10s' % ('test',)       #  left-justified
#newway = '{:10}'.format('test')    #  left-justified

# Ordinally:
#newway = '{1} {0}'.format('one', 'two')     # prints "two one"

# Dictionarily:
#newway = '{lastname}, {firstname}'.format(firstname='Rob', lastname='Siverd')

# Centered (new-only):
#newctr = '{:^10}'.format('test')      # prints "   test   "

# Numbers:
#oldway = '%06.2f' % (3.141592653589793,)
#newway = '{:06.2f}'.format(3.141592653589793)

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## Quick ASCII I/O:
#data_file = 'data.txt'
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = np.atleast_1d(np.genfromtxt(data_file, dtype=None, **gftkw))
#all_data = np.genfromtxt(fix_hashes(data_file), dtype=None, **gftkw)
#all_data = aia.read(data_file)

#all_data = append_fields(all_data, ('ra', 'de'), 
#         np.vstack((ra, de)), usemask=False)
#all_data = append_fields(all_data, cname, cdata, usemask=False)

#pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
#all_data = pd.read_csv(data_file)
#all_data = pd.read_csv(data_file, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)
#nskip, cnames = analyze_header(data_file)
#all_data = pd.read_csv(data_file, names=cnames, skiprows=nskip, **pdkwargs)
#all_data = pd.DataFrame.from_records(npy_data)
#all_data = pd.DataFrame(all_data.byteswap().newbyteorder()) # for FITS tables

### Strip leading '#' from column names:
#def colfix(df):
#    df.rename(columns={kk:kk.lstrip('#') for kk in df.keys()}, inplace=True)
#colfix(all_data)

#all_data.rename(columns={'old_name':'new_name'}, inplace=True)
#all_data.reset_index()
#firstrow = all_data.iloc[0]
#for ii,row in all_data.iterrows():
#    pass

#vot_file = 'neato.xml'
#vot_data = av.parse_single_table(vot_file)
#vot_data = av.parse_single_table(vot_file).to_table()

##--------------------------------------------------------------------------##
## Fancy list tricks:
#mask_list = [is_thing1, is_thing2, is_thing3]
#combo_or  = functools.reduce(operator.or_, mask_list)

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
## Misc:
#def log_10_product(x, pos):
#   """The two args are the value and tick position.
#   Label ticks with the product of the exponentiation."""
#   return '%.2f' % (x)  # floating-point
#
#fptformat = plt.FuncFormatter(log_10_product) # wrap function for use

## Convenient, percentile-based plot limits:
#def nice_limits(vec, pctiles=[1,99], pad=1.2):
#    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
#    middle = np.average(ends)
#    return (middle + pad * (ends - middle))

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
## Tip/tilt demo:
ytilt_mrad =   0.
xtilt_mrad = 100.
rmat_xtilt = frot.Rx(xtilt_mrad / 1e3)
rmat_ytilt = frot.Ry(ytilt_mrad / 1e3)

yxt_rmat = rmat_ytilt @ rmat_xtilt
xyt_rmat = rmat_xtilt @ rmat_ytilt

##--------------------------------------------------------------------------##
## Solve prep:
#ny, nx = img_vals.shape
#ny, nx = 4250., 4250.
#ny, nx = 1063., 1063.
#ny, nx = 1063., 1063.
npts = 100
npts =  50
x_list = np.linspace(-2125., 2125., npts)
y_list = x_list
#x_list = (0.5 + np.arange(nx)) / nx - 0.5            # relative (centered)
#y_list = (0.5 + np.arange(ny)) / ny - 0.5            # relative (centered)
#xx, yy = np.meshgrid(x_list, y_list)                 # relative (centered)
mxx, myy = np.meshgrid(x_list, y_list)                 # relative (centered)
mzz = np.zeros_like(mxx)

mxyz1d = np.vstack((mxx.ravel(), myy.ravel(), mzz.ravel()))
# [x0 ... xN],
# [y0 ... yN],
# [z0 ... zN]


## Before starting in anger, let's get inverse polynomial coefficients. The
## model currently coded is used to convert sky-compatible xrel/yrel to
## measured detector xdet,ydet. For this task (and others) we want to quickly
## invert the sense of that solution.
## +txcor, +tycor = calc_rdist_corrections(xrsky, yrsky, sky2det_coeffs)
## -txcor, -tycor = calc_rdist_corrections(txdet, yrsky, sky2det_coeffs)
sky2det_coeffs = spt.guess_distmod.copy()
xr_sky, yr_sky = np.meshgrid(x_list, y_list)
x_corr, y_corr = spt.calc_rdist_corrections(xr_sky, yr_sky, sky2det_coeffs)
xr_det, yr_det = xr_sky + x_corr, yr_sky + y_corr

badguess = np.ones_like(sky2det_coeffs)

rr_sky = np.hypot(xr_sky, yr_sky)
rr_det = np.hypot(xr_det, yr_det)
r_corr = np.hypot(x_corr, y_corr)



## Recover fit parameters:
sys.stderr.write("Try to recover parameters from data ...\n")
#fitme_redo = partial(
recov_par, recov_cov = opti.curve_fit(spt.poly_eval5, 
                                  rr_sky.ravel(), r_corr.ravel(), badguess)
sys.stderr.write("Got: %s\n" % str(recov_par))

## Fit the inverse:
sys.stderr.write("Next fit for inverse coefficients ...\n")
inver_par, inver_cov = opti.curve_fit(spt.poly_eval5,
                                  rr_det.ravel(), r_corr.ravel(), badguess)
sys.stderr.write("Got: %s\n" % str(inver_par))
det2sky_coeffs = inver_par

## Test run:
ixcorr, iycorr = spt.calc_rdist_corr_det2sky(xr_det, yr_det, det2sky_coeffs)
calc_xr_sky = xr_det + ixcorr
calc_yr_sky = yr_det + iycorr

# exponents = 1./(1.+np.arange(len(spt.guess_distmod)))


## Convert mosaic to focal plane coordinates (where distortion is circular):
fpxyz_det = yxt_rmat @ mxyz1d
fpx_det, fpy_det, fpz_det = fpxyz_det
fpxcorr, fpycorr = spt.calc_rdist_corr_det2sky(fpx_det, fpy_det, det2sky_coeffs)
fpx_sky, fpy_sky = fpx_det + fpxcorr, fpy_det + fpycorr
fpxyz_sky = np.vstack((fpx_sky, fpy_sky, fpz_det))
mxyz_sky = yxt_rmat.T @ fpxyz_sky
mxcorr, mycorr, _ = mxyz1d - mxyz_sky
mxc2d = mxcorr.reshape(mxx.shape)
myc2d = mycorr.reshape(myy.shape)

## Things to plot:
rnudge2d = np.hypot(mxc2d, myc2d)
xnudge = mxcorr
ynudge = mycorr


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Also make a grid to illustrate the warping:
g_list = np.linspace(-2000., 2000., 21)
gxx_det, gyy_det = np.meshgrid(g_list, g_list)
grr_det = np.hypot(gxx_det, gyy_det)
gxcorr, gycorr = spt.calc_rdist_corr_det2sky(gxx_det, gyy_det, det2sky_coeffs)
gxx_sky, gyy_sky = gxx_det + gxcorr, gyy_det + gycorr

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Radial-only ...
#rrel1d   = np.hypot(xrel, yrel)
#rnudge1d = np.hypot(xnudge1d, ynudge1d)
#rrel2d   = np.hypot(xx2d, yy2d)
#rnudge2d = np.hypot(xnudge2d, ynudge2d)

xrel = mxx.ravel()
yrel = myy.ravel()

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
fig_dims = (10, 9)
figkw = {'figsize':fig_dims, 'clear':True}
qfig = plt.figure(1, **figkw)
qax1 = qfig.add_subplot(111, aspect='equal')

rfig = plt.figure(2, **figkw)
rax1 = rfig.add_subplot(111, aspect='equal')

cfig = plt.figure(3, **figkw)
cax1 = cfig.add_subplot(111, aspect='equal')

gfig = plt.figure(4, **figkw)
gax1 = gfig.add_subplot(111, aspect='equal')

pfig = plt.figure(5, **figkw)
pax1 = pfig.add_subplot(111)

fig_list = [qfig, rfig, cfig, gfig, pfig]
axs_list = [qax1, rax1, cax1, gax1, pax1]

## Clear figs:
#for ff in fig_list:
#    ff.clf()

#fig, axs = plt.subplots(nrows=2, ncols=2, num=1, clear=True, figsize=fig_dims,
#                        sharex=True, squeeze=False)
# sharex='col' | sharex='row' | squeeze=False
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111, aspect='equal')
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.grid(True)
#ax1.axis('off')

ascale = 1
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
#ax1.quiver(xx, yy, ascale*xnudge, ascale*ynudge, **qxkw)
qax1.quiver(xrel, yrel, ascale*xnudge, ascale*ynudge, **qxkw)
#ax1.scatter(x
rax1.imshow(rnudge2d)

skw = {'lw':0, 's':15}


#cax1.contour(rnudge2d)
cax1.contour(mxx, myy, rnudge2d)


## Grid diagram:
detpkw = {'color':'blue', 'ls':'-'}
gax1.plot(gxx_det, gyy_det, **detpkw)
gax1.plot(gxx_det.T, gyy_det.T, **detpkw)
skypkw = {'color':'red', 'ls':'--'}
gax1.plot(gxx_sky, gyy_sky, **skypkw)
gax1.plot(gxx_sky.T, gyy_sky.T, **skypkw)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

def barrel1(r, zpt, k2):
    return zpt + r*(1 + k2*r*r)

## Distorted / undistorted ratio:
badguess1 = 0.1 * np.ones(2)
xvals = rr_sky.ravel()
yvals = rr_det.ravel()
ratio = yvals / xvals
tsmod = ts.linefit(xvals, ratio - 1.0)
xtest = 0.01 + np.linspace(0, 3000)
#ytest = 1.0 + tsmod[0] + tsmod[1]*xtest
ytest = 1.0 + tsmod[1]*xtest
bar1par, bar1cov = opti.curve_fit(barrel1, xvals, yvals)
b1_ytest = barrel1(xtest, *bar1par)
b1_ratio = b1_ytest / xtest

## Plotting:
pskw = {'lw':0, 's':15}
pax1.grid(True)
pax1.scatter(rr_sky, rr_det / rr_sky, **pskw)
pax1.set_ylim(bottom=1.0)
pax1.plot(xtest, ytest, c='r', label='theil-sen')
pax1.plot(xtest, b1_ratio, c='g', label='barrel1')
pax1.legend(loc='upper left')

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

#lnorm = LogNorm(vmin=0.05, vmax=1.0)

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

#fig.align_labels()
#fig.align_xlabels()
#fig.align_ylabels()
for ff in fig_list:
    ff.tight_layout()
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (elliptical_distortion.py):
#---------------------------------------------------------------------
#
#  2026-05-26:
#     -- Increased __version__ to 0.0.1.
#     -- First created elliptical_distortion.py.
#

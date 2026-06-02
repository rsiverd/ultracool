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
# Rob Siverd
# Created:       2026-05-26
# Last modified: 2026-05-26
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
#import operator
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
xx, yy = np.meshgrid(x_list, y_list)                 # relative (centered)
#xx, yy = np.meshgrid(nx*x_list, ny*y_list)           # absolute (centered)
#xx *= 4.0   # rescale to true size
#yy *= 4.0   # rescale to true size
#rr = np.hypot(xx, yy)
#xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))   # absolute
#yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij') # absolute
#yy, xx = np.nonzero(np.ones_like(img_vals))          # absolute
#yy, xx = np.mgrid[0:ny,   0:nx].astype('uint16')     # absolute (array)
#yy, xx = np.mgrid[1:ny+1, 1:nx+1].astype('uint16')   # absolute (pixel)


## Compute rdist corrections for all grid points:
xnudge2d, ynudge2d = spt.calc_rdist_corrections(xx, yy, spt.guess_distmod)
#xnudge1d, ynudge1d = spt.calc_rdist_corrections(xrel, yrel, spt.guess_distmod)
rnudge2d = np.hypot(xnudge2d, ynudge2d)
xnudge = xnudge2d.ravel()
ynudge = ynudge2d.ravel()


## Radial-only ...
#rrel1d   = np.hypot(xrel, yrel)
#rnudge1d = np.hypot(xnudge1d, ynudge1d)
#rrel2d   = np.hypot(xx2d, yy2d)
#rnudge2d = np.hypot(xnudge2d, ynudge2d)

xrel = xx.ravel()
yrel = yy.ravel()

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
fig.clf()

rfig = plt.figure(2, figsize=fig_dims)
rfig.clf()
rax1 = rfig.add_subplot(111)
#fig, axs = plt.subplots(nrows=2, ncols=2, num=1, clear=True, figsize=fig_dims,
#                        sharex=True, squeeze=False)
# sharex='col' | sharex='row' | squeeze=False
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(111, aspect='equal')
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.grid(True)
#ax1.axis('off')

ascale = 1
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
#ax1.quiver(xx, yy, ascale*xnudge, ascale*ynudge, **qxkw)
ax1.quiver(xrel, yrel, ascale*xnudge, ascale*ynudge, **qxkw)
#ax1.scatter(x
rax1.imshow(rnudge2d)

skw = {'lw':0, 's':15}
#rax1.scatter(rrel, rnudge)
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

#fig.align_labels()
#fig.align_xlabels()
#fig.align_ylabels()
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
rfig.tight_layout()
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

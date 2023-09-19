#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Test out the "fit the corner" procedure for our final matches.
#
# Rob Siverd
# Created:       2023-07-20
# Last modified: 2023-07-20
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

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
#import matplotlib.pyplot as plt
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
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Tangent projection helper:
import tangent_proj
reload(tangent_proj)
tp = tangent_proj

## Angular math routines:
import angle

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
##------------------      Dupuy & Liu (2012) Distortion     ----------------##
##--------------------------------------------------------------------------##

# Dupuy & Liu (2012) distortion coefficients. You can use these as
# the initial guess for the big solve too.
xcoeffs = np.array([ 1.173e-6, -1.303e-6,  5.105e-7,
    -5.287e-10, -4.130e-10, -5.338e-10, -1.353e-10])
ycoeffs = np.array([-6.409e-7,  1.117e-6, -1.191e-6,
    -1.466e-10, -4.589e-10, -3.884e-10, -5.872e-10])

# The following is a routine that applies the distortion coefficients
# and produces corrected X,Y positions. It expects "relative" X,Y
# coordinates as input (i.e., CRPIX1 and CRPIX2 subtracted off).
def dewarp(xcoeffs, ycoeffs, xp, yp):
    x_nudge = xcoeffs[0]*xp*xp + xcoeffs[1]*xp*yp + xcoeffs[2]*yp*yp \
            + xcoeffs[3]*xp*xp*xp + xcoeffs[4]*xp*xp*yp \
            + xcoeffs[5]*xp*yp*yp + xcoeffs[6]*yp*yp*yp
    y_nudge = ycoeffs[0]*xp*xp + ycoeffs[1]*xp*yp + ycoeffs[2]*yp*yp \
            + ycoeffs[3]*xp*xp*xp + ycoeffs[4]*xp*xp*yp \
            + ycoeffs[5]*xp*yp*yp + ycoeffs[6]*yp*yp*yp
    return xp + x_nudge, yp + y_nudge


##--------------------------------------------------------------------------##
##------------------      CFHT/WIRCam File Name Munger      ----------------##
##--------------------------------------------------------------------------##

## Extract unique tag from CFHT filename:
def parse_wircam(ipath):
    wroot = os.path.basename(ipath)     # snip folders
    wroot = wroot.split('.fits')[0]     # look before the .fits.
    wroot = wroot.split('_')[-1]        # last bit before the .fits
    return wroot

##--------------------------------------------------------------------------##
##------------------       CFHT/WIRCam Configuration        ----------------##
##--------------------------------------------------------------------------##

## Adopted center of mosaic (hopefully the optical axis):
crpix1 = 2122.690779
crpix2 =  -81.678888

## Sensor size:
sensor_xpix = 2048
sensor_ypix = 2048


##--------------------------------------------------------------------------##
## Load the matches list:
sys.stderr.write("Loading matches list ... ")
tik = time.time()
#data_file = '20230629--final_matches.csv'
data_file = '20230719--final_matches_new.csv'
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
data = pd.read_csv(data_file, **pdkwargs)
data['xrel'] = data['X Pixel'] - crpix1
data['yrel'] = data['Y Pixel'] - crpix2
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

## Load original image headers:
sys.stderr.write("Loading header data ... ")
tik = time.time()
#data_file = '20230629--final_matches.csv'
ihdr_file = 'header_data.csv'
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
hdrs = pd.read_csv(ihdr_file, **pdkwargs)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))
hdrs['ibase'] = [os.path.basename(x) for x in hdrs['FILENAME']]
hdrs['imtag'] = [parse_wircam(x) for x in hdrs['FILENAME']]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Fitting procedure:
def evaluator(params, imdata):
    pscale = params[0]
    acoeffs = params[1:8]   # X coefficients (7 total)
    bcoeffs = params[8:15]  # Y coefficients (7 total)
    pacrv = params[15:].reshape(-1, 3)   # (PA, CRVAL1, CRVAL2) per image

    deltas = []
    for ipars,subset in zip(pacrv, imdata):
        pa_deg, cv1, cv2 = ipars
        xcorr, ycorr = dewarp(acoeffs, bcoeffs,
                subset['xrel'].values, subset['yrel'].values)
        this_cdmat = tp.make_cdmat(pa_deg, pscale)
        tra, tdec = tp.xycd2radec(this_cdmat, xcorr, ycorr, cv1, cv2)
        #tra = tra % 360.0
        #gra = subset['Gaia RA'].values
        #gde = subset['Gaia Dec'].values
        #import pdb; pdb.set_trace()
        #deltas += angle.dAngSep(tra, tdec, gra, gde) 
        #deltas += angle.slow_dAngSep(tra, tdec, gra, gde) 
        deltas += angle.dAngSep(tra, tdec, subset['Gaia RA'].values, subset['Gaia Dec'].values).tolist()

    return np.sum(deltas)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

### Determine X,Y limits for the fitting region:
#corner_size = 256
#corner_xmin = sensor_xpix - corner_size
#corner_ymax = corner_size

## Group by image:
chunks = data.groupby('Image Name')
n_imgs = len(np.unique(data['Image Name']))

## Process everything:
sys.stderr.write("Processing stuff ...\n")
ntodo = 10
tik = time.time()
per_image_subsets = []
per_image_pacrval = []
for ii,(fcat_path,isubset) in enumerate(chunks, 1):
    #this_ibase = os.path.basename(tag)
    this_imtag = parse_wircam(fcat_path)
    match = hdrs[hdrs.imtag == this_imtag]
    per_image_pacrval.append(0)
    per_image_pacrval.append(match['CRVAL1'].values[0])
    per_image_pacrval.append(match['CRVAL2'].values[0])
    per_image_subsets.append(isubset)

    if (ntodo > 0) and (ii >= ntodo):
        break
#sys.stderr.write("done.\n")
tok = time.time()
sys.stderr.write("Cranked in %.3f seconds.\n" % (tok-tik))



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Initial parameter guess:
guess_pscale = 0.305
init_guess = [guess_pscale] + xcoeffs.tolist() + ycoeffs.tolist() \
        + per_image_pacrval

init_params = np.array(init_guess)


## Function to minimize:
sys.stderr.write("Starting minimization ... \n")
tik = time.time()
minimize_this = partial(evaluator, imdata=per_image_subsets)
answer = opti.fmin(minimize_this, init_params)
tok = time.time()
sys.stderr.write("Minimum found in %.3f seconds.\n" % (tok-tik))

sys.exit(0)

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
#rcam_H2_1319397p.fits.fzscm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#scm.set_array([])
#cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (15_fit_everything.py):
#---------------------------------------------------------------------
#
#  2023-07-20:
#     -- Increased __version__ to 0.0.1.
#     -- First created 15_fit_everything.py.
#

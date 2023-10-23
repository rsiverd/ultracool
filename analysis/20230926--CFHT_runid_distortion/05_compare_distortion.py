#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Compare distortion solutions for different WIRCam run IDs.
#
# Rob Siverd
# Created:       2023-09-26
# Last modified: 2023-09-26
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
import resource
import signal
import glob
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
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

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

## WCS tune-up helpers (beta):
try:
    import wircam_wcs_tuneup
    reload(wircam_wcs_tuneup)
    #ecl = extended_catalog.ExtendedCatalog()
    wwt = wircam_wcs_tuneup
except ImportError:
    logger.error("failed to import wircam_wcs_tuneup module!")
    sys.exit(1)

## WIRCam polynomial routines:
try:
    import wircam_poly
    reload(wircam_poly)
    wcp = wircam_poly.WIRCamPoly()
except ImportError:
    logger.error("failed to import wircam_poly module!")
    sys.exit(1)

## Region-file creation tools:
_have_region_utils = False
try:
    import region_utils
    reload(region_utils)
    rfy = region_utils
    _have_region_utils = True
except ImportError:
    sys.stderr.write(
            "\nWARNING: region_utils not found, DS9 regions disabled!\n")

## Tangent projection helper:
import tangent_proj
reload(tangent_proj)
tp = tangent_proj

## Angular math routines:
import angle
reload(angle)

## Custom polynomial fitter:
import custom_polyfit
reload(custom_polyfit)
cpf2_dx = custom_polyfit.CustomPolyFit2D()
cpf2_dy = custom_polyfit.CustomPolyFit2D()
cpf2_dx.set_degree(2, 3)
cpf2_dy.set_degree(2, 3)

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
## Save FITS image with clobber (astropy / pyfits):
#def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    if header:
#        while (len(header) < padkeys):
#            header.append() # pad header
#    if os.path.isfile(iname):
#        os.remove(iname)
#    pf.writeto(iname, idata, header=header, **kwargs)
#    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
## Save FITS image with clobber (fitsio):
#def qsave(iname, idata, header=None, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    #if os.path.isfile(iname):
#    #    os.remove(iname)
#    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
#    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()




##--------------------------------------------------------------------------##
##------------------         Identify CSV Files             ----------------##
##--------------------------------------------------------------------------##


coeff1_files = sorted(glob.glob('by_runid/*/run_coeffs_1.csv'))
coeff2_files = sorted(glob.glob('by_runid/*/run_coeffs_2.csv'))
rmatch_files = sorted(glob.glob('by_runid/*/run_match_subset.csv'))
runids_list  = [x.split('/')[1] for x in coeff1_files]

sys.stderr.write("Loading data ... ")
coeff1_frames = {k:pd.read_csv(x) for k,x in zip(runids_list, coeff1_files)}
coeff2_frames = {k:pd.read_csv(x) for k,x in zip(runids_list, coeff2_files)}
rmatch_frames = {k:pd.read_csv(x) for k,x in zip(runids_list, rmatch_files)}
sys.stderr.write("done.\n")

## Calculate median absolute error:
medabs_errors = {k:np.median(x['r_err']) for k,x in rmatch_frames.items()}

##--------------------------------------------------------------------------##
##------------------        Make Grid for Quiver Plot       ----------------##
##--------------------------------------------------------------------------##

npix = 2048
nbin = 32
#nbin = 16
bhalfsize = npix / nbin * 0.5
bctr = (np.arange(nbin) + 0.5) / float(nbin)
x_list = bctr * npix
y_list = bctr * npix
xx, yy = np.meshgrid(x_list, y_list)    # cell centers (abs x,y)
rel_xx = xx - wwt.crpix1                # cell center  (rel x)
rel_yy = yy - wwt.crpix2                # cell center  (rel y)

##--------------------------------------------------------------------------##
##------------------        Evaluate and Compare Fits       ----------------##
##--------------------------------------------------------------------------##

save_x_models = {}
save_y_models = {}

save_x_nudges = {}
save_y_nudges = {}

for runid in runids_list:
    cpf2_dx = custom_polyfit.CustomPolyFit2D()
    cpf2_dy = custom_polyfit.CustomPolyFit2D()
    cpf2_dx.set_degree(2, 3)
    cpf2_dy.set_degree(2, 3)


    #coeffs = coeff2_frames[runid]
    coeffs = coeff1_frames[runid]
    x_data = coeffs[coeffs.dim == 'x'].values[0]
    y_data = coeffs[coeffs.dim == 'y'].values[0]
    x_pars = x_data[2:]
    y_pars = y_data[2:]
    cpf2_dx.set_model(x_pars)
    cpf2_dy.set_model(y_pars)
    save_x_models[runid] = cpf2_dx
    save_y_models[runid] = cpf2_dy

    x_nudges = cpf2_dx.eval(rel_xx, rel_yy)
    y_nudges = cpf2_dy.eval(rel_xx, rel_yy)

    save_x_nudges[runid] = x_nudges
    save_y_nudges[runid] = y_nudges
    pass

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (14, 7)
fig = plt.figure(1, figsize=fig_dims)
fig.clf()

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

ax1 = fig.add_subplot(121, aspect='equal')
ax2 = fig.add_subplot(122, aspect='equal')
plt.show()
plt.ion()

for runid in runids_list:
    sys.stderr.write("Showing distortion from %s ...\n" % runid)
    ax1.imshow(np.abs(save_x_nudges[runid]))
    ax1.invert_yaxis()
    ax2.imshow(np.abs(save_y_nudges[runid]))
    ax2.invert_yaxis()
    plt.draw()
    plt.show()
    sys.stderr.write("Press ENTER to continue ('q' to quit) ... ")
    answer = input()
    if answer == 'q':
        break
    pass

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

#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (05_compare_distortion.py):
#---------------------------------------------------------------------
#
#  2023-09-26:
#     -- Increased __version__ to 0.0.1.
#     -- First created 05_compare_distortion.py.
#

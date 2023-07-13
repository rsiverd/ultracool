#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Illustrate the typical amount of movement due to proper motion for
# Gaia sources in the CFHT calib1 field. To summarize, we see a median
# movement distance of ~0.07 WIRCam pixels (0.3 arcsec/pix), which is
# worth correcting for. Some sources move significantly farther.
#
# Rob Siverd
# Created:       2023-06-14
# Last modified: 2023-06-14
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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

import gaia_match
gm  = gaia_match.GaiaMatch()


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
try:
    import my_kde
    reload(my_kde)
    mk = my_kde
except ImportError:
    sys.stderr.write("\nError!  my_kde module not found!\n"
           "Please install and try again ...\n\n")
    sys.exit(1)

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
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

# Prime the Gaia source matcher:
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/gaia_calib1_NE.csv'
gm.load_sources_csv(gaia_csv_path)


##--------------------------------------------------------------------------##
## Shorthand:
gsrc = gm._srcdata

## What columns we use:
## gsrc['ref_epoch']    # e.g., 2015.5
## gsrc['ra']           # e.g., 294.7030615828344   (degrees)
## gsrc['dec']          # e.g., 35.21197045656248   (degrees)
## gsrc['pmra']         # e.g., -4.383538496657437   (mas/yr) == pmRA * cosDE
## gsrc['pmdec']        # e.g., -6.200807321967185   (mas/yr)

## NOTE: the delta_t_yrs value is approximate, for illustration purposes
delta_t_yrs = 4.0           # 2015.5 epoch, 2011 observations

## CFHT / WIRCam params:
arcsec_per_pix = 0.3

## Select subset of sources with usable proper motions:
has_pmra = ~np.isnan(gsrc['pmra'])
has_pmde = ~np.isnan(gsrc['pmdec'])
gaia_use = gsrc[has_pmra & has_pmde]

## How far do they move:
cos_dec = np.cos(np.radians(gaia_use['dec'])).values
ra_motion_mas = delta_t_yrs * gaia_use['pmra'].values
de_motion_mas = delta_t_yrs * gaia_use['pmdec'].values
tot_motion_mas = np.hypot(ra_motion_mas, de_motion_mas)
tot_motion_pix = tot_motion_mas / 1e3 / arcsec_per_pix

## Final positions:
ra_adjust_arcsec = ra_motion_mas / 1e3 / cos_dec
de_adjust_arcsec = de_motion_mas / 1e3

ra_corr = gaia_use[ 'ra'].values + (ra_adjust_arcsec / 3600.0)
de_corr = gaia_use['dec'].values + (de_adjust_arcsec / 3600.0)

#tot_adjustment_mas = delta_t_yrs * np.hypot(gaia_use['pmra'], gaia_use['pmdec']).values
#tot_adjustment_pix = tot_adjustment_mas / 1e3 / arcsec_per_pix
#np.sum(tot_adjustment_pix > 0.5)    # 53
#dde_adjustment_mas = delta_t_yrs * gaia_use['pmdec']
#dra_adjustment_mas = delta_t_yrs * gaia_use['pmra'] / cos_dec


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Median shift distance:
med_motion_pix = np.median(tot_motion_pix)
sys.stderr.write("Median total motion (pix): %7.3f\n" % med_motion_pix)

## Announce how many sources moved more than half a pixel:
n_moved_far = np.sum(tot_motion_pix > 0.5)
sys.stderr.write("Have %d Gaia sources that moved farther than 0.5 pixels.\n"
        % n_moved_far)

## Percentils of movement distance:
pctiles = [25, 50, 75]
mvmt_pix_quartiles = np.percentile(tot_motion_pix, pctiles)
for stuff in zip(pctiles, mvmt_pix_quartiles):
    sys.stderr.write("%dth percentile movement (pix): %7.3f\n" % stuff)

##--------------------------------------------------------------------------##
## KDE:
maxval = 12.0
npts = 1000
#grid = (1.0 + np.arange(npts)) / float(npts) * maxval
#kde_pnts, kde_vals = mk.go(tot_motion_pix, npts=npts)


_logdist = True
#use_motion = tot_motion_pix
if _logdist:
    #use_motion = np.log10(tot_motion_pix)
    kde_pexp, kde_vals = mk.go(np.log10(tot_motion_pix), npts=npts)
    kde_pnts = 10.**kde_pexp
else:
    kde_pnts, kde_vals = mk.go(tot_motion_pix, logsamp=True, npts=npts)


#npts = 10000
#kde_pnts, kde_vals = mk.go(tot_motion_pix, xmin=0.001, xmax=12.0, npts=npts)

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
#plt.gcf().clf()
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(111)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
#ax1.axis('off')

ax1.plot(kde_pnts, kde_vals)
ax1.set_xscale('log')
ax1.axvline(med_motion_pix, c='r', ls='--', label='med_dist (%.3f)'%med_motion_pix)
ax1.set_xlabel('PM Distance (pixels)')
ax1.set_ylabel('Relative frequency')
ax1.legend(loc='upper right')

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

plot_name = 'gaia_mvmt.png'
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')


######################################################################
# CHANGELOG (gaia_mvmt.py):
#---------------------------------------------------------------------
#
#  2023-06-14:
#     -- Increased __version__ to 0.1.0.
#     -- First created gaia_mvmt.py.
#

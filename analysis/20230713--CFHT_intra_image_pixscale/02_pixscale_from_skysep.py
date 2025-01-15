#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Estimate pixel scale from on-sky separation. With plotting.
#
# Rob Siverd
# Created:       2023-07-13
# Last modified: 2023-07-13
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
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
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
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import cmocean
#import theil_sen as ts
#import window_filter as wf
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import angle
reload(angle)

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
## Quick ASCII I/O:
sys.stderr.write("Loading data file ... ")
tik = time.time()
#data_file = '20230629--final_matches.csv'
data_file = '20230719--final_matches_new.csv'
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
data = pd.read_csv(data_file, **pdkwargs)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

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

#all_data.rename(columns={'old_name':'new_name'}, inplace=True)
#all_data.reset_index()
#firstrow = all_data.iloc[0]
#for ii,row in all_data.iterrows():
#    pass

##--------------------------------------------------------------------------##
## Alternate column names:
#data.rename(columns={'Image Name':'ipath'}, inplace=True)
#data.rename(columns={'X Pixel':'x'}, inplace=True)
#data.rename(columns={'Y Pixel':'y'}, inplace=True)
#data.rename(columns={'Gaia RA':'gra'}, inplace=True)
#data.rename(columns={'Gaia Dec':'gde'}, inplace=True)
#data.rename(columns={'Calc RAs':'sra'}, inplace=True)
#data.rename(columns={'Calc Decs':'sde'}, inplace=True)


_DEBUG = True
_DEBUG = False

##--------------------------------------------------------------------------##
## Load/store ability (slow calculation):
pkl_save = 'pixel_scales.pickle'

# Group by image:
chunks = data.groupby('Image Name')
n_imgs = len(np.unique(data['Image Name']))

if os.path.isfile(pkl_save):
    sys.stderr.write("Loading data from %s ... " % pkl_save)
    with open(pkl_save, 'rb') as ppp:
        rscale_save = pickle.load(ppp)
    sys.stderr.write("done.\n")
else:
    tik = time.time()
    rscale_save = {}
    ntodo = 1
    max_sep = 50
    for ii,(tag,isubset) in enumerate(chunks, 1):
        sys.stderr.write("\rImage %d of %d ... " % (ii, n_imgs))
        cbase = os.path.basename(tag)
        xpix  = isubset['X Pixel'].values
        ypix  = isubset['Y Pixel'].values
        gra   = isubset['Gaia RA'].values
        gde   = isubset['Gaia Dec'].values
        npts  = len(isubset)

        if _DEBUG:
            for col,vec in zip(['xpix', 'ypix', 'gra', 'gde'], 
                                [xpix, ypix, gra, gde]):
                nuniq = len(np.unique(vec))
                if nuniq != npts:
                    sys.stderr.write("non-unique %s (%d) ... " % (col, npts - nuniq))

        rhits = []    
        yhits = []
        for tx,ty,tra,tde in zip(xpix, ypix, gra, gde):
            #xnear = np.abs(xpix - tx) < max_sep
            #ynear = np.abs(ypix - ty) < max_sep
            #check = xnear & ynear
            #rsep  = np.hypot(xpix[check] - tx, ypix[check] - ty)
            #rsep  = np.hypot(neato['X Pixel'] - tx, neato['Y Pixel'] - ty)

            rsep  = np.hypot(xpix - tx, ypix - ty)
            #asep  = angle.dAngSep(tra, tde, gra, gde)
            #pscl = 3600.0 * asep / rsep
            which = (0.0 < rsep) & (rsep < max_sep)
            rdist = rsep[which]
            
            adist = angle.dAngSep(tra, tde, gra[which], gde[which])
            #adist = angle.dAngSep(tra, tde, 
            #        gra[check][which], gde[check][which])
            pxscl = 3600.0 * adist / rdist
            #ydiff = np.abs(ty - ypix[which])
            #ddiff = np.abs(tde - gde[which])
            #dec_sep = np.abs(tde - gde)
            rhits += [(tx, ty, *rp) for rp in zip(rdist, pxscl) if rp[1]>0]
            pass
        rscale_save[cbase] = np.array(rhits)
    
        if (ntodo > 0) and (ii >= ntodo):
            break
    sys.stderr.write("done.\n")
    tok = time.time()
    sys.stderr.write("Cranked in %.3f seconds.\n" % (tok-tik))

    sys.stderr.write("Saving data to %s ... " % pkl_save)
    with open(pkl_save, 'wb') as ppp:
        pickle.dump(rscale_save, ppp)
    sys.stderr.write("done.\n")

sys.exit(0)

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
sys.stderr.write("Making a plot ... this could be slow.\n")
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(111, aspect='equal')
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
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

first = list(rscale_save.keys())[0]
#rsdata = rscale_save[first]
rsdata = np.concatenate(list(x for x in rscale_save.values()))

rx, ry, rpix, rpscl = rsdata.T

#skw = {'lw':0, 's':25, 'vmin':0.295, 'vmax':0.315}
skw = {'lw':0, 's':5, 'vmin':0.295, 'vmax':0.315}
spts = ax1.scatter(rx, ry, c=rpscl, **skw)


##cbar = fig.colorbar(spts, orientation='vertical')   # old way
cbnorm = mplcolors.Normalize(*spts.get_clim())
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

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

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (02_pixscale_from_skysep.py):
#---------------------------------------------------------------------
#
#  2023-07-13:
#     -- Increased __version__ to 0.1.0.
#     -- First created 02_pixscale_from_skysep.py.
#

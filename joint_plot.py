#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Plot results of joint fitting.
#
# Rob Siverd
# Created:       2023-01-25
# Last modified: 2023-01-25
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
import resource
import signal
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
try:
    import my_kde
    reload(my_kde)
    mk = my_kde
except ImportError:
    logger.error("module my_kde not found!  Install and retry.")
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
## New-style string formatting (more at https://pyformat.info/):

##--------------------------------------------------------------------------##
## On-the-fly file modifications:
#def fix_hashes(filename):
#    with open(filename, 'r') as ff:
#        for line in ff:
#            if line.startswith('#'):
#                if ('=' in line):
#                    continue                # param, ignore
#                else:
#                    yield line.lstrip('#')  # header, keep
#            else:
#                yield line

#def analyze_header(filename):
#    skip_rows = 0
#    col_names = []
#    with open(filename, 'r') as ff:
#        for line in ff:
#            if line.startswith('#'):
#                skip_rows += 1
#                if ('=' in line):
#                    continue
#                else:
#                    hline = line.rstrip()
#                    col_names = hline.lstrip('#').split()
#                    continue
#            else:
#                #sys.stderr.write("Found data ... stopping.\n")
#                break
#    return skip_rows, col_names

##--------------------------------------------------------------------------##
## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

## Matrix printer:
def mprint(matrix):
    for row in matrix:
        sys.stderr.write("  %s\n" % str(row))
    return

##--------------------------------------------------------------------------##
## Config:
_cryo_warm_cutoff = 2455562.5007660175      # JD (TDB)

##--------------------------------------------------------------------------##
## Pre-load full paths of pcat files:
pfp_list = 'pcat_file_paths.txt'
if not os.path.isfile(pfp_list):
    sys.stderr.write("File not found: %s\n" % pfp_list)
    sys.exit(1)
with open(pfp_list) as pfpl:
    pcat_files = [x.strip() for x in pfpl.readlines()]

## Create full path lookup table:
pfp_lut = {}
for ppath in pcat_files:
    pbase = os.path.basename(ppath)
    pfp_lut[pbase] = ppath

## Pre-load full paths of pcat files and object counts:
pcat_list = 'pcat_file_specs.csv'
gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
gftkw.update({'names':True, 'autostrip':True})
gftkw.update({'delimiter':','})
pcat_data = np.genfromtxt(pcat_list, dtype=None, **gftkw)

## Create full path lookup table:
obj_lut = dict(zip(pcat_data['pbase'], pcat_data['nobjs']))
pfp_lut = dict(zip(pcat_data['pbase'], pcat_data['ppath']))

##--------------------------------------------------------------------------##
## Quick ASCII I/O:
#data_file = 'results_joint.csv'
cdmat_file = 'results_joint.csv'
gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
gftkw.update({'names':True, 'autostrip':True})
gftkw.update({'delimiter':','})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
all_data = np.genfromtxt(cdmat_file, dtype=None, **gftkw)
#all_data = np.atleast_1d(np.genfromtxt(data_file, dtype=None, **gftkw))
#all_data = np.genfromtxt(fix_hashes(data_file), dtype=None, **gftkw)
#all_data = aia.read(data_file)

## Matching results from 'padeg' fit flavor:
padeg_file = 'results_joint_padeg.csv'
pad_data   = np.genfromtxt(padeg_file, dtype=None, **gftkw)
ngaia_lut  = dict(zip(pad_data['iname'], pad_data['ngaia']))

matched    = np.array([x in pad_data['iname'] for x in all_data['iname']])

#use_data   = all_data[matched]


## nobjs config (ExtCat vs Gaia matches):
nobjs_gaia = True
#nobjs_gaia = False
if nobjs_gaia:
    obj_lut    = ngaia_lut
    nobj_label = 'Gaia Matches'
    nobj_plot = 'joint_results_ngaia.png'
    use_data   = all_data[matched]
else:
    #obj_lut    = obj_lut    # from above
    nobj_label = 'Catalog Stars'
    nobj_plot = 'joint_results_ncat.png'
    use_data  = all_data

## Process fit results:
nobj_thresh = 20
derot_obj = {'I1':[], 'I2':[]}
derot_cdm = {'I1':[], 'I2':[]}
derot_tdb = {'I1':[], 'I2':[]}
for ii,stuff in enumerate(use_data):
    pbase = stuff['iname']
    channel = stuff['aor_tag'].split('_')[1]
    this_cdm = np.array([stuff['new_cd11'], stuff['new_cd12'],
        stuff['new_cd21'], stuff['new_cd22']]).reshape(2,2)
    this_rot = rotation_matrix(np.radians(stuff['padeg']))
    this_obj = obj_lut.get(stuff['iname'])
    #this_obj = ngaia_lut.get(stuff['iname'])
    #this_cdm = np.array([all_data['new_cd11'][ii], all_data['new_cd12'][ii],
    #    all_data['new_cd21'][ii], all_data['new_cd22'][ii]]).reshape(2,2)
    #this_rot = rotation_matrix(np.radians(all_data['padeg'][ii]))
    rot_prod = np.matmul(this_rot, this_cdm)
    derot_obj[channel].append(this_obj)
    derot_cdm[channel].append(rot_prod.flatten().tolist())
    derot_tdb[channel].append(stuff['jdtdb'])
    cd11_abs = 3600.0 * np.abs(rot_prod.flatten()[0])
    if cd11_abs > 1.225:
        sys.stderr.write("cd11_abs: %f\n" % cd11_abs)
        sys.stderr.write("stuff: %s\n" % str(stuff))
        sys.stderr.write("\n")
derot_cdm['I1'] = np.array(derot_cdm['I1'])
derot_cdm['I2'] = np.array(derot_cdm['I2'])
derot_tdb['I1'] = np.array(derot_tdb['I1'])
derot_tdb['I2'] = np.array(derot_tdb['I2'])
derot_obj['I1'] = np.array(derot_obj['I1'])
derot_obj['I2'] = np.array(derot_obj['I2'])

## Cheesy flip fix:
derot_cdm['I1'] = 3600.0 * np.abs(derot_cdm['I1'])
derot_cdm['I2'] = 3600.0 * np.abs(derot_cdm['I2'])


## Scatter in the CD matrix values:
ch1_cdm_med, ch1_cdm_iqr = rs.calc_ls_med_IQR(derot_cdm['I1'], axis=0)
ch2_cdm_med, ch2_cdm_iqr = rs.calc_ls_med_IQR(derot_cdm['I2'], axis=0)

ch1_cdm_med_arcsec = 3600.0 * ch1_cdm_med
ch1_cdm_iqr_arcsec = 3600.0 * ch1_cdm_iqr
ch2_cdm_med_arcsec = 3600.0 * ch2_cdm_med
ch2_cdm_iqr_arcsec = 3600.0 * ch2_cdm_iqr

## Compare many-source and sparse-source pixel scales:
fulldiv = 80 * '-'
ch1_cutoff = 25
ch2_cutoff = 20

sys.stderr.write("%s\n" % fulldiv)
sys.stderr.write("ch1 median CD11 (everything): %.6f +/- %.6f\n"
        % (ch1_cdm_med[0], ch1_cdm_iqr[0]))
sys.stderr.write("ch1 median CD22 (everything): %.6f +/- %.6f\n"
        % (ch1_cdm_med[3], ch1_cdm_iqr[3]))
keep = (derot_obj['I1'] > ch1_cutoff)
hq_ch1_cdm_med, hq_ch1_cdm_iqr = rs.calc_ls_med_IQR(derot_cdm['I1'][keep], axis=0)
sys.stderr.write("\n")
sys.stderr.write("ch1 median CD11 (manysource): %.6f +/- %.6f\n"
        % (hq_ch1_cdm_med[0], hq_ch1_cdm_iqr[0]))
sys.stderr.write("ch1 median CD22 (manysource): %.6f +/- %.6f\n"
        % (hq_ch1_cdm_med[3], hq_ch1_cdm_iqr[3]))



sys.stderr.write("%s\n" % fulldiv)
sys.stderr.write("ch2 median CD11 (everything): %.6f +/- %.6f\n"
        % (ch2_cdm_med[0], ch2_cdm_iqr[0]))
sys.stderr.write("ch2 median CD22 (everything): %.6f +/- %.6f\n"
        % (ch2_cdm_med[3], ch2_cdm_iqr[3]))
keep = (derot_obj['I2'] > ch2_cutoff)
hq_ch2_cdm_med, hq_ch2_cdm_iqr = rs.calc_ls_med_IQR(derot_cdm['I2'][keep], axis=0)
sys.stderr.write("\n")
sys.stderr.write("ch2 median CD11 (manysource): %.6f +/- %.6f\n"
        % (hq_ch2_cdm_med[0], hq_ch2_cdm_iqr[0]))
sys.stderr.write("ch2 median CD22 (manysource): %.6f +/- %.6f\n"
        % (hq_ch2_cdm_med[3], hq_ch2_cdm_iqr[3]))


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

#all_data.rename(columns={'old_name':'new_name'}, inplace=True)
#all_data.reset_index()
#firstrow = all_data.iloc[0]
#for ii,row in all_data.iterrows():
#    pass

#vot_file = 'neato.xml'
#vot_data = av.parse_single_table(vot_file)
#vot_data = av.parse_single_table(vot_file).to_table()

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
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
ax2.grid(True)
#ax1.axis('off')

skw = {'lw':0}
if len(derot_cdm['I1']):
    ch1_cd11_lab = '%.5f arcsec' % ch1_cdm_med[0]
    ax1.scatter(derot_tdb['I1'], derot_cdm['I1'][:,0], label='ch1 CD11', **skw)
    ax1.scatter(derot_tdb['I1'], derot_cdm['I1'][:,3], label='ch1 CD22', **skw)
    ax1.axvline(_cryo_warm_cutoff, ls='--', c='r', label='cryo/warm')
    ax1.axhline(ch1_cdm_med[0], ls='--', c='g', label=ch1_cd11_lab)
    ax1.set_ylabel('Plate Scale (arcsec / pix)')
    ax1.legend(loc='upper right')
if len(derot_cdm['I2']):
    ch2_cd11_lab = '%.5f arcsec' % ch2_cdm_med[0]
    ax2.scatter(derot_tdb['I2'], derot_cdm['I2'][:,0], label='ch2 CD11', **skw)
    ax2.scatter(derot_tdb['I2'], derot_cdm['I2'][:,3], label='ch2 CD22', **skw)
    ax2.axvline(_cryo_warm_cutoff, ls='--', c='r', label='cryo/warm')
    ax2.axhline(ch2_cdm_med[0], ls='--', c='g', label=ch2_cd11_lab)
    ax2.set_ylabel('Plate Scale (arcsec / pix)')
    ax2.legend(loc='upper right')

ax2.set_xlabel("JD (TDB)")

plot_name = 'joint_results.png'
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')

## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##

fig2 = plt.figure(2, figsize=fig_dims)
fig2.clf()
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212, sharex=ax1)
ax1.grid(True)
ax2.grid(True)


if len(derot_cdm['I1']):
    ch1_cd11_lab = '%.5f arcsec' % ch1_cdm_med[0]
    ax1.scatter(derot_obj['I1'], derot_cdm['I1'][:,0], label='ch1 CD11', **skw)
    ax1.scatter(derot_obj['I1'], derot_cdm['I1'][:,3], label='ch1 CD22', **skw)
    #ax1.axvline(_cryo_warm_cutoff, ls='--', c='r', label='cryo/warm')
    ax1.axhline(hq_ch1_cdm_med[0], ls='--', c='g', label=ch1_cd11_lab)
    ax1.set_ylabel('Plate Scale (arcsec / pix)')
    ax1.legend(loc='upper right')
if len(derot_cdm['I2']):
    ch2_cd11_lab = '%.5f arcsec' % ch2_cdm_med[0]
    ax2.scatter(derot_obj['I2'], derot_cdm['I2'][:,0], label='ch2 CD11', **skw)
    ax2.scatter(derot_obj['I2'], derot_cdm['I2'][:,3], label='ch2 CD22', **skw)
    #ax2.axvline(_cryo_warm_cutoff, ls='--', c='r', label='cryo/warm')
    ax2.axhline(hq_ch2_cdm_med[0], ls='--', c='g', label=ch2_cd11_lab)
    ax2.set_ylabel('Plate Scale (arcsec / pix)')
    ax2.legend(loc='upper right')

#ax2.set_xlabel('Objects')
ax2.set_xlabel(nobj_label)

#plot_name = 'joint_results_nobj.png'
fig2.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig2.savefig(nobj_plot, bbox_inches='tight')


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

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (joint_plot.py):
#---------------------------------------------------------------------
#
#  2023-01-25:
#     -- Increased __version__ to 0.0.1.
#     -- First created joint_plot.py.
#

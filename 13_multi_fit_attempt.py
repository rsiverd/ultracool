#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Take a crack at multi-dataset parallax fitting.
#
# Rob Siverd
# Created:       2021-04-13
# Last modified: 2021-04-15
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
from matplotlib.backends.backend_pdf import PdfPages
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
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
import theil_sen as ts
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

## This third version allows the user to specify keyword choice/order if
## desired but defaults to the keys provided in the first dictionary.
def recarray_from_dicts(list_of_dicts, use_keys=None):
    keys = use_keys if use_keys else list_of_dicts[0].keys()
    data = [np.array([d[k] for d in list_of_dicts]) for k in keys]
    return np.core.records.fromarrays(data, names=','.join(keys))


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
    Multi-dataset astrometry fitting system.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-s', '--site',
    #        help='Site to retrieve data for', required=True)
    #parser.add_argument('-n', '--number_of_days', default=1,
    #        help='Number of days of data to retrieve.')
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
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

##--------------------------------------------------------------------------##

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
## Input data files:
tgt_name = '2m0415'
#tgt_name = 'wise1828'
ch1_file = 'process/%s_ch1_pcat.pickle' % tgt_name
ch2_file = 'process/%s_ch2_pcat.pickle' % tgt_name
#ch1_file = 'process/wise1828_ch1_pcat.pickle'
#ch2_file = 'process/wise1828_ch2_pcat.pickle'
pkl_files = {'ch1':ch1_file, 'ch2':ch2_file}

## Load data:
tdata, sdata, gdata = {}, {}, {}
for tag,filename in pkl_files.items():
    with open(filename, 'rb') as pf:
        tdata[tag], sdata[tag], gdata[tag] = pickle.load(pf)

## Collect sources:
srcs_ch1 = set(sdata['ch1'].keys())
srcs_ch2 = set(sdata['ch2'].keys())
srcs_both = srcs_ch1.intersection(srcs_ch2)

##--------------------------------------------------------------------------##
## Count data points per set:
npts_ch1 = {x:len(sdata['ch1'][x]) for x in sdata['ch1'].keys()}
npts_ch2 = {x:len(sdata['ch2'][x]) for x in sdata['ch2'].keys()}

min_pts = 25
large_ch1 = [ss for ss,nn in npts_ch1.items() if nn>min_pts]
large_ch2 = [ss for ss,nn in npts_ch2.items() if nn>min_pts]

every_ch1_iname = np.concatenate([sdata['ch1'][x]['iname'] for x in large_ch1])
every_ch1_jdtdb = np.concatenate([sdata['ch1'][x]['jdtdb'] for x in large_ch1])
every_ch2_iname = np.concatenate([sdata['ch2'][x]['iname'] for x in large_ch2])
every_ch2_jdtdb = np.concatenate([sdata['ch2'][x]['jdtdb'] for x in large_ch2])

im2jd_ch1 = dict(zip(every_ch1_iname, every_ch1_jdtdb))
im2jd_ch2 = dict(zip(every_ch2_iname, every_ch2_jdtdb))
jd2im_ch1 = dict(zip(every_ch1_jdtdb, every_ch1_iname))
jd2im_ch2 = dict(zip(every_ch2_jdtdb, every_ch2_iname))

#large_ch1_inames = np.unique(every_ch1_iname)
#large_ch2_inames = np.unique(every_ch2_iname)

large_ch1_inames = [jd2im_ch1[x] for x in sorted(jd2im_ch1.keys())]
large_ch2_inames = [jd2im_ch2[x] for x in sorted(jd2im_ch2.keys())]

large_both = set(large_ch1).intersection(large_ch2)

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
## Results for channels 1/2:
sys.stderr.write("Performing 4-parameter fits ... ")
results_ch1 = {x:fit_4par(sdata['ch1'][x]) for x in large_ch1}
results_ch2 = {x:fit_4par(sdata['ch2'][x]) for x in large_ch2}
sys.stderr.write("done.\n")


## Per-source residual storage:
src_resids_ch1 = {x:[] for x in large_ch1}
src_resids_ch2 = {x:[] for x in large_ch2}

## Per-image residual storage:
#resid_data_all = {}
resid_data_ch1 = {x:[] for x in large_ch1_inames}
resid_data_ch2 = {x:[] for x in large_ch2_inames}

## Calculate residuals and divvy by image:
lookie_snr_ch1 = []
sys.stderr.write("Calculating ch1 residuals ... ")
for sid in large_ch1:
    stmp = sdata['ch1'][sid]
    model = results_ch1[sid]
    sra, sde = eval_4par(stmp, model)
    cos_dec = np.cos(np.radians(model['de_deg']))
    delta_ra_mas = 3.6e6 * (stmp[_ra_key] - sra) * cos_dec
    delta_de_mas = 3.6e6 * (stmp[_de_key] - sde)
    for iname,xpix,ypix,expt,rmiss,dmiss in zip(stmp['iname'],
            stmp['wx'], stmp['wy'], stmp['exptime'],
            delta_ra_mas, delta_de_mas):
        resid_data_ch1[iname].append({'x':xpix, 'y':ypix,
            'ra_err':rmiss, 'de_err':dmiss, 'exptime':expt})
    delta_tot_mas = np.sqrt(delta_ra_mas**2 + delta_de_mas**2)
    med_flux, iqr_flux = rs.calc_ls_med_IQR(stmp['flux'])
    med_resd, iqr_resd = rs.calc_ls_med_IQR(delta_tot_mas)
    med_signal = med_flux * stmp['exptime']
    #snr_scale = med_flux * np.sqrt(stmp['exptime'])
    snr_scale = np.sqrt(med_signal)
    approx_fwhm = delta_tot_mas * snr_scale
    med_fwhm, iqr_fwhm = rs.calc_ls_med_IQR(approx_fwhm)
    npoints = len(stmp['flux'])
    lookie_snr_ch1.append((med_flux, med_fwhm, npoints, 
        snr_scale[0], med_resd))
    sys.stderr.write("approx_fwhm: %s\n" % str(approx_fwhm))
sys.stderr.write("done.\n")

## Calculate residuals and divvy by image:
lookie_snr_ch2 = []
sys.stderr.write("Calculating ch2 residuals ... ")
for sid in large_ch2:
    stmp = sdata['ch2'][sid]
    model = results_ch2[sid]
    sra, sde = eval_4par(stmp, model)
    cos_dec = np.cos(np.radians(model['de_deg']))
    delta_ra_mas = 3.6e6 * (stmp[_ra_key] - sra) * cos_dec
    delta_de_mas = 3.6e6 * (stmp[_de_key] - sde)
    for iname,xpix,ypix,expt,rmiss,dmiss in zip(stmp['iname'],
            stmp['wx'], stmp['wy'], stmp['exptime'],
            delta_ra_mas, delta_de_mas):
        resid_data_ch2[iname].append({'x':xpix, 'y':ypix,
            'ra_err':rmiss, 'de_err':dmiss, 'exptime':expt})
    delta_tot_mas = np.sqrt(delta_ra_mas**2 + delta_de_mas**2)
    med_flux, iqr_flux = rs.calc_ls_med_IQR(stmp['flux'])
    med_resd, iqr_resd = rs.calc_ls_med_IQR(delta_tot_mas)
    med_signal = med_flux * stmp['exptime']
    snr_scale = np.sqrt(med_signal)
    approx_fwhm = delta_tot_mas * snr_scale
    med_fwhm, iqr_fwhm = rs.calc_ls_med_IQR(approx_fwhm)
    npoints = len(stmp['flux'])
    lookie_snr_ch2.append((med_flux, med_fwhm, npoints, 
        snr_scale[0], med_resd))
    sys.stderr.write("approx_fwhm: %s\n" % str(approx_fwhm))
sys.stderr.write("done.\n")

## Promote dictionary lists to recarrays:
resid_data_ch1 = {kk:recarray_from_dicts(vv) for kk,vv in resid_data_ch1.items()}
resid_data_ch2 = {kk:recarray_from_dicts(vv) for kk,vv in resid_data_ch2.items()}

## Save lookie_snr vectors to file for further analysis:

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
def get_baseline(data, jdkey='jdtdb'):
    return data[jdkey].max() - data[jdkey].min()

## Comparison of ch1/ch2 fit and residuals:
_pm_keys = ('pmra_degyr', 'pmde_degyr')
_ch_keys = ('ch1', 'ch2')
_save_me = ('sid', 'pmra_jnt', 'pmde_jnt', 'pmra_ch1', 'pmde_ch1', 
                'pmra_ch2', 'pmde_ch2')
many_things = []
for sid in large_both:
    sys.stderr.write("sid: %s\n" % sid)
    ch1_fit = results_ch1[sid]
    ch2_fit = results_ch2[sid]
    ch1_data = sdata['ch1'][sid]
    ch2_data = sdata['ch2'][sid]
    ch1_npts = len(ch1_data)
    ch2_npts = len(ch2_data)
    npoints  = {cc:len(sdata[cc][sid]) for cc in _ch_keys}
    baseline = {cc:get_baseline(sdata[cc][sid]) for cc in _ch_keys}
    jnt_data = np.hstack((ch1_data, ch2_data))
    jnt_fit  = fit_4par(jnt_data)
    things = {'sid':sid}
    things['pmra_ch1'] = ch1_fit['pmra_degyr'] * 3.6e6
    things['pmde_ch1'] = ch1_fit['pmde_degyr'] * 3.6e6
    things['pmra_ch2'] = ch2_fit['pmra_degyr'] * 3.6e6
    things['pmde_ch2'] = ch2_fit['pmde_degyr'] * 3.6e6
    things['pmra_jnt'] = jnt_fit['pmra_degyr'] * 3.6e6
    things['pmde_jnt'] = jnt_fit['pmde_degyr'] * 3.6e6
    many_things.append(things)
prmot = recarray_from_dicts(many_things)


fig_dims = (10, 10)
fig = plt.figure(2, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(211); ax1.grid(True)
ax2 = fig.add_subplot(212); ax2.grid(True)
pmra_diff = prmot['pmra_ch1']-prmot['pmra_ch2']
pmde_diff = prmot['pmde_ch1']-prmot['pmde_ch2']
ax1.hist(pmra_diff, range=(-200, 200), bins=11)
ax1.set_title('pmRA (CH1) - pmRA (CH2)')
ax1.set_xlabel('delta pmRA (mas)')
#ax2.set_xlim(-2, 82)
ax2.hist(pmde_diff, range=(-200, 200), bins=11)
ax2.set_xlabel('delta pmDE (mas)')
fig.tight_layout()
fig.savefig('prmot_1v2.png')

fig.clf()
ax1 = fig.add_subplot(211); ax1.grid(True)
ax2 = fig.add_subplot(212); ax2.grid(True)
ax1.scatter(np.abs(prmot['pmra_jnt']), np.abs(pmra_diff))
ax1.set_xlabel('|pmra_jnt|')
ax1.set_ylabel('pmra_diff')
ax2.scatter(np.abs(prmot['pmde_jnt']), np.abs(pmde_diff))
ax2.set_xlabel('|pmde_jnt|')
ax2.set_ylabel('pmde_diff')
ax1.set_ylim(-20, 200)
ax2.set_ylim(-20, 200)
ax1.set_xlim(-2, 82)
ax2.set_xlim(-2, 82)
fig.tight_layout()
fig.savefig('prmot_diff_vs_prmot.png')

#sys.exit(0)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

fig_dims = (10, 11)

def limify(ax):
    ax.set_xlim(0.2, 50.)
    ax.set_ylim(500., 2500.)

## CH1 FWHM:
flx, mfwhm, npoints, msnr, mresd = np.array(lookie_snr_ch1).T
medmed_fwhm = np.median(mfwhm)
mmf_txt = 'median: %.1f mas' % medmed_fwhm
fig = plt.figure(3, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(211); ax1.grid(True)
ax1.set_title("IRAC channel 1")
spts = ax1.scatter(flx, mfwhm, c=npoints)
ax1.axhline(medmed_fwhm, c='r', ls='--', label=mmf_txt)
limify(ax1)
ax1.set_xscale('log')
ax1.set_xlabel('med_flux')
ax1.set_ylabel('FWHM (mas)')
ax1.legend(loc='upper right')
cbnorm = mplcolors.Normalize(*spts.get_clim())
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
cbar.formatter.set_useOffset(False)
cbar.update_ticks()
cbar.set_label('Data Points')
ax2 = fig.add_subplot(212); ax2.grid(True)
spts = ax2.scatter(msnr, mresd, c=npoints)
ax2.set_ylabel('resid (mas)')
ax2.set_xlabel('med_snr')
ax2.set_xscale('log')
ax2.set_yscale('log')
#ax2.set_xlim(0.9, 110.)
ax2.set_xlim(0.9, 250.)
ax2.set_ylim(30., 2500.)
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
fig.tight_layout()
plt.draw()
fig.savefig('empirical_scatter_ch1.png')

## CH2 FWHM:
flx, mfwhm, npoints, msnr, mresd = np.array(lookie_snr_ch2).T
medmed_fwhm = np.median(mfwhm)
mmf_txt = 'median: %.1f mas' % medmed_fwhm
fig = plt.figure(4, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(211); ax1.grid(True)
ax1.set_title("IRAC channel 2")
spts = ax1.scatter(flx, mfwhm, c=npoints)
ax1.axhline(medmed_fwhm, c='r', ls='--', label=mmf_txt)
limify(ax1)
#ax1.set_xlim(0.2, 400.)
ax1.set_xscale('log')
ax1.set_xlabel('med_flux')
ax1.set_ylabel('FWHM (mas)')
ax1.legend(loc='upper right')
cbnorm = mplcolors.Normalize(*spts.get_clim())
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
cbar.formatter.set_useOffset(False)
cbar.update_ticks()
cbar.set_label('Data Points')
ax2 = fig.add_subplot(212); ax2.grid(True)
spts = ax2.scatter(msnr, mresd, c=npoints)
ax2.set_ylabel('resid (mas)')
ax2.set_xlabel('med_snr')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(0.9, 250.)
ax2.set_ylim(30., 2500.)
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
fig.tight_layout()
plt.draw()
fig.savefig('empirical_scatter_ch2.png')


sys.exit(0)


##--------------------------------------------------------------------------##
## Multi-page PDF of quiver plots:
fig_dims = (12, 10)

sys.stderr.write("\nAttempting multi-page PDF (ch1) ... \n")
#resid_data = resid_data_ch1
total1 = len(resid_data_ch1)
with PdfPages('ch1_4par_residuals.pdf') as pdf:
    for ii,imname in enumerate(large_ch1_inames, 1):
        sys.stderr.write("\r%s (image %d of %d) ...   " % (imname, ii, total1))
        iresid = resid_data_ch1[imname]
        fig = plt.figure(3, figsize=fig_dims)
        fig.clf()
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.grid(True)
        #ax1.scatter(iresid['x'], iresid['y'])
        ax1.quiver(iresid['x'], iresid['y'], iresid['ra_err'], iresid['de_err'])
        ax1.set_xlim(0, 260)
        ax1.set_ylim(0, 260)
        ax1.set_xlabel('X pixel')
        ax1.set_ylabel('Y pixel')
        ax1.set_title(imname)
        fig.tight_layout()
        plt.draw()
        pdf.savefig()
        plt.close()
sys.stderr.write("done.\n")


sys.stderr.write("\nAttempting multi-page PDF (ch2) ... \n")
total2 = len(resid_data_ch2)
with PdfPages('ch2_4par_residuals.pdf') as pdf:
    for ii,imname in enumerate(large_ch2_inames, 1):
        sys.stderr.write("\r%s (image %d of %d) ...   " % (imname, ii, total2))
        iresid = resid_data_ch2[imname]
        fig = plt.figure(3, figsize=fig_dims)
        fig.clf()
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.grid(True)
        #ax1.scatter(iresid['x'], iresid['y'])
        ax1.quiver(iresid['x'], iresid['y'], iresid['ra_err'], iresid['de_err'])
        ax1.set_xlim(0, 260)
        ax1.set_ylim(0, 260)
        ax1.set_xlabel('X pixel')
        ax1.set_ylabel('Y pixel')
        ax1.set_title(imname)
        fig.tight_layout()
        plt.draw()
        pdf.savefig()
        plt.close()
sys.stderr.write("done.\n")

sys.exit(0)

fig = plt.figure(3)
total = len(resid_data)
#for ii,(imname,iresid) in enumerate(resid_data.items(), 1):
for ii,imname in enumerate(large_ch1_inames, 1):
    sys.stderr.write("%s (image %d of %d) ...   \n" % (imname, ii, total))
    iresid = resid_data[imname]
    fig.clf()
    ax1 = fig.add_subplot(111, aspect='equal')
    ax1.grid(True)
    #ax1.scatter(iresid['x'], iresid['y'])
    ax1.quiver(iresid['x'], iresid['y'], iresid['ra_err'], iresid['de_err'])
    ax1.set_xlim(0, 260)
    ax1.set_ylim(0, 260)
    ax1.set_title(imname)
    fig.tight_layout()
    plt.draw()
    sys.stderr.write("press ENTER to continue ...\n")
    response = input()

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
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
ax2.grid(True)
#ax1.axis('off')

for tag,data in tdata.items():
    _ra, _de, _jd = data[_ra_key], data[_de_key], data['jdtdb']
    ax1.scatter(_jd, _ra, s=15, lw=0, label=tag)
    ax2.scatter(_jd, _de, s=15, lw=0, label=tag)


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

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')




######################################################################
# CHANGELOG (13_multi_fit_attempt.py):
#---------------------------------------------------------------------
#
#  2021-04-13:
#     -- Increased __version__ to 0.0.1.
#     -- First created 13_multi_fit_attempt.py.
#
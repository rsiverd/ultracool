#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Take a crack at multi-dataset parallax fitting.
#
# Rob Siverd
# Created:       2021-04-13
# Last modified: 2021-12-13
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
__version__ = "0.3.0"

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
import ast
import sys
import time
import pickle
import numpy as np
from numpy.lib.recfunctions import append_fields
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
import pandas as pd
import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Astrometry fitting module:
import astrom_test_2
reload(astrom_test_2)
at2 = astrom_test_2
af = at2.AstFit()   # used for target
afn = at2.AstFit()  # used for neighbors

## Cartesian rotations (testing):
import fov_rotation
reload(fov_rotation)
rfov = fov_rotation.RotateFOV()
r3d  = fov_rotation.Rotate3D()

## Spitzer error model:
import spitz_error_model
reload(spitz_error_model)
sem = spitz_error_model.SpitzErrorModel()

## MCMC sampler:
import emcee
import corner

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

## This routine streamlines addition of new columns to record arrays:
def add_recarray_columns(recarray, column_specs):
    """
    Add new fields/columns to a structured numpy array. The 'column_specs'
    input is a list of tuples containing names and data for appending.
    Example column_specs: [('col1', c1array), ('col2', c2array)]
    """

    result = recarray.copy()
    for cname, cdata in column_specs:
        result = append_fields(result, cname, cdata, usemask=False)
    return result

##--------------------------------------------------------------------------##
##------------------         Terminal Fanciness             ----------------##
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
    #parser.set_defaults(target=None)
    # ------------------------------------------------------------------
    parser.add_argument('-t', '--target', default=None, required=False,
            help='specify a target to process', type=str)
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
cat_type = 'pcat'
#cat_type = 'fcat'
tgt_name = '2m0415'
#tgt_name = '2m0729'
#tgt_name = 'pso043'
#tgt_name = 'ross458c'
#tgt_name = 'ugps0722'
#tgt_name = 'wise0148'
#tgt_name = 'wise0410'
#tgt_name = 'wise0458'
#tgt_name = 'wise1405'
#tgt_name = 'wise1541'
#tgt_name = 'wise1738'
#tgt_name = 'wise1741'
#tgt_name = 'wise1804'
#tgt_name = 'wise1828'
#tgt_name = 'wise2056'
sys.stderr.write("context.target: %s\n" % str(context.target))
if context.target:
    tgt_name = context.target
    sys.stderr.write("Using tgt_name %s from command line!\n" % tgt_name)
#sys.exit(0)
ch1_file = 'process/%s_ch1_%s.pickle' % (tgt_name, cat_type)
ch2_file = 'process/%s_ch2_%s.pickle' % (tgt_name, cat_type)
#ch1_file = 'process/wise1828_ch1_pcat.pickle'
#ch2_file = 'process/wise1828_ch2_pcat.pickle'
pkl_files = {'ch1':ch1_file, 'ch2':ch2_file}

plot_tag = '%s_%s_%s' % (centroid_method, tgt_name, cat_type)

## Load parameters file:
par_file = 'targets/%s.par' % tgt_name
if not os.path.isfile(par_file):
    sys.stderr.write("Can't find parameter file: %s\n" % par_file)
    sys.exit(1)
with open(par_file, 'r') as ff:
    prev_pars = ast.literal_eval(ff.read())

## Load data:
tdata, sdata, gdata = {}, {}, {}
for tag,filename in pkl_files.items():
    with open(filename, 'rb') as pp:
        tdata[tag], sdata[tag], gdata[tag] = pickle.load(pp)

## Collect sources:
srcs_ch1 = set(sdata['ch1'].keys())
srcs_ch2 = set(sdata['ch2'].keys())
srcs_both = srcs_ch1.intersection(srcs_ch2)

## Ensure presence of instrument keyword everywhere:
sys.stderr.write("Ensuring presence of 'instrument' keyword ... \n")
for sid,ss in sdata['ch1'].items():
    if not ('instrument' in ss.dtype.names):
        #sys.stderr.write("Adding IRAC1 to %s ...\n" % sid)
        new_fields = [('instrument', ['IRAC1' for x in range(len(ss))])]
        sdata['ch1'][sid] = add_recarray_columns(ss, new_fields)
for sid,ss in sdata['ch2'].items():
    if not ('instrument' in ss.dtype.names):
        #sys.stderr.write("Adding IRAC2 to %s ...\n" % sid)
        new_fields = [('instrument', ['IRAC2' for x in range(len(ss))])]
        sdata['ch2'][sid] = add_recarray_columns(ss, new_fields)
sys.stderr.write("Instrument keywords updated.\n")

##--------------------------------------------------------------------------##
## Count data points per set:
npts_ch1 = {x:len(sdata['ch1'][x]) for x in sdata['ch1'].keys()}
npts_ch2 = {x:len(sdata['ch2'][x]) for x in sdata['ch2'].keys()}

nmax_ch1 = max(npts_ch1.values())
nmax_ch2 = max(npts_ch2.values())

min_pts = 25
min_pts_1 = nmax_ch1 // 2
min_pts_2 = nmax_ch2 // 2
#min_pts_1 = min_pts
#min_pts_2 = min_pts
large_ch1 = [ss for ss,nn in npts_ch1.items() if nn>min_pts_1]
large_ch2 = [ss for ss,nn in npts_ch2.items() if nn>min_pts_2]

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
jd2im_both = {**jd2im_ch1, **jd2im_ch2}

##--------------------------------------------------------------------------##

## The following should probably be weighted by errors in a future version:
full_jdtdb_vector = np.array(list(jd2im_both.keys()))
mean_jdtdb = np.average(full_jdtdb_vector)
first_jdtdb = full_jdtdb_vector.min()

j2000_epoch = astt.Time('2000-01-01T12:00:00', scale='tt', format='isot')
j2000_epoch = astt.Time(first_jdtdb, scale='tdb', format='jd')
j2000_epoch = astt.Time(mean_jdtdb, scale='tdb', format='jd')

def fit_4par(data, debug=False):
    years = (data['jdtdb'] - j2000_epoch.tdb.jd) / 365.25
    ts_ra_model = ts.linefit(years, data[_ra_key])
    ts_de_model = ts.linefit(years, data[_de_key])
    if debug:
        sys.stderr.write("fit_4par --> years:\n\n--> %s\n" % str(years))
    return {'epoch_jdtdb'  :   j2000_epoch.tdb.jd,
                 'ra_deg'  :   ts_ra_model[0],
                 'de_deg'  :   ts_de_model[0],
             'pmra_degyr'  :   ts_ra_model[1],
             'pmde_degyr'  :   ts_de_model[1],
            }

def eval_4par(data, model, debug=False):
    years = (data['jdtdb'] - model['epoch_jdtdb']) / 365.25
    calc_ra = model['ra_deg'] + years * model['pmra_degyr']
    calc_de = model['de_deg'] + years * model['pmde_degyr']
    if debug:
        sys.stderr.write("years: %s\n" % str(years))
    return calc_ra, calc_de


def radec_plx_factors(RA_rad, DE_rad, X_au, Y_au, Z_au):
    """Compute parallax factors in arcseconds."""
    sinRA, cosRA = np.sin(RA_rad), np.cos(RA_rad)
    sinDE, cosDE = np.sin(DE_rad), np.cos(DE_rad)
    ra_factor = (X_au * sinRA - Y_au * cosRA) / cosDE
    de_factor =  X_au * cosRA * sinDE \
              +  Y_au * sinRA * sinDE \
              -  Z_au * cosDE
    return ra_factor, de_factor

#def theilsen_fit_5par(data):
#    model = 

##--------------------------------------------------------------------------##
##------------------          parallax factor drawing       ----------------##
##--------------------------------------------------------------------------##

eph_file = 'ephemerides/sst_eph_dense.csv'
full_eph = pd.read_csv(eph_file, low_memory=False)

def calc_full_plxf(ra_deg, de_deg, plx_mas):
    rra, rde = np.radians(ra_deg), np.radians(de_deg)
    plxr, plxd = radec_plx_factors(rra, rde,
            full_eph['obs_x'], full_eph['obs_y'], full_eph['obs_z'])
    plxr *= np.cos(np.radians(de_deg))
    return full_eph['jdtdb'], plx_mas*plxr, plx_mas*plxd


ssbx = full_eph.obs_x.values
ssby = full_eph.obs_y.values
ssbz = full_eph.obs_z.values

ssbpos = np.vstack((ssbx, ssby, ssbz))

tilt = np.average(np.arctan(ssbz / ssby))

flatpos = np.array(r3d.xrot(-tilt, ssbpos))

new_z = flatpos

#sys.exit(0)

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

## Residual calculation method:
do_robust_resid = True
do_robust_resid = False

## Calculate residuals and divvy by image:
lookie_snr_ch1 = []
inspection_ch1 = []
scatter_time_ch1 = []
sys.stderr.write("Calculating ch1 residuals ... ")
for sid in large_ch1:
    stmp = sdata['ch1'][sid]
    model = results_ch1[sid]
    sra, sde = eval_4par(stmp, model)
    cos_dec = np.cos(np.radians(model['de_deg']))
    delta_ra_mas = 3.6e6 * (stmp[_ra_key] - sra) * cos_dec
    delta_de_mas = 3.6e6 * (stmp[_de_key] - sde)
    src_resids_ch1[sid] = np.array([stmp['jdtdb'], delta_ra_mas, delta_de_mas])
    scatter_time_ch1.extend(list(zip(stmp['jdtdb'], 
                        delta_ra_mas, delta_de_mas)))
                        #delta_ra_mas, delta_de_mas, [sid for x in sra])))
    for iname,xpix,ypix,expt,rmiss,dmiss in zip(stmp['iname'],
            stmp['wx'], stmp['wy'], stmp['exptime'],
            delta_ra_mas, delta_de_mas):
        resid_data_ch1[iname].append({'x':xpix, 'y':ypix,
            'ra_err':rmiss, 'de_err':dmiss, 'exptime':expt})
    delta_tot_mas = np.sqrt(delta_ra_mas**2 + delta_de_mas**2)
    med_flux, iqr_flux = rs.calc_ls_med_IQR(stmp['flux'])
    if do_robust_resid:
        med_resd, iqr_resd = rs.calc_ls_med_IQR(delta_tot_mas)
    else:
        med_resd, iqr_resd = np.average(delta_tot_mas), np.std(delta_tot_mas)
    #med_resd = iqr_flux
    med_signal = med_flux * stmp['exptime']
    #snr_scale = med_flux * np.sqrt(stmp['exptime'])
    snr_scale = np.sqrt(med_signal)
    approx_fwhm = delta_tot_mas * snr_scale
    med_fwhm, iqr_fwhm = rs.calc_ls_med_IQR(approx_fwhm)
    npoints = len(stmp['flux'])
    typical_ra = np.median(stmp['dra'])
    typical_de = np.median(stmp['dde'])
    jdtdb_mean = np.average(stmp['jdtdb'])
    jdtdb_sdev = np.std(stmp['jdtdb'])
    lookie_snr_ch1.append((med_flux, med_fwhm, npoints, 
        snr_scale[0], med_resd, typical_ra, typical_de,
        jdtdb_mean, jdtdb_sdev, iqr_flux))
    inspection_ch1.extend(list(zip(delta_tot_mas, stmp['flux'],
        stmp['flux']*stmp['exptime'])))
    #sys.stderr.write("approx_fwhm: %s\n" % str(approx_fwhm))
sys.stderr.write("done.\n")

## Calculate residuals and divvy by image:
lookie_snr_ch2 = []
inspection_ch2 = []
scatter_time_ch2 = []
#lookie_pos_ch2 = []
sys.stderr.write("Calculating ch2 residuals ... ")
for sid in large_ch2:
    stmp = sdata['ch2'][sid]
    model = results_ch2[sid]
    sra, sde = eval_4par(stmp, model)
    cos_dec = np.cos(np.radians(model['de_deg']))
    delta_ra_mas = 3.6e6 * (stmp[_ra_key] - sra) * cos_dec
    delta_de_mas = 3.6e6 * (stmp[_de_key] - sde)
    src_resids_ch2[sid] = np.array([stmp['jdtdb'], delta_ra_mas, delta_de_mas])
    scatter_time_ch2.extend(list(zip(stmp['jdtdb'], 
                        delta_ra_mas, delta_de_mas)))
                        #delta_ra_mas, delta_de_mas, [sid for x in sra])))
    for iname,xpix,ypix,expt,rmiss,dmiss in zip(stmp['iname'],
            stmp['wx'], stmp['wy'], stmp['exptime'],
            delta_ra_mas, delta_de_mas):
        resid_data_ch2[iname].append({'x':xpix, 'y':ypix,
            'ra_err':rmiss, 'de_err':dmiss, 'exptime':expt})
    delta_tot_mas = np.sqrt(delta_ra_mas**2 + delta_de_mas**2)
    med_flux, iqr_flux = rs.calc_ls_med_IQR(stmp['flux'])
    if do_robust_resid:
        med_resd, iqr_resd = rs.calc_ls_med_IQR(delta_tot_mas)
    else:
        med_resd, iqr_resd = np.average(delta_tot_mas), np.std(delta_tot_mas)
    med_signal = med_flux * stmp['exptime']
    snr_scale = np.sqrt(med_signal)
    approx_fwhm = delta_tot_mas * snr_scale
    med_fwhm, iqr_fwhm = rs.calc_ls_med_IQR(approx_fwhm)
    npoints = len(stmp['flux'])
    typical_ra = np.median(stmp['dra'])
    typical_de = np.median(stmp['dde'])
    jdtdb_mean = np.average(stmp['jdtdb'])
    jdtdb_sdev = np.std(stmp['jdtdb'])
    lookie_snr_ch2.append((med_flux, med_fwhm, npoints, 
        snr_scale[0], med_resd, typical_ra, typical_de,
        jdtdb_mean, jdtdb_sdev, iqr_flux))
    inspection_ch2.extend(list(zip(delta_tot_mas, stmp['flux'],
        stmp['flux']*stmp['exptime'])))
    #lookie_pos_ch2.append((med_flux, med_fwhm, npoints,
    #    snr_scale[0], med_resd, typical_ra, typical_de))
    #sys.stderr.write("approx_fwhm: %s\n" % str(approx_fwhm))
sys.stderr.write("done.\n")

## Promote dictionary lists to recarrays:
resid_data_ch1 = {kk:recarray_from_dicts(vv) for kk,vv in resid_data_ch1.items()}
resid_data_ch2 = {kk:recarray_from_dicts(vv) for kk,vv in resid_data_ch2.items()}

## Promote residual data to arrays:
scatter_time_ch1 = np.array(scatter_time_ch1)
scatter_time_ch2 = np.array(scatter_time_ch2)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Well-formatted printing of 4-par models:
def nice_print_4par(model, stream=sys.stderr):
    pmde_masyr = model['pmde_degyr'] * 3.6e6
    pmra_masyr = model['pmra_degyr'] * 3.6e6
    pmra_cosdec_masyr = pmra_masyr * np.cos(np.radians(model['de_deg']))
    stream.write("Epoch (JDTDB):  %.1f\n" % model['epoch_jdtdb'])
    stream.write("RA (degrees):   %12.6f\n" % model['ra_deg'])
    stream.write("DE (degrees):   %12.6f\n" % model['de_deg'])
    stream.write("pmRA  (mas/yr): %12.6f\n" % pmra_masyr)
    stream.write("pmRA* (mas/yr): %12.6f\n" % pmra_cosdec_masyr)
    stream.write("pmDE  (mas/yr): %12.6f\n" % pmde_masyr)
    return

def nice_print_prev(params, stream=sys.stderr):
    pmra_cosdec_masyr = 1e3 * params['pmra_cosdec_asyr']
    pmde_masyr = 1e3 * params['pmde_asyr']
    prlx_mas = 1e3 * params['parallax_as']
    stream.write("pmRA* (mas/yr): %9.3f\n" % pmra_cosdec_masyr)
    stream.write("pmDE  (mas/yr): %9.3f\n" % pmde_masyr)
    stream.write("prlx  (mas):    %9.3f\n" % prlx_mas)
    return

def less_first(data):
    return data - data[0]

## 5-parameter fit to target:
#order_ch1 = np.argsort(tdata['ch1']['jdtdb'])
#order_ch2 = np.argsort(tdata['ch2']['jdtdb'])
#tgt_ch1 = tdata['ch1'][order_ch1]
#tgt_ch2 = tdata['ch2'][order_ch2]
tgt_ch1 = tdata['ch1']
tgt_ch2 = tdata['ch2']
if not ('ra_err_mas' in tgt_ch1.dtype.names):
    sys.stderr.write("No errors, adding empirical ones!\n")
    signal_ch1 = tgt_ch1['flux'] * tgt_ch1['exptime']
    asterr_ch1 = sem.signal2error(signal_ch1, 1)
    new_fields = [('ra_err_mas', asterr_ch1), ('de_err_mas', asterr_ch1)]
    tgt_ch1 = add_recarray_columns(tgt_ch1, new_fields)

    signal_ch2 = tgt_ch2['flux'] * tgt_ch2['exptime']
    asterr_ch2 = sem.signal2error(signal_ch2, 2)
    new_fields = [('ra_err_mas', asterr_ch2), ('de_err_mas', asterr_ch2)]
    tgt_ch2 = add_recarray_columns(tgt_ch2, new_fields)
    #tgt_ch1 = append_fields(tgt_ch1, 'ra_err_mas', asterr_ch1, usemask=False)
    #tgt_ch1 = append_fields(tgt_ch1, 'de_err_mas', asterr_ch1, usemask=False)
    #tgt_ch2 = append_fields(tgt_ch2, 'ra_err_mas', asterr_ch2, usemask=False)
    #tgt_ch2 = append_fields(tgt_ch2, 'de_err_mas', asterr_ch2, usemask=False)
else:
    sys.stderr.write("Empirical errors already exist!\n")

## Add instrument tag if not already present:
if not ('instrument' in tgt_ch1.dtype.names):
    sys.stderr.write("No 'instrument' column found for ch1, adding!\n")
    new_fields = [('instrument', ['IRAC1' for x in range(len(tgt_ch1))])]
    tgt_ch1 = add_recarray_columns(tgt_ch1, new_fields)
if not ('instrument' in tgt_ch2.dtype.names):
    sys.stderr.write("No 'instrument' column found for ch2, adding!\n")
    new_fields = [('instrument', ['IRAC2' for x in range(len(tgt_ch2))])]
    tgt_ch2 = add_recarray_columns(tgt_ch2, new_fields)


tgt_both = np.concatenate((tgt_ch1, tgt_ch2))
ref_time = np.median(tgt_both['jdtdb'])
#del order_ch1, order_ch2

tmodel_4par = {cc:fit_4par(dd) for cc,dd in tdata.items()}
tmodel_4par['both'] = fit_4par(tgt_both)

## Select data set(s), sanity check for extra data points:
have_datasets = {'both':tgt_both, 'ch1':tgt_ch1, 'ch2':tgt_ch2}
dwhich = 'ch1'
dwhich = 'ch2'
dwhich = 'both'
use_dataset = have_datasets[dwhich]
uniq_imgs = len(set(use_dataset['iname']))
total_pts = len(use_dataset)
sys.stderr.write("Have %d data points and %d unique images.\n" 
        % (total_pts, uniq_imgs))

## Dictionary-based occurrence counter:
def count_occurrences(values):
    hdict = {x:0 for x in values}
    for x in values:
        hdict[x] += 1
    return hdict

## Counts per image/JD:
iname_hist = count_occurrences(use_dataset['iname'])
jdtdb_hist = count_occurrences(use_dataset['jdtdb'])

if (uniq_imgs != total_pts):
    sys.stderr.write("WARNING: non-unique images/timestamps!\n")
    sys.stderr.write("Duplicates come from / have:\n")
    dupe_iname = [x for x,n in iname_hist.items() if n>1]
    dupe_jdtdb = [x for x,n in jdtdb_hist.items() if n>1]
    dupe_index = np.nonzero(use_dataset['iname'] == dupe_iname[0])[0]
    dupe_data  = use_dataset[use_dataset['iname'] == dupe_iname[0]]
    sys.stderr.write("iname: %s\n" % str(dupe_iname))
    sys.stderr.write("jdtdb: %s\n" % str(dupe_jdtdb))
    sys.stderr.write("Press ENTER to continue ... \n")
    derp = input()

## Compute parallax factors (dRA*cos(DE), dDE):
ts_fit_4par = fit_4par(use_dataset, debug=False)
m4ra, m4de = eval_4par(use_dataset, ts_fit_4par, debug=False)
nom_ra_deg = np.average(m4ra)
nom_de_deg = np.average(m4de)
nom_ra_rad = np.radians(nom_ra_deg)
nom_de_rad = np.radians(nom_de_deg)
pfra, pfde = radec_plx_factors(np.radians(nom_ra_deg), np.radians(nom_ra_rad),
        use_dataset['obs_x'], use_dataset['obs_y'], use_dataset['obs_z'])

#sys.stderr.write("out here, pfra: %s\n" % str(pfra))

ra_resids_deg = use_dataset['dra'] - m4ra       # ==> prlx * pfra
ra_resids_deg *= np.cos(nom_de_rad)
de_resids_deg = use_dataset['dde'] - m4de       # ==> prlx * pfde

ra_resids_mas = 3.6e6 * ra_resids_deg
de_resids_mas = 3.6e6 * de_resids_deg

ra_plxval = 3.6e6 * ra_resids_deg / pfra
de_plxval = 3.6e6 * de_resids_deg / pfde
all_plxval = np.hstack((ra_plxval, de_plxval))

### TESTING -- get parallax and zero-point correction from linear fit:
#adj_raw_ra_resids_as = 3600. * (use_dataset['dra'] - m4ra)
#adj_raw_de_resids_as = 3600. * (use_dataset['dde'] - m4de)
#ts_ra_adjustment = ts.linefit(pfra, adj_raw_ra_resids_as)
#ts_de_adjustment = ts.linefit(pfde, adj_raw_de_resids_as)
#
### Get me the pmRA/pmDE in mas:
#def gimme_pm_mas(model):
#    cos_dec = np.cos(np.radians(model['de_deg']))
#    pmra_mas = 3.6e6 * model['pmra_degyr']
#    pmde_mas = 3.6e6 * model['pmde_degyr']
#    return pmra_mas*cos_dec, pmde_mas
#
### Zero-point agnostic parallax estimate:
#which_ra_lower = (pfra < 0.0)
#which_ra_upper = (pfra > 0.0)
#which_de_lower = (pfde < 0.0)
#which_de_upper = (pfde > 0.0)
#avg_pfra_lower = np.average(pfra[which_ra_lower])
#avg_pfra_upper = np.average(pfra[which_ra_upper])
#avg_pfde_lower = np.average(pfde[which_de_lower])
#avg_pfde_upper = np.average(pfde[which_de_upper])
#med_rres_lower = np.median(ra_resids_mas[which_ra_lower])
#med_rres_upper = np.median(ra_resids_mas[which_ra_upper])
#med_dres_lower = np.median(de_resids_mas[which_de_lower])
#med_dres_upper = np.median(de_resids_mas[which_de_upper])
#
#pfra_plx_guess = \
#        (med_rres_upper - med_rres_lower) / (avg_pfra_upper - avg_pfra_lower)
#pfde_plx_guess = \
#        (med_dres_upper - med_dres_lower) / (avg_pfde_upper - avg_pfde_lower)
#
#ts_fit_ra_lower = fit_4par(use_dataset[which_ra_lower])
#ts_fit_ra_upper = fit_4par(use_dataset[which_ra_upper])
#pmra_guess_lower = gimme_pm_mas(ts_fit_ra_lower)[0]
#pmra_guess_upper = gimme_pm_mas(ts_fit_ra_upper)[0]
#
#ts_fit_de_lower = fit_4par(use_dataset[which_de_lower])
#ts_fit_de_upper = fit_4par(use_dataset[which_de_upper])
#pmde_guess_lower = gimme_pm_mas(ts_fit_de_lower)[1]
#pmde_guess_upper = gimme_pm_mas(ts_fit_de_upper)[1]
#
### Various parallax estimations:
#fulldiv = 80 * '-'
#sys.stderr.write("\n%s\n" % fulldiv)
#sys.stderr.write("Target: %s (%s, %s)\n" % (tgt_name, dwhich, cat_type))
#nice_print_4par(ts_fit_4par)
#sys.stderr.write("\n")
#sys.stderr.write("Median plxval: %8.3f\n" % np.median(all_plxval))
#sys.stderr.write("plxval (RA-based, med): %8.3f\n" % np.median(ra_plxval))
#sys.stderr.write("plxval (RA-based, avg): %8.3f\n" % np.average(ra_plxval))
#sys.stderr.write("plxval (DE-based, med): %8.3f\n" % np.median(de_plxval))
#sys.stderr.write("plxval (DE-based, avg): %8.3f\n" % np.average(de_plxval))
#sys.stderr.write("\nPrev:\n")
#nice_print_prev(prev_pars)
#
#sys.stderr.write("\n%s\n" % fulldiv)
#sys.stderr.write("pmra_guess_lower: %8.3f\n" % pmra_guess_lower)
#sys.stderr.write("pmra_guess_upper: %8.3f\n" % pmra_guess_upper)
#sys.stderr.write("\n")
#sys.stderr.write("pmde_guess_lower: %8.3f\n" % pmde_guess_lower)
#sys.stderr.write("pmde_guess_upper: %8.3f\n" % pmde_guess_upper)
#sys.stderr.write("\n")
#sys.stderr.write("pfra_plx_guess:   %8.3f\n" % pfra_plx_guess)
#sys.stderr.write("pfde_plx_guess:   %8.3f\n" % pfde_plx_guess)
#sys.stderr.write("\n%s\n" % fulldiv)

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

## Errors to use in fitting:
median_cosdec = np.median(np.cos(np.radians(use_dataset['dde'])))
ra_deg_errs = use_dataset['ra_err_mas'] / 3.6e6 / median_cosdec
de_deg_errs = use_dataset['de_err_mas'] / 3.6e6
ra_rad_errs = np.radians(ra_deg_errs)
de_rad_errs = np.radians(de_deg_errs)

## Try out fitting now:
sigcut = 5
#af.setup(use_dataset)
af.setup(use_dataset, RA_err=ra_rad_errs, DE_err=de_rad_errs,
        jd_tdb_ref=j2000_epoch.tdb.jd)
bestpars = af.fit_bestpars(sigcut=sigcut)
firstpars = bestpars.copy()

## Iterate a bunch to better solution:
iterpars = af.iter_update_bestpars(bestpars)
for i in range(30):
    iterpars = af.iter_update_bestpars(iterpars)
bestpars = iterpars

## Best-fit parameters in sane units:
use_cos_dec = np.cos(bestpars[1])
best_dra = np.degrees(bestpars[0])
best_dde = np.degrees(bestpars[1])

## Inspect residuals:
raw_res_ra, raw_res_de = af._calc_radec_residuals(bestpars)
raw_res_ra_mas = 3.6e6 * np.degrees(raw_res_ra) * use_cos_dec
raw_res_de_mas = 3.6e6 * np.degrees(raw_res_de)

## Split by instrument:
which_ch1 = (use_dataset['instrument'] == 'IRAC1')
which_ch2 = (use_dataset['instrument'] == 'IRAC2')

## Medians for each:
med_ra_res_ch1 = np.median(raw_res_ra_mas[which_ch1])
med_ra_res_ch2 = np.median(raw_res_ra_mas[which_ch2])
med_de_res_ch1 = np.median(raw_res_de_mas[which_ch1])
med_de_res_ch2 = np.median(raw_res_de_mas[which_ch2])

## Make a plot:
hbins = len(raw_res_ra_mas) // 20
bspec = {'range':(-800, 800), 'bins':hbins, 'histtype':'step'}
fig_dims = (10, 10)
acfig = plt.figure(23, figsize=fig_dims)
acfig.clf()
acfra = acfig.add_subplot(211)
acfra.set_title(plot_tag)
acfra.grid(True)
acfra.hist(raw_res_ra_mas[which_ch1], color='b', label='ch1', **bspec)
acfra.hist(raw_res_ra_mas[which_ch2], color='g', label='ch2', **bspec)
acfra.axvline(med_ra_res_ch1, color='r', ls=':', 
        label='median RA (ch1): %.3f mas'%med_ra_res_ch1)
acfra.axvline(med_ra_res_ch2, color='m', ls=':', 
        label='median RA (ch2): %.3f mas'%med_ra_res_ch2)
acfra.set_xlabel('RA Residual (mas)')
acfra.legend(loc='upper left')
acfde = acfig.add_subplot(212)
acfde.grid(True)
acfde.hist(raw_res_de_mas[which_ch1], color='b', label='ch1', **bspec)
acfde.hist(raw_res_de_mas[which_ch2], color='g', label='ch2', **bspec)
acfde.axvline(med_ra_res_ch1, color='r', ls=':', 
        label='median DE (ch1): %.3f mas'%med_de_res_ch1)
acfde.axvline(med_ra_res_ch2, color='m', ls=':', 
        label='median DE (ch2): %.3f mas'%med_de_res_ch2)
acfde.set_xlabel('DE Residual (mas)')
acfde.legend(loc='upper left')
acfig.tight_layout()
plt.draw()
pname = 'plots/residual_hist_%s.png' % plot_tag
pname = 'plots/residual_hist_%s.pdf' % plot_tag
acfig.savefig(pname)

#ra_resids, de_resids = af._calc_radec_residuals_sigma(bestpars)
#sys.exit(0)

## Get residuals from 4-parameter fit for plotting:
iter4par = iterpars.copy()
iter4par[4] = 0.0
ra_4p_res_rad, de_4p_res_rad = af._calc_radec_residuals(iter4par)
cosdec = np.cos(iter4par[1])
ra_resids_deg = np.degrees(ra_4p_res_rad) * cosdec
de_resids_deg = np.degrees(de_4p_res_rad)
#ra_resids_deg = use_dataset['dra'] - m4ra       # ==> prlx * pfra
#ra_resids_deg *= np.cos(nom_de_rad)
#de_resids_deg = use_dataset['dde'] - m4de       # ==> prlx * pfde

ra_resids_mas = 3.6e6 * ra_resids_deg
de_resids_mas = 3.6e6 * de_resids_deg


sys.stderr.write("\n%s\n" % halfdiv)
# old mra_scatter: 296.15471
# old mde_scatter:  57.61126
old_mra_scatter = 296.15471
old_mde_scatter =  57.61126
sys.stderr.write("old_mra_scatter: %10.5f\n" % old_mra_scatter)
sys.stderr.write("old_mde_scatter: %10.5f\n" % old_mde_scatter)
sys.stderr.write("\n")

new_rra_scatter = at2.calc_MAR(raw_res_ra)
new_rde_scatter = at2.calc_MAR(raw_res_de)
new_mra_scatter = at2._MAS_PER_RADIAN * new_rra_scatter
new_mde_scatter = at2._MAS_PER_RADIAN * new_rde_scatter
new_mra_fairdev = new_mra_scatter * use_cos_dec
sys.stderr.write("new_mra_scatter: %10.5f\n" % new_mra_scatter)
sys.stderr.write("new_mra_fairdev: %10.5f\n" % new_mra_fairdev)
sys.stderr.write("new_mde_scatter: %10.5f\n" % new_mde_scatter)
sys.stderr.write("%s\n" % halfdiv)

## Chi-square estimate:
chi2_irls = af._calc_chi_square(bestpars)
pts_used = np.sum(af.inliers)
chi2_irls_dof = chi2_irls / float(2 * pts_used)
#chi2_dof_tru = chi2_dof / np.sqrt(2.)
chi2_message = "points used: %d\n" % pts_used
chi2_message += "total chi2 (IRLS): %10.5f\n" % chi2_irls
chi2_message += "chi2 / dof (IRLS): %10.5f\n" % chi2_irls_dof
#chi2_message += "chi2 / dof (coord): %10.5f\n" % chi2_dof_tru
#sys.stderr.write("total chi2 (hypot): %10.5f\n" % chi2)
#sys.stderr.write("chi2 / dof (hypot): %10.5f\n" % chi2_dof)
#sys.stderr.write("chi2 / dof (coord): %10.5f\n" % chi2_dof_tru)
sys.stderr.write(chi2_message)
sys.stderr.write("%s\n" % halfdiv)

### Re-fit with plx-savvy scatter:
#savvy_ra_errs = np.ones(len(use_dataset)) * new_rra_scatter
#savvy_de_errs = np.ones(len(use_dataset)) * new_rde_scatter
#af.setup(use_dataset, RA_err=savvy_ra_errs, DE_err=savvy_de_errs)
#savvy_pars = af.fit_bestpars(sigcut=sigcut)

## -----------------------------------------------------------------------
## -----------------         Select and Fit Neighbors       --------------
## -----------------------------------------------------------------------


have_tgt_ndata = len(tgt_both)          # total ch1+ch2 target data points
min_nei_ndata  = 0.95 * have_tgt_ndata  # need 95% matched points

nei_want = 10
nei_ndata = {}
use_nei_ids = []
for nn in large_both:
    nch1 = len(sdata['ch1'][nn])
    nch2 = len(sdata['ch2'][nn])
    ntot = nch1 + nch2
    nei_ndata[nn] = ntot
    if (ntot > min_nei_ndata):
        use_nei_ids.append(nn)
    if (len(use_nei_ids) >= nei_want):
        break
    pass

## Residual examinatino:

## Matching residual arrays:
#targ_jdtdb = use_dataset['jdtdb']
targ_jdtdb = use_dataset['jdtdb'][af.inliers]
matched_ra_resvecs = {}

def patched_matched_residuals(want_jds, have_jds, res):
    matched_data = np.zeros_like(want_jds, dtype='float32')
    jd2idx = {jj:vv for jj,vv in zip(want_jds, np.arange(len(want_jds)))}
    #lut = {jj:vv for jj,vv in zip(have_jds, res)}
    for jj,rr in zip(have_jds, res):
        #sys.stderr.write("jj: %s, rr: %s\n" % (str(jj), str(rr)))
        if jj in jd2idx.keys():
            matched_data[jd2idx[jj]] = rr
    return matched_data

## 5-paramteer IRLS fits to neighbors:
sigcut = 5
#af.setup(use_dataset)
nei_both = {}
nei_pars = {}
irls_iters = 30
ra_resids = {}
de_resids = {}
#aligned_ra_resids = {}
#aligned_de_resids = {}
aligned_ra_resids = []
aligned_de_resids = []

## Fit neighbor astrometry:
trend_resid_vecs = {}
trend_meta = {}
for ii,nid in enumerate(use_nei_ids):
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Starting IRLS fit for %s ... \n" % nid)
    _nboth = np.concatenate((sdata['ch1'][nid], sdata['ch2'][nid]))
    #add_recarray_columns
    nei_both[nid] = _nboth

    mdata = {'signal':np.median(_nboth['exptime']*_nboth['flux']),
             'ra_deg':np.median(_nboth['dra']),
             'de_deg':np.median(_nboth['dde'])}
    trend_meta[nid] = mdata

    #afn.setup(use_dataset, RA_err=ra_rad_errs, DE_err=de_rad_errs,
    afn.setup(_nboth, 
            jd_tdb_ref=j2000_epoch.tdb.jd)
    bestpars = afn.fit_bestpars(sigcut=sigcut)
    firstpars = bestpars.copy()
    iterpars = afn.iter_update_bestpars(bestpars)
    for i in range(irls_iters):
        iterpars = afn.iter_update_bestpars(iterpars)
    nei_pars[nid] = iterpars.copy()
    ra_resid, de_resid = afn._calc_radec_residuals(iterpars)
    ra_resids[nid] = ra_resid
    de_resids[nid] = de_resid
    trend_resid_vecs[nid] = pd.DataFrame({  'jdtdb':_nboth['jdtdb'],
                                       'ra_resid':ra_resid,
                                       'de_resid':de_resid    })
    #aligned_ra_resids[nid] = patched_matched_residuals(targ_jdtdb, _nboth['jdtdb'], ra_resid)
    #aligned_de_resids[nid] = patched_matched_residuals(targ_jdtdb, _nboth['jdtdb'], de_resid)
    aligned_ra_resids.append(patched_matched_residuals(targ_jdtdb, _nboth['jdtdb'], ra_resid))
    aligned_de_resids.append(patched_matched_residuals(targ_jdtdb, _nboth['jdtdb'], de_resid))

#def detrend_using_subset(targ_jds, nei_both, ra_resids, de_resids, trend_stars):
#    _aligned_ra_res = []
#    _aligned_de_res = []
#    for nid in trend_stars:
#        _nboth = nei_both[nid]
#        _ra_resid = ra_resids[nid]
#        _de_resid = de_resids[nid]
#        _aligned_ra_res.append(patched_matched_residuals(targ_jds, 
#            nei_both[nid]['jdtdb'], ra_resids[nid]))
#        _aligned_de_res.append(patched_matched_residuals(targ_jds,
#            nei_both[nid]['jdtdb'], de_resids[nid]))


tfa_results_file = 'tfa_results.csv'
tfa_results_cols = ['name', 'ra_deg', 'de_deg', 'signal', 'ra_res', 'ra_tfa', 'de_res', 'de_tfa']

## Check whether a header row is needed:
def file_is_empty(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            content = f.readlines()
        if len(content) >= 1:
            return False
    return True

## Stats file updater:
def dump_to_file(stat_savefile, cols, data, delimiter=','):
    if file_is_empty(stat_savefile):
        sys.stderr.write("No stats file!\n")
        with open(stat_savefile, 'w') as ff:
            ff.write("%s\n" % delimiter.join(cols))
    with open(stat_savefile, 'a') as ff:
        ff.write("%s\n" % delimiter.join([str(x) for x in data]))

# 
for nid in trend_resid_vecs.keys():
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("nid: %s\n" % nid)
    others = [x for x in trend_resid_vecs.keys() if x != nid]
    #sys.stderr.write("others: %s\n" % str(others))
    sys.stderr.write("signal: %15.5f\n" % trend_meta[nid]['signal'])
    _target = trend_resid_vecs[nid]
    _trends = {k:trend_resid_vecs[k] for k in others}
    _aligned_ra_res = []
    _aligned_de_res = []
    for trend,trdata in _trends.items():
        _aligned_ra_res.append(patched_matched_residuals(_target['jdtdb'],
                                         trdata['jdtdb'], trdata['ra_resid']))
        _aligned_de_res.append(patched_matched_residuals(_target['jdtdb'],
                                         trdata['jdtdb'], trdata['de_resid']))

    save_things = [nid, trend_meta[nid]['ra_deg'], trend_meta[nid]['de_deg'],
                        trend_meta[nid]['signal'], ]

    # RA detrend:
    _ra_dmat = np.array(_aligned_ra_res).T              # design matrix
    _ra_nmat = np.dot(_ra_dmat.T, _ra_dmat)             # normal matrix
    _ra_xtxi = np.linalg.inv(_ra_nmat)                  # X_transpose_X-inverse
    _ra_prod = np.dot(_ra_dmat.T, _target['ra_resid'])  # X_transpose * Y
    _ra_coef = np.dot(_ra_xtxi, _ra_prod)
    _ra_filt = np.dot(_ra_dmat, _ra_coef)
    _target['ra_detrend'] = _target['ra_resid'] - _ra_filt
    #_, old_scatter = rs.calc_ls_med_IQR(_target['ra_resid']) * at2._MAS_PER_RADIAN
    #_, new_scatter = rs.calc_ls_med_IQR(_target['ra_detrend']) * at2._MAS_PER_RADIAN
    old_scatter = np.std(_target['ra_resid'])   * at2._MAS_PER_RADIAN
    new_scatter = np.std(_target['ra_detrend']) * at2._MAS_PER_RADIAN
    pct_improvement = 100. * (old_scatter - new_scatter) / old_scatter
    sys.stderr.write("old RA scatter: %s\n" % old_scatter)
    sys.stderr.write("new RA scatter: %s\n" % new_scatter)
    sys.stderr.write("Improved by %6.1f%%\n" % pct_improvement)
    save_things += [old_scatter, new_scatter]

    # DE detrend:
    _de_dmat = np.array(_aligned_de_res).T              # design matrix
    _de_nmat = np.dot(_de_dmat.T, _de_dmat)             # normal matrix
    _de_xtxi = np.linalg.inv(_de_nmat)                  # X_transpose_X-inverse
    _de_prod = np.dot(_de_dmat.T, _target['de_resid'])  # X_transpose * Y
    _de_coef = np.dot(_de_xtxi, _de_prod)
    _de_filt = np.dot(_de_dmat, _de_coef)
    _target['de_detrend'] = _target['de_resid'] - _de_filt
    #_, old_scatter = rs.calc_ls_med_IQR(_target['de_resid'])
    #_, new_scatter = rs.calc_ls_med_IQR(_target['de_detrend'])
    #old_scatter, new_scatter = np.std(_target['de_resid']), np.std(_target['de_detrend'])
    old_scatter = np.std(_target['de_resid'])   * at2._MAS_PER_RADIAN
    new_scatter = np.std(_target['de_detrend']) * at2._MAS_PER_RADIAN
    pct_improvement = 100. * (old_scatter - new_scatter) / old_scatter
    sys.stderr.write("old DE scatter: %s\n" % old_scatter)
    sys.stderr.write("new DE scatter: %s\n" % new_scatter)
    sys.stderr.write("Improved by %6.1f%%\n" % pct_improvement)
    save_things += [old_scatter, new_scatter]

    # save it!
    dump_to_file(tfa_results_file, tfa_results_cols, save_things)


## Plot target and neighbor astrometry:
sys.stderr.write("\n\nPlotting target residuals ... \n")
nfig = plt.figure(66)
nfig.clf()
raax = nfig.add_subplot(211); raax.grid(True); raax.set_ylabel("RA residuals")
deax = nfig.add_subplot(212); deax.grid(True); deax.set_ylabel("DE residuals")
pkw = {'lw':0, 's':25}
raax.scatter(targ_jdtdb, raw_res_ra[af.inliers], **pkw)
deax.scatter(targ_jdtdb, raw_res_de[af.inliers], **pkw)
fixed_ra_yshift = 10. * np.std(raw_res_ra)
fixed_de_yshift = 10. * np.std(raw_res_de)
sys.stderr.write("Adding neighbor residuals ... \n")
for ii,nid in enumerate(use_nei_ids):
    _nboth = nei_both[nid]
    iterpars = nei_pars[nid]
    ra_resid, de_resid = afn._calc_radec_residuals(iterpars)
    ra_shift = (ii + 1) * fixed_ra_yshift
    de_shift = (ii + 1) * fixed_de_yshift
    #ra_resids.append(ra_resid)
    #de_resids.append(de_resid)
    #aligned_ra_resids.append(patched_matched_residuals(targ_jdtdb, _nboth['jdtdb'], ra_resid))
    #aligned_de_resids.append(patched_matched_residuals(targ_jdtdb, _nboth['jdtdb'], de_resid))
    raax.scatter(_nboth['jdtdb'], ra_resids[nid]+ra_shift, **pkw)
    deax.scatter(_nboth['jdtdb'], de_resids[nid]+de_shift, **pkw)
sys.stderr.write("Plotting complete.\n")
nfig.tight_layout()

#raax.scatter(use_dataset['jdtdb'], raw_res_ra + 0
nfig.savefig('nei_residuals.png')

## Median neighbor residual:
#taligned_ra_resid = [aligned_ra_resids[nn] for nn in use_nei_ids]
#taligned_de_resid = [aligned_de_resids[nn] for nn in use_nei_ids]
#medalign_ra_resid = np.median(np.array(aligned_ra_resids), axis=0)
#medalign_de_resid = np.median(np.array(aligned_de_resids), axis=0)

# Does it match:
#res_ratio_ra = raw_res_ra / medalign_ra_resid
#res_ratio_de = raw_res_de / medalign_de_resid

use_res_ra = raw_res_ra[af.inliers]
use_res_de = raw_res_de[af.inliers]

ra_dmat = np.array(aligned_ra_resids).T         # design matrix
ra_nmat = np.dot(ra_dmat.T, ra_dmat)            # normal matrix
ra_xtxi = np.linalg.inv(ra_nmat)                # X_transpose_X-inverse
ra_prod = np.dot(ra_dmat.T, use_res_ra)         # X_transpose * Y
ra_coef = np.dot(ra_xtxi, ra_prod)
de_dmat = np.array(aligned_de_resids).T         # design matrix
de_nmat = np.dot(de_dmat.T, de_dmat)            # normal matrix
de_xtxi = np.linalg.inv(de_nmat)                # X_transpose_X-inverse
de_prod = np.dot(de_dmat.T, use_res_de)         # X_transpose * Y
de_coef = np.dot(de_xtxi, de_prod)

## Filter:
ra_filt = np.dot(ra_dmat, ra_coef)
de_filt = np.dot(de_dmat, de_coef)

## Single shot:
#rckw = {'rcond':None} if (_have_np_vers >= 1.14) else {}
#ra_soln = np.linalg.lstsq(ra_dmat, use_res_ra, **rckw)
#de_soln = np.linalg.lstsq(de_dmat, use_res_de, **rckw)


#res_ratio_de = raw_res_de / medalign_de_resid

cln_res_ra = use_res_ra - ra_filt
cln_res_de = use_res_de - de_filt

## Target signal for comparison:
targ_signal = np.median(use_dataset['exptime']*use_dataset['flux'])
targ_ra_deg = np.median(use_dataset['dra'])
targ_de_deg = np.median(use_dataset['dde'])
sys.stderr.write("Target signal level: %15.5f\n" % targ_signal)

## prepare data for dump:
save_things = [plot_tag, targ_ra_deg, targ_de_deg, targ_signal]

old_res = np.std(use_res_ra) * at2._MAS_PER_RADIAN
new_res = np.std(cln_res_ra) * at2._MAS_PER_RADIAN
pct_win = 100.0 * (old_res - new_res) / old_res
sys.stderr.write("\n")
sys.stderr.write("Old RA res: %10.4f\n" % old_res)
sys.stderr.write("New RA res: %10.4f\n" % new_res)
sys.stderr.write("Improvement: %6.1f%%\n" % pct_win)
save_things += [old_res, new_res]

old_res = np.std(use_res_de) * at2._MAS_PER_RADIAN
new_res = np.std(cln_res_de) * at2._MAS_PER_RADIAN
pct_win = 100.0 * (old_res - new_res) / old_res
sys.stderr.write("\n")
sys.stderr.write("Old DE res: %10.4f\n" % old_res)
sys.stderr.write("Old DE res: %10.4f\n" % new_res)
sys.stderr.write("Improvement: %6.1f%%\n" % pct_win)
save_things += [old_res, new_res]


dump_to_file(tfa_results_file, tfa_results_cols, save_things)

sys.exit(0)

plt.plot(use_res_ra)
plt.plot(cln_res_ra)

plt.plot(use_res_de)
plt.plot(cln_res_de)

#bestpars = af.fit_bestpars(sigcut=sigcut)
#firstpars = bestpars.copy()

### Iterate a bunch to better solution:
#iterpars = af.iter_update_bestpars(bestpars)
#for i in range(30):
#    iterpars = af.iter_update_bestpars(iterpars)
#bestpars = iterpars

## -----------------------------------------------------------------------
## -----------------         MCMC Attempt (emcee)           --------------
## -----------------------------------------------------------------------

_PERFORM_MCMC = False
_PERFORM_MCMC = True

def lnprior(params):
    #rra, rde, pmra, pmde, prlx = params
    #if prlx < 0:
    #    return -np.inf
    return 0

def lnlike(params, rra, rde, rra_err, rde_err):
    mrra, mrde = af.eval_model(params)
    delta_ra = (mrra[af.inliers] - rra) / rra_err
    delta_de = (mrde[af.inliers] - rde) / rde_err
    delta_tot = delta_ra**2 + delta_de**2
    return -0.5*np.sum(delta_tot)

def lnprob(params, rra, rde, rra_err, rde_err):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, rra, rde, rra_err, rde_err)

## Identify useful data points:
use_rra = np.radians(use_dataset['dra'][af.inliers])
use_rde = np.radians(use_dataset['dde'][af.inliers])
use_rra_err_raw = ra_rad_errs[af.inliers]
use_rde_err_raw = de_rad_errs[af.inliers]
use_rra_err_adj = af._use_RA_err[af.inliers]
use_rde_err_adj = af._use_DE_err[af.inliers]
use_rra_err = use_rra_err_adj
use_rde_err = use_rde_err_adj

## Save a copy of best-fit parameters for posterity:
par_save_file = 'results_params.txt'
with open(par_save_file, 'a') as pf:
    pf.write(plot_tag)
    for vv in bestpars:
        pf.write(" %15.8f" % vv)
    pf.write("\n")

## Save residuals and errors for external examination:
save_residuals = True
resids_dir = 'residuals'
if save_residuals:
    scaled_ra_resid, scaled_de_resid = af._calc_radec_residuals_sigma(bestpars)
    scaled_ra_resid = scaled_ra_resid[af.inliers]
    scaled_de_resid = scaled_de_resid[af.inliers]
    if not os.path.isdir(resids_dir):
        os.mkdir(resids_dir)
    resids_file = '%s/resid_%s.csv' % (resids_dir, plot_tag)
    with open(resids_file, 'w') as rf:
        rf.write("scaled_ra_resid,scaled_de_resid\n")
        for sradec in zip(scaled_ra_resid, scaled_de_resid):
            rf.write("%.3f,%.3f\n" % sradec)

sys.exit(0)

arglist = (use_rra, use_rde, use_rra_err, use_rde_err)
if _PERFORM_MCMC:
    tik = time.time()
    initial = iterpars.copy()
    ndim = len(initial)
    nwalkers = 32
    p0 = [np.array(initial) + 1e-6*initial*np.random.randn(5) \
            for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=arglist)

    sys.stderr.write("Running burn-in ... ")
    p0, _, _ = sampler.run_mcmc(p0, 200)
    sys.stderr.write("done.\n")
    sampler.reset()

    sys.stderr.write("Running full MCMC ... ")
    niter, thinned = 4000, 15
    niter, thinned = 2000, 5
    pos, prob, state = sampler.run_mcmc(p0, niter)
    sys.stderr.write("done.\n")

    ra_chain, de_chain, pmra_chain, pmde_chain, prlx_chain = sampler.flatchain.T

    plabels = ['ra', 'de', 'pmra', 'pmde', 'prlx']
    flat_samples = sampler.get_chain(discard=100, thin=thinned, flat=True)
    #labels=plabels)
    tok = time.time()
    sys.stderr.write("Running MCMC took %.2f seconds.\n" % (tok-tik))

    # best parameters from mcmc:
    mcmcpars = np.array([np.median(x) for x in sampler.flatchain.T])

    # chi^2 of best parameters:
    chi2_mcmc = af._calc_chi_square(mcmcpars)
    chi2_mcmc_dof = chi2_mcmc / float(2 * np.sum(af.inliers))
    chi2_message += "total chi2 (MCMC): %10.5f\n" % chi2_mcmc
    chi2_message += "chi2 / dof (MCMC): %10.5f\n" % chi2_mcmc_dof

    # plot parallax posterior distribution in mas:
    cfig = plt.figure(22)
    cfig.clf()
    plxax = cfig.add_subplot(111)
    plxax.grid(True)
    prlx_chain_mas = 3.6e6 * np.degrees(prlx_chain)
    plxax.hist(prlx_chain_mas, bins=50)
    plxax.set_xlabel("Parallax (mas)")
    cfig.tight_layout()
    plt.draw()
    save_plot = 'plots/plx_posterior_%s.png' % plot_tag
    cfig.savefig(save_plot) #, bbox='tight')

    # create corner plot of MCMC results:
    corn_file = 'corner_plot_%s.pdf' % plot_tag
    cornerfig = plt.figure(31, figsize=(9,7))
    cornerfig.clf()
    corner.corner(flat_samples, labels=plabels, fig=cornerfig,
            truths=iterpars)
    #cornerfig.tight_layout()
    cornerfig.suptitle("%s\n%s" % (plot_tag, chi2_message))
    plt.draw()
    cornerfig.savefig(corn_file)


## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

## isolate troublesome dates (ch2):
trouble_jds = use_dataset['jdtdb'][(ra_resids_mas > 800)] 
trouble_imgs = [jd2im_both[x] for x in trouble_jds]

## timestamp for plots:
plot_time = use_dataset['jdtdb'] - ref_time

### Get/save list of AORs and AOR-specific stats:
#aor_savefile = 'aor_data_%s_%s.csv' % (tgt_name, dwhich)
##targ_aors = np.int_([x.split('_')[2] for x in tdata['ch2']['iname']])
#targ_aors = np.int_([x.split('_')[2] for x in tdata[dwhich]['iname']])
#with open(aor_savefile, 'w') as af:
#    af.write("aor,avgjd,ra_resid,ra_resid_std,de_resid,de_resid_std\n")
#    for this_aor in np.unique(targ_aors):
#        sys.stderr.write("\n--------------------------\n")
#        sys.stderr.write("this_aor: %d\n" % this_aor)
#        which = (targ_aors == this_aor)
#        avg_tt = np.average(plot_time[which])
#        avg_jd = np.average(use_dataset['jdtdb'][which])
#        avg_ra_miss = np.average(ra_resids_mas[which])
#        avg_de_miss = np.average(de_resids_mas[which])
#        std_ra_miss = np.std(ra_resids_mas[which])
#        std_de_miss = np.std(de_resids_mas[which])
#        sys.stderr.write("Time point: %f\n" % avg_tt) 
#        sys.stderr.write("AOR JDTDB: %f\n" % avg_jd) 
#        sys.stderr.write("RA errors: %.2f (%.2f)\n" % (avg_ra_miss, std_ra_miss))
#        sys.stderr.write("DE errors: %.2f (%.2f)\n" % (avg_de_miss, std_de_miss))
#        af.write("%d,%f,%f,%f,%f,%f\n" % (this_aor, avg_jd, 
#            avg_ra_miss, std_ra_miss, avg_de_miss, std_de_miss))
#        pass
#
### Get/save list of image-specific residuals and data:
#res_savefile = 'res_data_%s_%s.csv' % (tgt_name, dwhich)
#for stuff in zip(use_dataset['iname'], ra_resids_mas, de_resids_mas):
#    pass

#sys.exit(0)

solved_plx_mas = bestpars[4]*at2._MAS_PER_RADIAN
prev_plx_mas = 1e3 * prev_pars['parallax_as']
#eph_jdtdb, eph_plxra, eph_plxde = calc_full_plxf(63.832226, -9.585037, 167.)
eph_jdtdb, eph_plxra, eph_plxde = calc_full_plxf(nom_ra_deg, nom_de_deg, 1.)
eph_ptime = eph_jdtdb - ref_time

## A text blurb about object position:
coords_lab = 'RA, DE = %10.5f, %10.5f' \
        % (prev_pars['ra_deg'], prev_pars['de_deg'])

# trim to span available data:
padding = 0.1
data_duration = plot_time.max() - plot_time.min()
trim_lower = plot_time.min() - padding * data_duration
trim_upper = plot_time.max() + padding * data_duration
trim = (trim_lower <= eph_ptime) & (eph_ptime <= trim_upper)


fig_dims = (14, 10)
fig = plt.figure(12, figsize=fig_dims)
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=fig_dims, num=12)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
every_ax = [ax1, ax2, ax3, ax4]
for aa in every_ax:
    aa.grid(True)

mar_4par_ra = at2.calc_MAR(ra_resids_mas)
mar_5par_ra = at2.calc_MAR(raw_res_ra_mas)
lab_4par_ra = '4par res: %.2f mas' % mar_4par_ra
lab_5par_ra = '5par res: %.2f mas' % mar_5par_ra

ax1.scatter(plot_time, ra_resids_mas, c='g', lw=0, label=lab_4par_ra)
ax1.plot(eph_ptime[trim], prev_plx_mas*eph_plxra[trim], c='r',
        label='previous sln (%.2f mas)'%prev_plx_mas)
ax1.plot(eph_ptime[trim], solved_plx_mas*eph_plxra[trim], c='b',
        label='solved plx (%.2f mas)'%solved_plx_mas)
ax1.set_ylabel('RA Residual * cos(Dec) (mas)')
ax1.set_title('Target: %s' % tgt_name)
ax2.scatter(plot_time, raw_res_ra_mas, c='b', lw=0, label=lab_5par_ra)
ax2.set_title(coords_lab)

mar_4par_de = at2.calc_MAR(de_resids_mas)
mar_5par_de = at2.calc_MAR(raw_res_de_mas)
lab_4par_de = '4par res: %.2f mas' % mar_4par_de
lab_5par_de = '5par res: %.2f mas' % mar_5par_de
ax3.scatter(plot_time, de_resids_mas, c='g', lw=0, label=lab_4par_de)
ax3.plot(eph_ptime[trim], prev_plx_mas*eph_plxde[trim], c='r',
        label='previous sln (%.2f mas)'%prev_plx_mas)
ax3.plot(eph_ptime[trim], solved_plx_mas*eph_plxde[trim], c='b',
        label='solved plx (%.2f mas)'%solved_plx_mas)
ax3.set_ylabel('DE Residual (mas)')
ax3.set_xlabel('Time (~days)')
ax4.set_xlabel('Time (~days)')
ax3.set_ylim(-500, 500)
#ax3.legend(loc='upper right')
ax4.scatter(plot_time, raw_res_de_mas, c='b', lw=0, label=lab_5par_de)

for aa in every_ax:
    aa.grid(True)
    aa.legend(loc='upper left')

fig.tight_layout()
pname = 'plots/%s_tgt_4par_resid_%s_%s.png' % (tgt_name, dwhich, cat_type)
#pname = '%s_tgt_4par_resid_%s_%s.pdf' % (tgt_name, dwhich, cat_type)
fig.savefig(pname)


## Draw observation locations in SSB:
def ssb_lims(ax, lower=-1.10, upper=1.10):
    ax.grid(True)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

fig = plt.figure(14, figsize=fig_dims)
fig.clf()
axy = fig.add_subplot(221, aspect='equal')
ssb_lims(axy)
axy.scatter(use_dataset['obs_x'], use_dataset['obs_y'], lw=0)
axy.set_xlabel('X (au)')
axy.set_ylabel('Y (au)')

axz = fig.add_subplot(222, aspect='equal')
ssb_lims(axz)
axz.scatter(use_dataset['obs_x'], use_dataset['obs_z'], lw=0)
axz.set_xlabel('X (au)')
axz.set_ylabel('Z (au)')

ayz = fig.add_subplot(223, aspect='equal')
ayz.scatter(use_dataset['obs_y'], use_dataset['obs_z'], lw=0)
ssb_lims(ayz)
ayz.set_xlabel('Y (au)')
ayz.set_ylabel('Z (au)')
fig.tight_layout()


### For reference, let's see histograms of RA/DE parallax factors:
#hopts = {'range':(-1.1, 1.1), 'bins':22}
#fig = plt.figure(15, figsize=fig_dims)
#fig.clf()
#ax1 = fig.add_subplot(211)
#ax1.hist(pfra, **hopts)
#ax2 = fig.add_subplot(212)
#ax2.hist(pfde, **hopts)



##
### 277.1377951+26.8662544
###
### neighbors / other stars ...
##nei_data = scatter_time_ch2
#nei_data = scatter_time_ch1
#ntime, nra_res, nde_res = nei_data.T
##ntime, nra_res, nde_res = src_resids_ch2['277.1377951+26.8662544']
##ntime, nra_res, nde_res = src_resids_ch2['277.1407216+26.8323065']
##ntime, nra_res, nde_res = src_resids_ch2['277.1416760+26.8399335']
##ntime, nra_res, nde_res = src_resids_ch2['277.1474782+26.8489036']
##ntime, nra_res, nde_res = src_resids_ch2['277.1476409+26.8547661']
##ntime, nra_res, nde_res = src_resids_ch2['277.1527917+26.8247245']
##ntime, nra_res, nde_res = src_resids_ch2['277.1613947+26.8283669']
#ntime, nra_res, nde_res = src_resids_ch2['277.1377510+26.8663015']
#nfg = plt.figure(13, figsize=fig_dims)
#nfg.clf()
#ax1 = nfg.add_subplot(211)
#ax2 = nfg.add_subplot(212, sharex=ax1, sharey=ax1)
#ax1.grid(True)
##ax1.scatter(ntime, nra_res, lw=0, s=2)
#ax1.scatter(ntime, nra_res, lw=0)
#ax1.set_ylabel('RA Residual * cos(Dec) (mas)')
#ax1.set_title('Other stars with data ...')
#ax2.grid(True)
##ax2.scatter(ntime, nde_res, lw=0, s=2)
#ax2.scatter(ntime, nde_res, lw=0)
#ax2.set_ylabel('DE Residual (mas)')
#ax2.set_xlabel('Time (~days)')
#
#
#nfg.tight_layout()
#pname = '%s_nei_4par_resid_ch2_%s.png' % (tgt_name, plot_tag)
#nfg.savefig(pname)
##sys.exit(0)

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


## EARLY diagnostic plots, uncomment to reproduce
#fig_dims = (10, 10)
#fig = plt.figure(2, figsize=fig_dims)
#fig.clf()
#ax1 = fig.add_subplot(211); ax1.grid(True)
#ax2 = fig.add_subplot(212); ax2.grid(True)
#pmra_diff = prmot['pmra_ch1']-prmot['pmra_ch2']
#pmde_diff = prmot['pmde_ch1']-prmot['pmde_ch2']
#ax1.hist(pmra_diff, range=(-200, 200), bins=11)
#ax1.set_title('pmRA (CH1) - pmRA (CH2)')
#ax1.set_xlabel('delta pmRA (mas)')
##ax2.set_xlim(-2, 82)
#ax2.hist(pmde_diff, range=(-200, 200), bins=11)
#ax2.set_xlabel('delta pmDE (mas)')
#fig.tight_layout()
#fig.savefig('prmot_1v2.png')
#
#fig.clf()
#ax1 = fig.add_subplot(211); ax1.grid(True)
#ax2 = fig.add_subplot(212); ax2.grid(True)
#ax1.scatter(np.abs(prmot['pmra_jnt']), np.abs(pmra_diff))
#ax1.set_xlabel('|pmra_jnt|')
#ax1.set_ylabel('pmra_diff')
#ax2.scatter(np.abs(prmot['pmde_jnt']), np.abs(pmde_diff))
#ax2.set_xlabel('|pmde_jnt|')
#ax2.set_ylabel('pmde_diff')
#ax1.set_ylim(-20, 200)
#ax2.set_ylim(-20, 200)
#ax1.set_xlim(-2, 82)
#ax2.set_xlim(-2, 82)
#fig.tight_layout()
#fig.savefig('prmot_diff_vs_prmot.png')

#sys.exit(0)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


fig_dims = (10, 11)

def limify(ax):
   # ax.set_xlim(0.2, 50.)
    #ax.set_ylim(500., 2500.)
    ax.set_xlim(0.9, 250.)
    ax.set_ylim(0.01, 1.0)
    ax.set_yscale('log')

## CH1 FWHM:
every_delta_mas, every_flux, every_signal = np.array(inspection_ch1).T
every_snr = np.sqrt(every_signal)
every_invsnr = 1.0 / every_snr
design_matrix = np.column_stack((np.ones(every_invsnr.size), every_invsnr))
rlm_res = sm.RLM(every_delta_mas, design_matrix).fit()
icept, slope = rlm_res.params
snr_order = np.argsort(every_invsnr)
srt_invsnr = every_invsnr[snr_order]
fitted_resid = icept + slope * srt_invsnr
fitlab = '%.2f, %.2f' % (icept, slope)

flx, mfwhm, npoints, msnr, mresd, avgra, avgde, avgjd, stdjd, iqrflx = \
                                np.array(lookie_snr_ch1).T
rel_scatter = iqrflx / flx
medmed_fwhm = np.median(mfwhm)
mmf_txt = 'median: %.1f mas' % medmed_fwhm
fig = plt.figure(3, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(211); ax1.grid(True)
ax1.set_title("IRAC channel 1 (%s)" % plot_tag)
#spts = ax1.scatter(flx, mfwhm, c=npoints)
#spts = ax1.scatter(flx, rel_scatter, c=npoints)
spts = ax1.scatter(msnr, rel_scatter, c=npoints)
ax1.axhline(medmed_fwhm, c='r', ls='--', label=mmf_txt)
limify(ax1)
ax1.set_xscale('log')
ax1.set_xlabel('med_flux')
#ax1.set_ylabel('FWHM (mas)')
ax1.set_ylabel('Relative Flux Scatter')
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
pct5 = np.percentile(mresd, 5)
pct_lab = '5th pctile: %.1f mas' % pct5
ax2.axhline(pct5, lw=0.5, ls='--', c='r', label=pct_lab)
#spts = ax2.scatter(every_invsnr, every_delta_mas, lw=0, s=1)
#ax2.plot(srt_invsnr, fitted_resid, c='r', label=fitlab)
ax2.set_ylabel('resid (mas)')
ax2.set_xlabel('med_snr')
ax2.set_xscale('log')
ax2.set_yscale('log')
#ax2.set_xlim(0.9, 110.)
ax2.set_xlim(0.9, 250.)
#ax2.set_ylim(30., 2500.)
ax2.set_ylim(30., 2500.)
ax2.legend(loc='upper right')
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
#ax3 = fig.add_subplot(313, sharex=ax1); ax3.grid(True)
#spts = ax3.scatter(msnr, mresd*rel_scatter, c=npoints)
##limify(ax3)
#ax3.set_xscale('log')
##ax3.set_yscale('log')
#scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#scm.set_array([])
#cbar = fig.colorbar(scm, orientation='vertical')
fig.tight_layout()
plt.draw()
plot_name = 'plots/empirical_scatter_ch1_%s.png' % plot_tag
fig.savefig(plot_name)


## CH2 FWHM:
every_delta_mas, every_flux, every_signal = np.array(inspection_ch2).T
every_snr = np.sqrt(every_signal)
every_invsnr = 1.0 / every_snr
design_matrix = np.column_stack((np.ones(every_invsnr.size), every_invsnr))
rlm_res = sm.RLM(every_delta_mas, design_matrix).fit()
icept, slope = rlm_res.params
snr_order = np.argsort(every_invsnr)
srt_invsnr = every_invsnr[snr_order]
fitted_resid = icept + slope * srt_invsnr
fitlab = '%.2f, %.2f' % (icept, slope)

flx, mfwhm, npoints, msnr, mresd, avgra, avgde, avgjd, stdjd, iqrflx = \
                                np.array(lookie_snr_ch2).T
rel_scatter = iqrflx / flx
medmed_fwhm = np.median(mfwhm)
mmf_txt = 'median: %.1f mas' % medmed_fwhm
fig = plt.figure(4, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(211); ax1.grid(True)
ax1.set_title("IRAC channel 2 (%s)" % plot_tag)
#spts = ax1.scatter(flx, mfwhm, c=npoints)
#spts = ax1.scatter(flx, rel_scatter, c=npoints)
spts = ax1.scatter(msnr, rel_scatter, c=npoints)
ax1.axhline(medmed_fwhm, c='r', ls='--', label=mmf_txt)
limify(ax1)
#ax1.set_xlim(0.2, 400.)
ax1.set_xscale('log')
ax1.set_xlabel('med_flux')
#ax1.set_ylabel('FWHM (mas)')
ax1.set_ylabel('Relative Flux Scatter')
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
pct5 = np.percentile(mresd, 5)
pct_lab = '5th pctile: %.1f mas' % pct5
ax2.axhline(pct5, lw=0.5, ls='--', c='r', label=pct_lab)
#spts = ax2.scatter(msnr, mresd, c=avgjd)
#spts = ax2.scatter(every_invsnr, every_delta_mas, lw=0, s=1)
#ax2.plot(srt_invsnr, fitted_resid, c='r', label=fitlab)
ax2.set_ylabel('resid (mas)')
ax2.set_xlabel('med_snr')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(0.9, 250.)
ax2.set_ylim(30., 2500.)
ax2.legend(loc='upper right')
#ax2.set_ylim(30., 2500.)
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
fig.tight_layout()
plt.draw()
plot_name = 'plots/empirical_scatter_ch2_%s.png' % plot_tag
fig.savefig(plot_name)

fig = plt.figure(6, figsize=(10,8))
fig.clf()
ax1 = fig.add_subplot(111, aspect='equal'); ax1.grid(True)
ax1.set_title("IRAC channel 2 (%s)" % plot_tag)
spts = ax1.scatter(avgra, avgde, c=npoints)
ax1.set_xlabel('typical RA')
ax1.set_ylabel('typical DE')
cbnorm = mplcolors.Normalize(*spts.get_clim())
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
cbar.formatter.set_useOffset(False)
cbar.update_ticks()
cbar.set_label('Data Points')
fig.tight_layout()
plt.draw()
plot_name = 'plots/npoints_vs_radec_ch2_%s.png' % plot_tag
fig.savefig(plot_name)

fig.clf()
ax1 = fig.add_subplot(111, aspect='equal'); ax1.grid(True)
ax1.set_title("IRAC channel 2 (%s)" % plot_tag)
#pcolors = mresd
pcolors = np.log10(mresd)
#pcolors = mresd * msnr
#pcolors = np.log10(mresd * msnr)
spts = ax1.scatter(avgra, avgde, c=pcolors)
ax1.set_xlabel('typical RA')
ax1.set_ylabel('typical DE')
cbnorm = mplcolors.Normalize(*spts.get_clim())
scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
scm.set_array([])
cbar = fig.colorbar(scm, orientation='vertical')
cbar.formatter.set_useOffset(False)
cbar.update_ticks()
cbar.set_label('log10(residual / mas)')
#cbar.set_label('log10(residual * SNR)')
fig.tight_layout()
plt.draw()
plot_name = 'plots/resid_vs_radec_ch2_%s.png' % plot_tag
fig.savefig(plot_name)

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

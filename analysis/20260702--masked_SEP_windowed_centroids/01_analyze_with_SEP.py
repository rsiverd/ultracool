#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Analyze the sample image using a few different masked+windowed
# routine candidates.
#
# Rob Siverd
# Created:       2026-07-02
# Last modified: 2026-07-02
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

### Python version-agnostic module reloading (cute, 2.7+?):
#import sys
#reload = sys.modules['imp' if 'imp' in sys.modules else 'importlib'].reload

## Modules:
#import argparse
#import shutil
#import resource
#import signal
#import glob
#import math
#import ast
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
from numpy.lib.recfunctions import append_fields
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

## Save object as string:
def stash_as_stringrep(filename, thing):
    with open(filename, 'w') as sasf:
        sasf.write(str(thing) + '\n')
        #pprint.pprint(thing, stream=sasf, width=120)  # pretty!
    return

## Stringified object load (ast):
def load_stringrep_object(filename):
    with open(filename, 'r') as srof: 
        return ast.literal_eval(srof.read())

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
    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

## Star extraction:
try:
    import sep 
except ImportError:
    #sys.stderr.write("Error: sep module not installed!\n\n")
    logger.error("sep module not installed!\n\n")
    sys.exit(1)

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
## Save FITS image with clobber (astropy / pyfits):
def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
    this_func = sys._getframe().f_code.co_name
    parent_func = sys._getframe(1).f_code.co_name
    sys.stderr.write("Writing to '%s' ... " % iname)
    if header:
        while (len(header) < padkeys):
            header.append() # pad header
    if os.path.isfile(iname):
        os.remove(iname)
    pf.writeto(iname, idata, header=header, **kwargs)
    sys.stderr.write("done.\n")


##--------------------------------------------------------------------------##
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Parse arguments and run script:
#class MyParser(argparse.ArgumentParser):
#    def error(self, message):
#        sys.stderr.write('error: %s\n' % message)
#        self.print_help()
#        sys.exit(2)
#
### Enable raw text AND display of defaults:
#class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
#                        argparse.RawDescriptionHelpFormatter):
#    pass
#
### Parse the command line:
#if __name__ == '__main__':
#
#    # ------------------------------------------------------------------
#    prog_name = os.path.basename(__file__)
#    descr_txt = """
#    PUT DESCRIPTION HERE.
#    
#    Version: %s
#    """ % __version__
#    parser = argparse.ArgumentParser(
#            prog='PROGRAM_NAME_HERE',
#            prog=os.path.basename(__file__),
#            #formatter_class=argparse.RawTextHelpFormatter)
#            description='PUT DESCRIPTION HERE.')
#            #description=descr_txt)
#    parser = MyParser(prog=prog_name, description=descr_txt)
#                          #formatter_class=argparse.RawTextHelpFormatter)
#    # ------------------------------------------------------------------
#    parser.set_defaults(thing1='value1', thing2='value2')
#    # ------------------------------------------------------------------
#    parser.add_argument('firstpos', help='first positional argument')
#    parser.add_argument('-w', '--whatever', required=False, default=5.0,
#            help='some option with default [def: %(default)s]', type=float)
#    parser.add_argument('-s', '--site',
#            help='Site to retrieve data for', required=True)
#    parser.add_argument('-n', '--number_of_days', default=1,
#            help='Number of days of data to retrieve.')
#    parser.add_argument('-o', '--output_file', 
#            default='observations.csv', help='Output filename.')
#    parser.add_argument('--start', type=str, default=None, 
#            help="Start time for date range query.")
#    parser.add_argument('--end', type=str, default=None,
#            help="End time for date range query.")
#    parser.add_argument('-d', '--dayshift', required=False, default=0,
#            help='Switch between days (1=tom, 0=today, -1=yest', type=int)
#    parser.add_argument('-e', '--encl', nargs=1, required=False,
#            help='Encl to make URL for', choices=all_encls, default=all_encls)
#    parser.add_argument('-s', '--site', nargs=1, required=False,
#            help='Site to make URL for', choices=all_sites, default=all_sites)
#    parser.add_argument('remainder', help='other stuff', nargs='*')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    #iogroup = parser.add_argument_group('File I/O')
#    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
#    #        help='Output filename', type=str)
#    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
#    #        help='KELT image with WCS')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    ofgroup = parser.add_argument_group('Output format')
#    fmtparse = ofgroup.add_mutually_exclusive_group()
#    fmtparse.add_argument('--python', required=False, dest='output_mode',
#            help='Return Python dictionary with results [default]',
#            default='pydict', action='store_const', const='pydict')
#    bash_var = 'ARRAY_NAME'
#    bash_msg = 'output Bash code snippet (use with eval) to declare '
#    bash_msg += 'an associative array %s containing results' % bash_var
#    fmtparse.add_argument('--bash', required=False, default=None,
#            help=bash_msg, dest='bash_array', metavar=bash_var)
#    fmtparse.set_defaults(output_mode='pydict')
#    # ------------------------------------------------------------------
#    # Miscellany:
#    miscgroup = parser.add_argument_group('Miscellany')
#    miscgroup.add_argument('--debug', dest='debug', default=False,
#            help='Enable extra debugging messages', action='store_true')
#    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
#            help='less progress/status reporting')
#    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
#            help='more progress/status reporting')
#    # ------------------------------------------------------------------
#
#    context = parser.parse_args()
#    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
#    context.prog_name = prog_name
#
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Stats helper:
def _calc_imstats(idata):
    iclean = idata[~np.isnan(idata)]
    pix_med, pix_iqrdev = rs.calc_ls_med_IQR(iclean)
    pix_avg, pix_stddev = np.average(iclean), np.std(iclean)
    return {'med':pix_med, 'iqrdev':pix_iqrdev,
            'avg':pix_avg, 'stddev':pix_stddev,}

def _calc_sig_threshold(sigmas):
    return sigmas * self.im_stats['iqrdev']


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Load FITS image:
image_file = './wircam_J_1325839p.fits.fz'
img_u16, ihdrs = pf.getdata(image_file, header=True)

## Make a floating-point version for SEP to work with:
img_f32 = img_u16.astype('float32')     # promote

## SEP options:
sep_minpixels = 5
sep_pixorigin = 1
sep_win_sigma = 2.0
sep_calc_wpos = True
sep_sigthresh = 2.0

## Collect stats:
istats = _calc_imstats(img_f32)

## Extraction threshold in ADU comes from image stats and sigthresh:
sep_aduthresh = sep_sigthresh * istats['iqrdev']

##--------------------------------------------------------------------------##
## Estimate background:
sys.stderr.write("Estimate background ... ")
bkg_128 = sep.Background(img_f32, bw=128, bh=128)
bkg_064 = sep.Background(img_f32, bw=64, bh=64)
bkg_use = 0.5 * (bkg_064.back() + bkg_128.back())
sys.stderr.write("subtract ... ")
cln_data = img_f32 - bkg_use
sys.stderr.write("done.\n")

## Save copies of background and subtracted images for inspection:
qsave('bkg_use.fits', bkg_use)
qsave('cln_data.fits', cln_data)

##--------------------------------------------------------------------------##
## Make initial catalog:
sys.stderr.write("Initial star extraction ... ")
cat = sep.extract(cln_data, sep_aduthresh)
sys.stderr.write("done.\n")

## Extract windowed positions:
sys.stderr.write("Standard windowed positions ... ")
xyf = sep.winpos(cln_data, cat['x'], cat['y'], sig=sep_win_sigma)
cat = append_fields(cat, ('wx', 'wy', 'wflag'), xyf, usemask=False)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##


## Note window flags:
wflag_raised = (cat['wflag'] > 0)
flagged_srcs = cat[wflag_raised]

## Examine position changes:
win_xdelta = cat['wx'] - cat['x']
win_ydelta = cat['wy'] - cat['y']
win_rdelta = np.hypot(win_xdelta, win_ydelta)

## Make a mask image that blocks out successful windowed centroids:
sys.stderr.write("Build a mask ... ")
tik = time.time()
cutoff = 1.0
wpos_pass = (win_rdelta < cutoff)
good_srcs = cat[wpos_pass]
srcs_mask = np.zeros_like(cln_data)
for sss in good_srcs:
    srcs_mask[sss['ymin']:sss['ymax'], sss['xmin']:sss['xmax']] = 1.0

#xslices = [slice(*x) for x in good_srcs[['xmin', 'xmax']]
tok = time.time()
msk_taken = tok - tik
sys.stderr.write("done.  Took %.3f seconds.\n" % msk_taken)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

tik = time.time()

## Original values of redone objects:
xyf_v1 = tuple([vv[~wpos_pass] for vv in xyf])

## Select objects that need a redo:
redo_srcs = cat[~wpos_pass]
xyf_v2 = sep.winpos(cln_data, redo_srcs['x'], redo_srcs['y'],
                  mask=srcs_mask, sig=sep_win_sigma)
tok = time.time()
redo_taken = tok - tik
sys.stderr.write("Rerunning winpos took %.3f seconds.\n" % redo_taken)

new_win_xdelta = xyf_v2[0] - redo_srcs['x']
new_win_ydelta = xyf_v2[1] - redo_srcs['y']

old_win_xdelta = redo_srcs['wx'] - redo_srcs['x']
old_win_ydelta = redo_srcs['wy'] - redo_srcs['y']

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

sys.exit(0)
##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(nrows=2, ncols=2, num=1, clear=True, figsize=fig_dims,
#                        sharex=True, squeeze=False)
# sharex='col' | sharex='row' | squeeze=False
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
#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (01_analyze_with_SEP.py):
#---------------------------------------------------------------------
#
#  2026-07-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created 01_analyze_with_SEP.py.
#

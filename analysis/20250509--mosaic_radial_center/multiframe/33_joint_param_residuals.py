#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Examine the astrometric residuals of individual detections that result
# from the joint fitting of entire QRUNIDs.
# 
# For a given QRUNID, the steps are:
#   1) Load best-fit parameters and image listings. These were previously
# calculated by 23_fine_tune_RUNID_WCS.py. This code is basically copied
# from 28_fit_sensor_transforms.py.
#   2) Load pickled detection catalogs for listed images. These catalogs
# were previously produced by 12_factor_fit_4pack.py. We keep only a few
# critical columns from each. Code copied from 28_fit_sensor_transforms.py.
#   3) For each image, evaluate parameters from joint fit in 'diags' mode
# to collect residuals and other relevant data.
#   4) Merge residuals from all images into a single, large collection for
# plotting. Given the number of sources, try to evaluate the whether we
# have reached the noise floor as a function of source flux. Also look for
# large-scale trends in the residuals. I would think that the lack of
# refraction and aberration correction would show up here.
#   5) Optionally iterate over multiple QRUNIDs.
#
# NOTES:
# * rather than iterate over QRUNIDs, I am going to load everything at
# once using code already in 28_fit_sensor_transforms.py. This should fit
# in memory without issues and provides a means of doing a GLOBAL examination
# of residuals across all time so far.
#
# Rob Siverd
# Created:       2026-05-14
# Last modified: 2026-05-14
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
import glob
#import io
import gc
import os
import ast
import sys
import time
#import pprint
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
#import scipy.stats as scst
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Coordinate solve helpers:
import slv_helper
reload(slv_helper)
slvh = slv_helper

## Solution parameter helpers:
import slv_par_tools
reload(slv_par_tools)
spt = slv_par_tools
quads = spt._quads



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
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()




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

## Extract RUNID from filename:
def runid_from_filename(filename):
    return os.path.basename(filename).split('_')[1]

## Extract observing year from RUNID:
def year_from_runid(runid):
    return int(runid[:2])

## Load parameter set from file:
def load_parameters(filename):
    with open(filename, 'r') as fff:
        return ast.literal_eval(fff.read())

## Extract CD matrix and CRPIX from parameter list:
def get_cdm_crpix(parameters):
    return np.array(parameters[:24]).reshape(4, -1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## List of available J-only joint parameter files:
par_flist = sorted(glob.glob('joint_pars/jpars_??????_J.txt'))
par_runid = [runid_from_filename(x) for x in par_flist]
par_files = dict(zip(par_runid, par_flist))

## Clean these out based on year/semester:
ythresh = 99
keepers = {}
for rr,fname in par_files.items():
    if year_from_runid(rr) <= ythresh:
    #if year_from_runid(rr) >= ythresh:
        keepers[rr] = fname
par_files = keepers

## Load those files:
#jnt_params = {kk:load_parameters(vv) for kk,vv in par_files.items()}
jnt_params = {}
jnt_inames = {}
for runid,fname in par_files.items():
    jnt_params[runid], jnt_inames[runid] = load_parameters(fname)

qrun_list = list(jnt_params.keys())

##--------------------------------------------------------------------------##
##------------------         Load Pickled Catalogs          ----------------##
##--------------------------------------------------------------------------##

## Only keep columns from gstars we actually use to save space:
gstcols = ['x', 'y', 'flux', 'flag',
           'fwhm', 'dumbsnr', 'realerr', 'gid', 'gra', 'gde']

## Iterate over QRUNIDs in use:
all_data = {}       # diags data from single-image best-fit (not used)
all_gstr = {}       # star catalog with updated Gaia matches
all_pars = {}       # initial, single-image best-fit parameters (not used)
for qrun in qrun_list:
    qrun_data, qrun_gstr, qrun_pars = {}, {}, {}    # per-QRUNID of above
    qrun_dir = 'results/%s' % qrun
    sys.stderr.write("qrun: %s\n" % qrun)
    sys.stderr.write("qrun_dir: %s\n" % qrun_dir)
    if not os.path.isdir(qrun_dir):
        sys.stderr.write("Error: folder not found: %s\n" % qrun_dir)
        sys.exit(1)
    #pickles = ['{}/{}.pickle'.format(qrun_dir, x) for x in jnt_inames[qrun]]
    use_pickles = {x:qrun_dir+'/'+x+'.pickle' for x in jnt_inames[qrun]}
    if not all([os.path.isfile(x) for x in use_pickles.values()]):
        sys.stderr.write("Error: pickled catalog(s) missing ...\n")
        sys.exit(1)
    # Load pickles:
    for pbase,ppath in use_pickles.items():
        _gstr, _data, _pars = load_pickled_object(ppath)
        trim_gstr = {qq:gg[gstcols].copy() for qq,gg in _gstr.items()}
        #qrun_gstr[pbase] = _gstr
        #qrun_gstr[pbase] = _gstr[gstcols].copy()    # keep useful columns
        qrun_gstr[pbase] = trim_gstr
        #qrun_data[pbase] = _data
        qrun_pars[pbase] = _pars
    # Stash finished dictionaries:
    all_gstr[qrun] = qrun_gstr
    #all_data[qrun] = qrun_data
    all_pars[qrun] = qrun_pars
    del qrun_gstr, qrun_data, qrun_pars
    del _gstr, _data, _pars

## Old all_gstr (full size): ~11.9 GiB
## New trim_gstr (useful cols): ~2.0 GiB

gcounts = {}
gc_avgs = {}
gc_meds = {}
for qrunid,dataset in all_gstr.items():
    imqstars = {img:{qq:len(cat) for qq,cat in vvv.items()} \
            for img,vvv in dataset.items()}
    gcounts[qrunid] = imqstars
    avgcount = np.mean([list(x.values()) for x in imqstars.values()], axis=0)
    medcount = np.median([list(x.values()) for x in imqstars.values()], axis=0)
    gc_avgs[qrunid] = dict(zip(spt._quads, avgcount))
    gc_meds[qrunid] = dict(zip(spt._quads, medcount))
    pass


##--------------------------------------------------------------------------##
## Work an example ...
sys.stderr.write("\n%s\n" % fulldiv)
sys.stderr.write("Do some analysis ...\n")
working = qrun_list[:1]
for qrun in working:

    #jpars = jnt_params[qrun]
    #run_= get_cdm_crpix(jpars)
    run_jntpar = np.array(jnt_params.get(qrun))
    run_imlist = jnt_inames.get(qrun)
    run_gstars = all_gstr.get(qrun)
    run_diags  = spt.multi_squared_residuals_foc2ccd_rdist(run_jntpar,
                                         run_gstars, run_imlist, diags=True)

    # first, let's see what the typical residual is like in every frame ...
    for ibase,diags in run_diags.items():

        pass
    pass

# diags['NE']['dumbsnr'], diags['NE']['scaled_rerror']
# plt.scatter(25-np.log(diags['NE']['flux']),
#                       diags['NE']['rerror'], lw=0,s=15)

#plt.scatter(diags['NE']['xmeas'], diags['NE']['ymeas'], 
#            c=diags['NE']['rerror'], lw=0, s=15)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
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
#fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig, axs = plt.subplots(nrows=2, ncols=2, num=1, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=False)
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

axmap = dict(zip(spt.sensor_order, axs.flatten()))

## Bin cell config:
cells = 32
cellpix = 2048 / cells

#cells = 32
#cellpix = 2048 / cells
skw = {'lw':0, 's':15, 'vmax':0.5}
count = 0
#rerr_max_pix = 0.75
pixerr_thresh =   0.75
objsnr_thresh = 100.
everything = {qq:[] for qq in spt.sensor_order}
cleanstuff = {qq:[] for qq in spt.sensor_order}
dirtystuff = {qq:[] for qq in spt.sensor_order}
clean_text = 'pixerr <= %.2f AND SNR >= %.0f' % (pixerr_thresh, objsnr_thresh)
for ibase,diags in run_diags.items():
    count += 1
    for qq,ax in axmap.items():
        tdata = pd.DataFrame.from_dict(diags[qq])
        tdata['xbin'] = np.int_(tdata['xmeas'] / cellpix)
        tdata['ybin'] = np.int_(tdata['ymeas'] / cellpix)
        #which = (tdata['dumbsnr'] > 250.)
        bright = (tdata['dumbsnr'] >= objsnr_thresh)
        nearby = (tdata[ 'rerror'] <= pixerr_thresh)
        which = bright & nearby
        clean = tdata[which].copy()
        dirty = tdata[~which].copy()
        #clean['xbin'] = np.int_(clean['xmeas'] / cellpix)
        #clean['ybin'] = np.int_(clean['ymeas'] / cellpix)
        everything[qq].append(tdata)
        cleanstuff[qq].append(clean)
        dirtystuff[qq].append(dirty)
        ax.scatter(clean['xmeas'], clean['ymeas'], 
            c=clean['rerror'], **skw)
        ax.set_aspect('equal')

    if count > 1000:
        break

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

##--------------------------------------------------------------------------##
##------------------         CONCATENATE DATAFRAMES         ----------------##
##--------------------------------------------------------------------------##


## Concatenate DataFrames:
everything = {qq:pd.concat(ll) for qq,ll in everything.items()}
cleanstuff = {qq:pd.concat(ll) for qq,ll in cleanstuff.items()}
dirtystuff = {qq:pd.concat(ll) for qq,ll in dirtystuff.items()}

### Add 
#for qq,bigdf in everything.items():
#    bigdf['xbin'] = np.int_(bigdf['xmeas'] / cellpix)
#    bigdf['ybin'] = np.int_(bigdf['ymeas'] / cellpix)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Another figure with heatmaps:
fg2, axs2 = plt.subplots(nrows=2, ncols=2, num=2, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=False)
axmap2 = dict(zip(spt.sensor_order, axs2.flatten()))

## Iterate and make heatmaps:
hmap = {}
lnorm = LogNorm(vmin=0.05, vmax=1.0)
#pckw = {'vmin':0.05, 'vmax':0.65}
pckw = {'vmin':0.05, 'vmax':0.5}
#pckw = {'norm':LogNorm(vmin=0.05, vmax=1.0)}
for qq,bigdf in everything.items():
    #groups = bigdf.groupby(['xbin', 'ybin'])
    bam = bigdf.groupby(['ybin', 'xbin'])['rerror'].median()
    hmap[qq] = bam
    # plt.pcolor(bam.unstack())
    #axmap2[qq].pcolor(bam.unstack(), **pckw)
    ax = axmap2[qq]
    ax.imshow(bam.unstack(), **pckw)
    #ax.imshow(np.sqrt(bam.unstack()))
    ax.yaxis.set_inverted(False)
    ax.set_aspect('equal')

fg2.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fg2.savefig(plot_name, bbox_inches='tight')
 
## IS THERE NONSENSE IN THE CORNERS?
swdata = everything['SW']
lrcorn = 2048, 1
lrdist = np.hypot(swdata.xmeas - 2048., swdata.ymeas - 1.0)
bogus  = lrdist <= 150.
sys.stderr.write("Nsrcs SW extreme LR corner: %d\n" % np.sum(bogus))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


## Radial distance/error plot:
fg3, axs3 = plt.subplots(nrows=2, ncols=2, num=3, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=True)
#fg3 = plt.figure(3, figsize=fig_dims)
#fg3.clf()
#ax3 = fg3.add_subplot(111)
rskw = {'lw':0, 's':5}
xeax, yeax, reax, prax = axs3.flatten()
for qq,bigdf in everything.items():
    xeax.scatter(bigdf.rdist, bigdf.xerror, label=qq, **rskw)
    yeax.scatter(bigdf.rdist, bigdf.yerror, label=qq, **rskw)
    reax.scatter(bigdf.rdist, bigdf.rerror, label=qq, **rskw)
    #prax.scatter(bigdf.rdist, 

    pass
xeax.set_ylabel('X error [pix]')
yeax.set_ylabel('Y error [pix]')
axs3[0,0].legend(loc='upper left')

ndecades = 1.0
#axs3[0].set_yscale('symlog', basey=10, linthreshy=0.5, linscaley=ndecades)
#axs3[0,0].set_yscale('log')
#axs3[0,0].set_ylim(bottom=0.003)
axs3[1,0].set_ylabel("R error [pix]")
axs3[1,0].set_xlabel("Distance from CRPIX [pix]")
fg3.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Quiver config:
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
#ascale = 100
ascale = 50
fig_dims = (10, 9)

## Quiver plot with everything:
fg5, axs5 = plt.subplots(nrows=2, ncols=2, num=5, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=True)
axmap5 = dict(zip(spt.sensor_order, axs5.flatten()))
for qq,bigdf in everything.items():
    ax = axmap5[qq]
    #bam = bigdf.groupby(['ybin', 'xbin'])['rerror'].median()
    xpos = bigdf.groupby(['ybin', 'xbin'])['xmeas'].median()
    ypos = bigdf.groupby(['ybin', 'xbin'])['ymeas'].median()
    xerr = bigdf.groupby(['ybin', 'xbin'])['xerror'].median()
    yerr = bigdf.groupby(['ybin', 'xbin'])['yerror'].median()
    ax.quiver(xpos, ypos, ascale*xerr, ascale*yerr, color='r', **qxkw)
    ax.set_aspect('equal')
fg5.suptitle('everything')
fg5.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()

## Same plot, clean objects only though:
fg6, axs6 = plt.subplots(nrows=2, ncols=2, num=6, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=True)
axmap6 = dict(zip(spt.sensor_order, axs6.flatten()))
for qq,bigdf in cleanstuff.items():
    ax = axmap6[qq]
    #bam = bigdf.groupby(['ybin', 'xbin'])['rerror'].median()
    xpos = bigdf.groupby(['ybin', 'xbin'])['xmeas'].median()
    ypos = bigdf.groupby(['ybin', 'xbin'])['ymeas'].median()
    xerr = bigdf.groupby(['ybin', 'xbin'])['xerror'].median()
    yerr = bigdf.groupby(['ybin', 'xbin'])['yerror'].median()
    ax.quiver(xpos, ypos, ascale*xerr, ascale*yerr, color='r', **qxkw)
    ax.set_aspect('equal')
fg6.suptitle(clean_text)
fg6.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()








#ndecades = 1.0   # for symlog, set width of linear portion in units of dex
#nonposx='mask' | nonposx='clip' | nonposy='mask' | nonposy='clip'
#ax1.set_xscale('log', basex=10, nonposx='mask', subsx=mtickpos)
#ax1.set_xscale('log', nonposx='clip', subsx=[3])
#ax1.set_yscale('symlog', basey=10, linthreshy=0.1, linscaley=ndecades)

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
# CHANGELOG (33_joint_param_residuals.py):
#---------------------------------------------------------------------
#
#  2026-05-14:
#     -- Increased __version__ to 0.0.1.
#     -- First created 33_joint_param_residuals.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Test load a pickled WCS solution from astrometry.net/astroquery.
#
# Example commands:
# ./11_test_load.py -i solutions/wircam_J_2507756p.fits.fz.fcat.p2.pickle
# ./11_test_load.py -i solutions/wircam_H2_2626497p.fits.fz.fcat.p3.pickle
# ./11_test_load.py -i solutions/wircam_J_1069950p.fits.fz.fcat.p4.pickle
# ./11_test_load.py -i solutions/wircam_J_1662729p.fits.fz.fcat.p5.pickle
#
#
#
# Rob Siverd
# Created:       2024-06-28
# Last modified: 2024-06-28
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
import argparse
#import shutil
#import resource
#import signal
import glob
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

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    sys.stderr.write("failed to import extended_catalog module!")
    sys.exit(1)

## Make objects:
ccc = ec.ExtendedCatalog()

## Angular math:
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
    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

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
    Test load ast.net solution from pickle file.
    
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
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-i', '--input_file', default=None, required=True,
            help='input pickle filename', type=str)
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

## Grab the ibase from a filename:
def ibase_from_filename(fits_path):
    return os.path.basename(fits_path).split('.')[0]

## Create a by-ibase lookup dictionary from a list of paths:
def make_ibase_lookup_dict(files_list):
    ibases = [ibase_from_filename(x) for x in files_list]
    lookup = dict(zip(ibases, files_list))
    return lookup

## Load the list of fcat paths that created the solutions:
fp_file = 'fcat_paths.txt'
with open(fp_file, 'r') as fff:
    fcat_paths = [x.strip() for x in fff.readlines()]

## Make a lookup table using image base:
fcat_ibases = [ibase_from_filename(x) for x in fcat_paths]
#ibase2fcat  = dict(zip(fcat_ibases, fcat_paths))
ibase2fcat = make_ibase_lookup_dict(fcat_paths)

##--------------------------------------------------------------------------##

## Load a list of all pickled solutions:
p2_pickle_list = sorted(glob.glob('solutions/wircam_*.p2.pickle'))
p3_pickle_list = sorted(glob.glob('solutions/wircam_*.p3.pickle'))
p4_pickle_list = sorted(glob.glob('solutions/wircam_*.p4.pickle'))
p5_pickle_list = sorted(glob.glob('solutions/wircam_*.p5.pickle'))

## Make ibase lists and mappings for solutions:
#p2_ibases = [ibase_from_filename(x) for x in p2_pickle_list]
#p3_ibases = [ibase_from_filename(x) for x in p3_pickle_list]
#p4_ibases = [ibase_from_filename(x) for x in p4_pickle_list]
#p5_ibases = [ibase_from_filename(x) for x in p5_pickle_list]
ibase2pickle2 = make_ibase_lookup_dict(p2_pickle_list)
ibase2pickle3 = make_ibase_lookup_dict(p3_pickle_list)
ibase2pickle4 = make_ibase_lookup_dict(p4_pickle_list)
ibase2pickle5 = make_ibase_lookup_dict(p5_pickle_list)

pickle_lookups = {2:ibase2pickle2, 3:ibase2pickle3, 
                  4:ibase2pickle4, 5:ibase2pickle5}
#pickle_lookups = [ibase2pickle2, ibase2pickle3, ibase2pickle4, ibase2pickle5]
#solve_pickles = sorted(glob.glob('solutions/wircam_*.p?.pickle'))

##--------------------------------------------------------------------------##

## Abort if input file not found:
if not os.path.isfile(context.input_file):
    sys.stderr.write("\nError: input file not found: %s\n" % context.input_file)
    sys.exit(1)

## Get the ibase from the input file:
solve_ibase = ibase_from_filename(context.input_file)
if solve_ibase not in ibase2fcat.keys():
    sys.stderr.write("\nError: unrecognized ibase: %s\n" % solve_ibase) 
    sys.exit(1)

## Make a list of known solution pickles:
have_pickles = {}
for pp,lut in pickle_lookups.items():
    if solve_ibase in lut.keys():
        have_pickles[pp] = lut.get(solve_ibase)
npickles = len(have_pickles)
sys.stderr.write("Found %d pickled solutions.\n" % npickles)
if npickles < 2:
    sys.stderr.write("WARNING!!!! Only one solution, hijinx may ensue ...\n")
    sys.exit(1)
if npickles < 3:
    sys.stderr.write("Bailing ... few solves.\n")
    sys.exit(1)

## Load all pickled solution headers:
wcs_headers = {pp:load_pickled_object(ff) for pp,ff in have_pickles.items()}

## Initialize WCS objects from solution headers:
wcses = {pp:awcs.WCS(hh) for pp,hh in wcs_headers.items()}

## Load data from pickle:
#wcs_header = load_pickled_object(context.input_file)

## Initialize a WCS object:
#ww = awcs.WCS(wcs_header)

##--------------------------------------------------------------------------##
## Load the corresponding fcat if we made it this far:
this_fcat = ibase2fcat.get(solve_ibase)
ccc.load_from_fits(this_fcat)
imcat = ccc.get_catalog()
nsrcs = len(imcat)

##--------------------------------------------------------------------------##
##------------------       Variation Among Solutions        ----------------##
##--------------------------------------------------------------------------##

## Pixel positions for WCS evaluation:
ny, nx = 2048, 2048
#x_list = (0.5 + np.arange(nx)) / nx - 0.5            # relative (centered)
#y_list = (0.5 + np.arange(ny)) / ny - 0.5            # relative (centered)
#x_list = 
halfstep = 8
x_list = np.arange(0, nx, 2*halfstep) + halfstep
y_list = np.arange(0, ny, 2*halfstep) + halfstep
xx, yy = np.meshgrid(x_list, y_list)                 # sparse
#xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))   # absolute

## Compute sky positions for each reference pixel point:
calc_ra_pos = []
calc_de_pos = []
for pp,ww in wcses.items():
    sys.stderr.write("Poly order: %d\n" % pp)
    ra, de = ww.all_pix2world(xx, yy, 1)
    calc_ra_pos.append(ra)
    calc_de_pos.append(de)
#calc_ra_cube = np.array(calc_ra_pos)
#calc_de_cube = np.array(calc_de_pos)
calc_ra_cube = np.dstack(calc_ra_pos)
calc_de_cube = np.dstack(calc_de_pos)

## Avg RA/DE at each point:
calc_ra_avg  = np.mean(calc_ra_cube, axis=2)
calc_de_avg  = np.mean(calc_de_cube, axis=2)

## Stddev at each point:
calc_ra_std  = np.std(calc_ra_cube, axis=2) * np.cos(np.radians(calc_de_avg))
calc_de_std  = np.std(calc_de_cube, axis=2)

## In arcsec:
calc_ra_std *= 3600.0
calc_de_std *= 3600.0

## Layer-by-layer, with proper trig:
ang_seps = np.zeros_like(calc_ra_cube)
for ii in range(calc_ra_cube.shape[2]):
    ang_seps[:, :, ii] = 3600*angle.dAngSep(calc_ra_cube[:,:,ii], calc_de_cube[:,:,ii],
                                            calc_ra_avg, calc_de_avg)
    pass

## Worst-case difference at each location:
worst_per_pos = np.max(ang_seps, axis=2)

## Worst-case difference at any grid point:
worst_offset  = np.max(ang_seps)

##--------------------------------------------------------------------------##
## Check 

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
# CHANGELOG (11_test_load.py):
#---------------------------------------------------------------------
#
#  2024-06-28:
#     -- Increased __version__ to 0.0.1.
#     -- First created 11_test_load.py.
#

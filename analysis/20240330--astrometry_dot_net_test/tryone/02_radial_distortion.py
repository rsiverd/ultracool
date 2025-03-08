#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Estimate / inspect radial distortion using known-good ast.net positions.
#
# Rob Siverd
# Created:       2025-01-29
# Last modified: 2025-01-29
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
#import pickle
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
#import scipy.stats as scst
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
import matplotlib.collections as mcoll
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
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

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

## Tangent projection:
import tangent_proj as tp

## Gaia catalog matching:
import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()

## Helpers for this investigation:
import helpers
reload(helpers)

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

### Various from astropy:
#try:
##    import astropy.io.ascii as aia
##    import astropy.io.fits as pf
##    import astropy.io.votable as av
##    import astropy.table as apt
#    import astropy.time as astt
##    import astropy.wcs as awcs
##    from astropy import constants as aconst
##    from astropy import coordinates as coord
##    from astropy import units as uu
#except ImportError:
##    logger.error("astropy module not found!  Install and retry.")
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

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## WCS defaults:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

## Configuration:
npoly = 5
cmethod = 'sap'     # centroid method ('sap' or 'win')
#cmethod = 'win'

#jra_keyspec = {'sap':'jntupd_ra', 'win':'wjntupd_ra'}
#jde_keyspec = {'sap':'jntupd_de', 'win':'wjntupd_de'}
#_jra_key = jra_keyspec.get(cmethod)
#_jde_key = jde_keyspec.get(cmethod)

## Centroid method fallout:
#if cmethod == 'sap':
#    _jra_key = 'jntupd_ra'
#    _jde_key = 'jntupd_de'

## Input files:
this_fcat = 'wircam_J_2413738p_eph.fits.fz.fcat'
anet_data_sap = 'data/srcs_ast_sap.p%d.txt' % npoly
anet_data_win = 'data/srcs_ast_win.p%d.txt' % npoly
wcs_params = 'wircam_J_2413738p_eph.fits.fz.fcat.txt'
#gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.csv'
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d3.csv'

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Load Gaia catalog:
sys.stderr.write("Loading Gaia ... ")
gm.load_sources_csv(gaia_csv_path)
sys.stderr.write("done.\n")

## Load saved WCS parameters from jointupd process:
wdata = pd.read_csv(wcs_params)
wpars = wdata.values[0].copy()
#wpars[4] -= 3.0 / 3600.
#wpars[4] += 3.0 / 3600.
#wpars[4] -= 1.0 / 3600.
#wpars[0] *= 1.01
#wpars[1] = 0.0
#wpars[2] = 0.0
#wpars[3] *= 1.01

## Load ast.net-derived coordinates:
sys.stderr.write("Loading sap ... ")
sap_xx, sap_yy, sap_ra, sap_de = helpers.load_skypix_output(anet_data_sap)
#sap_cosdec = np.cos(np.radians(sap_de))
sys.stderr.write("win ... ")
win_xx, win_yy, win_ra, win_de = helpers.load_skypix_output(anet_data_win)
#win_cosdec = np.cos(np.radians(win_de))
sys.stderr.write("done.\n")

sys.stderr.write("Loading %s ... " % this_fcat)
ccc.load_from_fits(this_fcat)
stars = ccc.get_catalog()
imhdr = ccc.get_header()
sys.stderr.write("done.\n")

## Initialize Gaia matcher with appropriate time:
obs_time = helpers.wircam_timestamp_from_header(imhdr)
gm.set_epoch(obs_time)

##--------------------------------------------------------------------------##
## Sanity check:
if     (not np.allclose(stars['wx'], win_xx, atol=1e-3)) \
    or (not np.allclose(stars['wy'], win_yy, atol=1e-3)) \
    or (not np.allclose(stars[ 'x'], sap_xx, atol=1e-3)) \
    or (not np.allclose(stars[ 'y'], sap_yy, atol=1e-3)):
    sys.stderr.write("Inconsistent!!\n")
    sys.exit(1)

def choose_errors(tra, tde):
    de_offset, de_scatter = rs.calc_ls_med_IQR(tde)
    ra_offset, ra_scatter = rs.calc_ls_med_IQR(tra)
    ra_text = r"$\sigma_\alpha = %.3f$ mas" % ra_scatter
    de_text = r"$\sigma_\delta = %.3f$ mas" % de_scatter
    #return ra_offset, ra_scatter, ra_text, de_offset, de_scatter, de_text
    return tra, ra_scatter, ra_text, tde, de_scatter, de_text

##--------------------------------------------------------------------------##
## Sort out coordinate types:
if cmethod == 'sap':
    jnt_ra, jnt_de = stars[ 'jntupd_ra'], stars[ 'jntupd_de']
    ant_ra, ant_de = sap_ra, sap_de
    ant_xx, ant_yy = stars[ 'x'], stars[ 'y']
elif cmethod == 'win':
    jnt_ra, jnt_de = stars['wjntupd_ra'], stars['wjntupd_de']
    ant_ra, ant_de = win_ra, win_de
    ant_xx, ant_yy = stars['wx'], stars['wy']
else:
    sys.stderr.write("Unhandled cmethod: '%s'\n" % cmethod)
    sys.exit(1)
ant_cosdec = np.cos(np.radians(ant_de))

##--------------------------------------------------------------------------##
## Measure difference between 'jntupd' coordinates (current distortion model)
## and the inferred coordinates from the astrometry.net fit:
#if cmethod == 'sap':
#    shift_de_mas = 3.6e6 * (sap_de - stars[ 'jntupd_de'])
#    shift_ra_mas = 3.6e6 * (sap_ra - stars[ 'jntupd_ra']) * sap_cosdec
#if cmethod == 'win':
#    shift_de_mas = 3.6e6 * (win_de - stars['wjntupd_de'])
#    shift_ra_mas = 3.6e6 * (win_ra - stars['wjntupd_ra']) * win_cosdec
shift_de_mas = 3.6e6 * (ant_de - jnt_de)
shift_ra_mas = 3.6e6 * (ant_ra - jnt_ra) * ant_cosdec

de_offset, de_scatter = rs.calc_ls_med_IQR(shift_de_mas)
ra_offset, ra_scatter = rs.calc_ls_med_IQR(shift_ra_mas)

## Also measure difference between dra/dde coordinates
xrel = stars['x'] - crpix1
yrel = stars['y'] - crpix2
calc_ra, calc_de = helpers.eval_cdmcrv(wpars, xrel, yrel)
calc_ra %= 360.0

#dist_de_mas = 3.6e6 * (sap_de - calc_de)
#dist_ra_mas = 3.6e6 * (sap_ra - calc_ra) * sap_cosdec
anet_dist_de_mas = 3.6e6 * (ant_de - calc_de)
anet_dist_ra_mas = 3.6e6 * (ant_ra - calc_ra) * ant_cosdec

##--------------------------------------------------------------------------##
## Perform a Gaia match using the ast.net solution:
sys.stderr.write("Gaia matching ... ")
mtol_arcsec = 0.3
matches = gm.twoway_gaia_matches(ant_ra, ant_de, mtol_arcsec)
idx, gra, gde, gid = matches
gcosdec = np.cos(np.radians(gde))
sys.stderr.write("done.\n")

## Downselect other vectors:
agm_xx, agm_yy =  ant_xx[idx],  ant_yy[idx]
agm_ra, agm_de =  ant_ra[idx],  ant_de[idx]
cgm_ra, cgm_de = calc_ra[idx], calc_de[idx]
gaia_dist_de_mas = 3.6e6 * (gde - cgm_de)
gaia_dist_ra_mas = 3.6e6 * (gra - cgm_ra) * gcosdec

## Gaia vs anet:
gaia_anet_de_mas = 3.6e6 * (gde - agm_de)
gaia_anet_ra_mas = 3.6e6 * (gra - agm_ra) * gcosdec

## Gaia dewarped positions on the detector:
gxx_rel, gyy_rel = helpers.inverse_tan_cdmcrv(wpars, gra, gde)
gxx_pix = gxx_rel + crpix1 - 1.25
gyy_pix = gyy_rel + crpix2 + 1.25
grr_pix = np.hypot(gxx_rel, gyy_rel)

## X- and Y-offsets (== corr_i - meas_i):
xnudge = gxx_pix - agm_xx
ynudge = gyy_pix - agm_yy
rnudge = np.hypot(xnudge, ynudge)

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
## Make a ton of lines pointing in the offset direction to overlap the 
## focal plane center. The starting points will be (gxx_pix, gyy_pix) and
## the ending points will be (qend_x, qend_y):
multiplier = 2.0 * grr_pix / rnudge
qend_x = gxx_pix + multiplier * xnudge
qend_y = gyy_pix + multiplier * ynudge
draw_me = (rnudge > 10)
draw_me = (2000 < grr_pix)
draw_me = ((gxx_pix <  500) & (gyy_pix > 1600)) \
        | ((gxx_pix <  500) & (gyy_pix <  500)) \
        | ((gxx_pix > 1600) & (gyy_pix > 1600))
#draw_me = (gxx_pix < 500) & (gyy_pix > 1600)
#draw_me = (gxx_pix < 500) & (gyy_pix <  500)
#segments = [[(a,b),(c,d)] for a,b,c,d in zip(gxx_pix, gyy_pix, qend_x, qend_y)]
segs1 = [[(a,b),(c,d)] for a,b,c,d,w in zip(gxx_pix, gyy_pix, qend_x, qend_y, draw_me) if w]
sclr1 = ['r' for s in segs1]
draw_me  = (500 < grr_pix) & (grr_pix < 1000)
segs2 = [[(a,b),(c,d)] for a,b,c,d,w in zip(gxx_pix, gyy_pix, qend_x, qend_y, draw_me) if w]
sclr2 = ['b' for s in segs2]
segs2, sclr2 = [], []
linecoll = mcoll.LineCollection(segs1+segs2, colors=sclr1+sclr2, linewidths=0.10)

## A separate figure for fiddling in pixel-space:
fig_dims = (10, 9)
fg2 = plt.figure(2, figsize=fig_dims)
fg2.clf()
pax = fg2.add_subplot(111, aspect='equal')
pax.grid(True)
pax.add_collection(linecoll)
pax.quiver(agm_xx, agm_yy, xnudge, ynudge)
pax.scatter(crpix1, crpix2, s=100, c='b')
fg2.tight_layout()

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style

#fig_dims = (9, 9)
fig_dims = (10, 9)
fig = plt.figure(1, figsize=fig_dims)
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111)
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.axis('off')

#dx = stars['x'][idx] - gxx_pix #gxx_ra
#dy = stars['y'][idx] - gyy_pix #gxx_ra

## Ast.net polynomial vs jointupd:
roff, rsig, ra_text, doff, dsig, de_text = \
        choose_errors(shift_ra_mas, shift_de_mas)
fig.clf()
ax1 = fig.add_subplot(221, aspect='equal')
ax1.grid(True)
ax1.quiver(stars['x'], stars['y'], -roff, doff) #, shift_de_mas)
#ax1.quiver(stars['x'][idx], stars['y'][idx], dx, dy) #, shift_de_mas)
ax1.set_xlabel("X pixel")
ax1.set_ylabel("Y pixel")
ax1.set_title("ast.net poly=%d vs. joint-updated\n%s | %s"
        % (npoly, ra_text, de_text))

plot_name = 'dist_quiver.anet%d_vs_jnt.png' % npoly
#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

## Ast.net polynomial vs nominal undistorted:
roff, rsig, ra_text, doff, dsig, de_text = \
        choose_errors(anet_dist_ra_mas, anet_dist_de_mas)
#fig.clf()
ax2 = fig.add_subplot(222, aspect='equal')
ax2.grid(True)
ax2.quiver(stars['x'], stars['y'], -roff, doff) # east positive LEFT
ax2.set_xlabel("X pixel")
#ax1.set_ylabel("Y pixel")
ax2.set_title("ast.net poly=%d vs. distorted\n%s | %s"
        % (npoly, ra_text, de_text))

## Gaia vs ast.net polynomial:
roff, rsig, ra_text, doff, dsig, de_text = \
        choose_errors(gaia_anet_ra_mas, gaia_anet_de_mas)
ax3 = fig.add_subplot(223, aspect='equal')
ax3.grid(True)
ax3.quiver(agm_xx, agm_yy, -roff, doff) # east positive LEFT
ax3.set_xlabel("X pixel")
ax3.set_title("ast.net poly=%d vs. gaia\n%s | %s"
        % (npoly, ra_text, de_text))

## Gaia vs nominal undistorted:
roff, rsig, ra_text, doff, dsig, de_text = \
        choose_errors(gaia_dist_ra_mas, gaia_dist_de_mas)
ax4 = fig.add_subplot(224, aspect='equal')
ax4.grid(True)
ax4.quiver(agm_xx, agm_yy, -roff, doff) # east positive LEFT
ax4.set_xlabel("X pixel")
ax4.set_title("gaia vs. distorted\n%s | %s" % (ra_text, de_text))

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

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
# CHANGELOG (02_radial_distortion.py):
#---------------------------------------------------------------------
#
#  2025-01-29:
#     -- Increased __version__ to 0.0.1.
#     -- First created 02_radial_distortion.py.
#

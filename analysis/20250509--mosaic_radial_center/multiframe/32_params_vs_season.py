#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Evaluate the stability of various joint fit parameters.
#
# Rob Siverd
# Created:       2026-05-12
# Last modified: 2026-05-12
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
import glob
#import io
import gc
import os
import ast
import sys
import time
import pprint
#import pickle
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
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
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
## New-style string formatting (more at https://pyformat.info/):

#oldway = '%s %s' % ('one', 'two')
#newway = '{} {}'.format('one', 'two')

#oldway = '%d %d' % (1, 2)
#newway = '{} {}'.format(1, 2)

# With padding:
#oldway = '%10s' % ('test',)        # right-justified
#newway = '{:>10}'.format('test')   # right-justified
#oldway = '%-10s' % ('test',)       #  left-justified
#newway = '{:10}'.format('test')    #  left-justified

# Ordinally:
#newway = '{1} {0}'.format('one', 'two')     # prints "two one"

# Dictionarily:
#newway = '{lastname}, {firstname}'.format(firstname='Rob', lastname='Siverd')

# Centered (new-only):
#newctr = '{:^10}'.format('test')      # prints "   test   "

# Numbers:
#oldway = '%06.2f' % (3.141592653589793,)
#newway = '{:06.2f}'.format(3.141592653589793)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Extract RUNID from filename:
def runid_from_filename(filename):
    return os.path.basename(filename).split('_')[1]

## Load parameter set from file:
def load_parameters(filename):
    with open(filename, 'r') as fff:
        return ast.literal_eval(fff.read())

## Extract CD matrix and CRPIX from parameter list:
def get_cdm_crpix(parameters):
    return np.array(parameters[:24]).reshape(4, -1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## List of available J-only joint parameter files:
par_flist = sorted(glob.glob('joint_pars/jpars_??????_J.txt'))

runid_list = [runid_from_filename(x) for x in par_flist]
par_files = dict(zip(runid_list, par_flist))

## Load those files:
#raw_params = {kk:load_parameters(vv) for kk,vv in par_files.items()}
raw_params = {}
raw_inames = {}
for runid,fname in par_files.items():
    raw_params[runid], raw_inames[runid] = load_parameters(fname)

## Params over time:
par_stack = np.dstack([get_cdm_crpix(raw_params[x]) for x in runid_list])
ne_pstack, nw_pstack, se_pstack, sw_pstack = par_stack

##--------------------------------------------------------------------------##
##------------------         Load fcats and seasonal        ----------------##
##--------------------------------------------------------------------------##

## Coordinates:
cfht_latNdeg =   19.825252
cfht_lonEdeg = -155.468876

## Files:
fcat_list = 'fcat_paths.csv'
season_data = 'seasonal_data.csv'

## Load the fcat table:
sys.stderr.write("Loading %s ... " % fcat_list)
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
cat_table = pd.read_csv(fcat_list, **pdkwargs)

## Load the seasonal data:
sys.stderr.write("%s ... " % season_data)
seasonal = pd.read_csv(season_data, **pdkwargs)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------       Deal with LST/HA and similar     ----------------##
##--------------------------------------------------------------------------##

twopi = 2.0 * np.pi

## Convert HHMMSS.SS to degrees:
def hms2deg(hmstxt):
    th, tm, ts = hmstxt.split(':')
    is_neg = hmstxt.startswith('-')
    mm = float(ts) / 60.0 + float(tm)
    hh = float(th) - mm/60. if is_neg else float(th) + mm/60.
    return 15.0*hh

## This works because LST doesn't get close to 360 ...
lst1 = np.array([hms2deg(x) for x in seasonal.SIDTIME])
lst2 = np.array([hms2deg(x) for x in seasonal.LSTEND])
lst_deg = 0.5 * (lst1 + lst2)

## HA is +/- and doesn't wrap, also OK for this method:
ha1 = np.array([hms2deg(x) for x in seasonal.HA])
ha2 = np.array([hms2deg(x) for x in seasonal.HAEND])
ha_deg = 0.5 * (ha1 + ha2)

## Compute parallactic angle:
#r_dec = np.radians(seasonal.CRVAL2 - 20)
r_dec = np.radians(seasonal.CRVAL2)
numer = np.sin(np.radians(ha_deg))
denom = np.tan(np.radians(cfht_latNdeg)) * np.cos(r_dec) \
        - np.sin(r_dec) * np.cos(np.radians(ha_deg))
#parang_rad = np.arctan2(numer, denom)
parang_rad = np.arctan2(numer, denom) % twopi
#parang_rad = np.arctan(numer / denom)
parang_deg = np.degrees(parang_rad)
seasonal['PARANG'] = parang_deg

##--------------------------------------------------------------------------##
##------------------         Mean/stddev of parameters      ----------------##
##--------------------------------------------------------------------------##

## Compute per-RUNID averages and stddevs for various seasonal quantities:
## 'MCTR_RA', 'MCTR_DEC'
statcols = ['JDTDB', 'AIRMASS', 'TELALT', 'TELAZ', 'CRVAL1', 'CRVAL2',
            'OBS_X', 'OBS_Y', 'OBS_Z', 'OBS_VX', 'OBS_VY', 'OBS_VZ', 'PARANG']
rstats = {}
for runid,imlist in raw_inames.items():
    subset = seasonal[seasonal.qrunid == runid]
    things = {}
    for col in statcols:
        #things[col] = {'avg':np.average(subset[col]),
        #               'med':np.median(subset[col]),
        #               'std':np.std(subset[col])}
        #sys.stderr.write("avg %s : %f\n" % (col, np.average(subset[col])))
        #sys.stderr.write("std %s : %f\n" % (col, np.std(subset[col])))
        things[col] = (np.average(subset[col]), np.std(subset[col]))
    rstats[runid] = things
    pass

##--------------------------------------------------------------------------##
##------------------         Seasonal Helper Routines       ----------------##
##--------------------------------------------------------------------------##

## RA/Dec to Cartesian:
def rade2xyz(rad_ra, rad_de):
    x = np.cos(rad_de) * np.cos(rad_ra)
    y = np.cos(rad_de) * np.sin(rad_ra)
    z = np.sin(rad_de)
    return np.vstack((x, y, z))

## Convert Cartesian points to RA, Dec:
def xyz2equ(xyz_pts):
    # Shape/dimension sanity check:
    if ((xyz_pts.ndim != 2) or (xyz_pts.shape[0] != 3)):
        sys.stderr.write("XYZ points have wrong shape!\n")
        return (0,0)
    tx = np.array(xyz_pts[0]).flatten()
    ty = np.array(xyz_pts[1]).flatten()
    tz = np.array(xyz_pts[2]).flatten()
    ra = np.arctan2(ty, tx)
    de = np.arcsin(tz)
    equ_coo = np.vstack((ra, de))
    return equ_coo

## Vector length:
def veclen(vec):
    return np.sqrt(np.sum(vec*vec, axis=0))

##--------------------------------------------------------------------------##
##------------------         CRPIX Breakout/Diffs           ----------------##
##--------------------------------------------------------------------------##

## CRPIX breakout and plots:
ne_crpix1, ne_crpix2 = ne_pstack[4:]
nw_crpix1, nw_crpix2 = nw_pstack[4:]
se_crpix1, se_crpix2 = se_pstack[4:]
sw_crpix1, sw_crpix2 = sw_pstack[4:]

ne_nw_dx = ne_crpix1 - nw_crpix1 ; ne_nw_dy = ne_crpix2 - nw_crpix2
se_sw_dx = se_crpix1 - sw_crpix1 ; se_sw_dy = se_crpix2 - sw_crpix2
ne_se_dx = ne_crpix1 - se_crpix1 ; ne_se_dy = ne_crpix2 - se_crpix2
nw_sw_dx = ne_crpix1 - se_crpix1 ; nw_sw_dy = ne_crpix2 - se_crpix2

## For good measure (and transformation guess):
ne_sw_dx = ne_crpix1 - sw_crpix1 ; ne_sw_dy = ne_crpix2 - sw_crpix2

##--------------------------------------------------------------------------##
##------------------         CD Matrix / PA Breakout        ----------------##
##--------------------------------------------------------------------------##

## Breakdown from:
## https://arxiv.org/html/2602.04041v1
def analyze(cddata):
    cd11, cd12, cd21, cd22 = cddata.ravel()[:4]
    xscale = np.sqrt(cd11*cd11 + cd12*cd12)
    yscale = np.sqrt(cd21*cd21 + cd22*cd22)
    yangle = np.degrees(np.arctan2(cd12, cd22))
    xangle = np.degrees(np.arctan2(cd11, cd21))
    #yangle = np.degrees(np.arctan(cd12 / cd22))
    #xangle = np.degrees(np.arctan(cd11 / cd21))
    axskew = yangle - xangle - 90.0
    return xscale, yscale, yangle, axskew

ne_cdinfo =  np.array([analyze(x) for x in ne_pstack[:4].T])
nw_cdinfo =  np.array([analyze(x) for x in nw_pstack[:4].T])
se_cdinfo =  np.array([analyze(x) for x in se_pstack[:4].T])
sw_cdinfo =  np.array([analyze(x) for x in sw_pstack[:4].T])

##--------------------------------------------------------------------------##
## Reorder the CDxx according to various seasonal parameters ...
cd11 = par_stack[0,0,:]
cd22 = par_stack[0,3,:]

## Averages of everything:
rstats_runid_order = {ss:np.array([rstats[rr][ss][0] for rr in runid_list]) \
                                                        for ss in statcols}
## Order by fractional year:
j2k_epoch = 2451545.0
rst_jdtdb = rstats_runid_order['JDTDB']
rst_yfrac = (rst_jdtdb - j2k_epoch) % 365.25
yfr_order = np.argsort(rst_yfrac)
#cd11_yfr  = cd11[yfr_order]

## Order by azimuth:
rst_telaz = rstats_runid_order['TELAZ']
azm_order = np.argsort(rst_telaz)

## Order by altitude:
rst_altit = rstats_runid_order['TELALT']
alt_order = np.argsort(rst_altit)

## Order by parallactic angle:
rst_paran = rstats_runid_order['PARANG']
par_order = np.argsort(rst_paran)

## XYZ position and velocity vectors:
xyzpos = np.column_stack([rstats_runid_order[x] \
                        for x in ['OBS_X', 'OBS_Y', 'OBS_Z']])
xyzvel = np.column_stack([rstats_runid_order[x] \
                        for x in ['OBS_VX', 'OBS_VY', 'OBS_VZ']])
rtotal = np.sum(xyzpos**2, axis=1)**0.5
vtotal = np.sum(xyzvel**2, axis=1)**0.5
u_xyzvel = xyzvel / veclen(xyzvel.T)[:, None]

## What are the XYZ coordinates of the nominal pointing RA/Dec?
avg_ra_rad = np.radians(np.average(rstats_runid_order['CRVAL1']))
avg_de_rad = np.radians(np.average(rstats_runid_order['CRVAL2']))
calib1_xyz = np.squeeze(rade2xyz(avg_ra_rad, avg_de_rad))

## What is the angle between Earth's velocity and that pointing?
## Use theta = np.arccos(A dot B):
vdotcalib1 = np.degrees(np.arccos([np.dot(calib1_xyz, x) for x in u_xyzvel]))

## Order by vdotcalib1:
vel_order = np.argsort(vdotcalib1)

##--------------------------------------------------------------------------##
#_FIXED_LIMITS = True
_FIXED_LIMITS = False


#axs[4,0].plot(runid_list, par_stack[0,0,:])
#axs[4,1].plot(runid_list, par_stack[0,3,:])

figtrans = True
figtrans = False
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
#fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fg1, axs1 = plt.subplots(nrows=4, ncols=2, num=1, clear=True, figsize=fig_dims,
#fig, axs = plt.subplots(nrows=2, ncols=6, num=1, clear=True, figsize=fig_dims,
                        sharex=False, squeeze=True)

def fts(cc,rr):
    if figtrans:
        return rr,cc
    return cc,rr

for ax in axs1.ravel():
    ax.grid(True)

i = 0
axs1[fts(i,0)].scatter(rst_yfrac[yfr_order], cd11[yfr_order])
axs1[fts(i,0)].set_ylabel("CD11")
axs1[fts(i,0)].set_xlabel("Day of year")
axs1[fts(i,1)].scatter(rst_yfrac[yfr_order], cd22[yfr_order])
axs1[fts(i,1)].set_ylabel("CD22")
axs1[fts(i,1)].set_xlabel("Day of year")
#if _FIXED_LIMITS:
#    axs1[0,0].set_ylim(2183.5, 2184.15)
#    axs1[0,1].set_ylim(-8.9, -7.4)

i += 1
axs1[fts(i,0)].scatter(vdotcalib1[vel_order], cd11[vel_order])
axs1[fts(i,0)].set_ylabel("CD11")
axs1[fts(i,0)].set_xlabel("$v_E \cdot RA/DE$")
axs1[fts(i,1)].scatter(vdotcalib1[vel_order], cd22[vel_order])
axs1[fts(i,1)].set_ylabel("CD22")
axs1[fts(i,1)].set_xlabel("$v_E \cdot RA/DE$")
#if _FIXED_LIMITS:
#    axs1[1,0].set_ylim(2186.9, 2190.9)
#    axs1[1,1].set_ylim(-12.8, -10.6)

#i += 1
#axs1[fts(i,0)].scatter(rst_telaz[azm_order], cd11[azm_order])
#axs1[fts(i,0)].set_ylabel("CD11")
#axs1[fts(i,0)].set_xlabel("TELAZ (deg)")
#axs1[fts(i,1)].scatter(rst_telaz[azm_order], cd22[azm_order])
#axs1[fts(i,1)].set_ylabel("CD22")
#axs1[fts(i,1)].set_xlabel("TELAZ (deg)")
##if _FIXED_LIMITS:
##    axs1[2,0].set_ylim(-4.2, 1.2)
##    axs1[2,1].set_ylim(-2194.4, -2192.3)

#i += 1
#axs1[fts(i,0)].scatter(rst_altit[alt_order], cd11[alt_order])
#axs1[fts(i,0)].set_ylabel("CD11")
#axs1[fts(i,0)].set_xlabel("TELALT (deg)")
#axs1[fts(i,1)].scatter(rst_altit[alt_order], cd22[alt_order])
#axs1[fts(i,1)].set_ylabel("CD22")
#axs1[fts(i,1)].set_xlabel("TELALT (deg)")
##if _FIXED_LIMITS:
##    axs1[3,0].set_ylim(-4.2, 1.2)
##    axs1[3,1].set_ylim(-2194.4, -2192.3)

i += 1
axs1[fts(i,0)].scatter(rst_paran[par_order], cd11[par_order])
axs1[fts(i,0)].set_ylabel("CD11")
axs1[fts(i,0)].set_xlabel("PARANG (deg)")
axs1[fts(i,1)].scatter(rst_paran[par_order], cd22[par_order])
axs1[fts(i,1)].set_ylabel("CD22")
axs1[fts(i,1)].set_xlabel("PARANG (deg)")
#if _FIXED_LIMITS:
#    axs[3,0].set_ylim(-4.2, 1.2)
#    axs[3,1].set_ylim(-2194.4, -2192.3)

#axs[3,0].plot(runid_list, nw_sw_dx)
#axs[3,0].set_ylabel("CRPIX1 (NW - SW)")
#axs[3,1].plot(runid_list, nw_sw_dy)
#axs[3,1].set_ylabel("CRPIX2 (NW - SW)")
#if _FIXED_LIMITS:
#    axs[3,0].set_ylim(-4.2, 1.0)
#    axs[3,1].set_ylim(-2194.4, -2192.2)

axs1[fts(-1,0)].plot(runid_list, par_stack[0,0,:])
axs1[fts(-1,1)].plot(runid_list, par_stack[0,3,:])
if _FIXED_LIMITS:
    axs1[-1,0].set_ylim(-8.5105e-5, -8.5075e-5)
    axs1[-1,1].set_ylim(8.5015e-5, 8.5105e-5)

for label in axs1[fts(-1,0)].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8) 
for label in axs1[fts(-1,1)].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8) 

fg1.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
plot_name = 'seasonal_CD11_CD22.png'
fg1.savefig(plot_name, bbox_inches='tight')


##--------------------------------------------------------------------------##
##------------------         Sensor Orientations            ----------------##
##--------------------------------------------------------------------------##

fig_dims = (11, 9)
fg2, anaxs = plt.subplots(nrows=3, ncols=2, num=2, clear=True, 
                            figsize=fig_dims, sharex=True, squeeze=True)

info_arrays = [ne_cdinfo, nw_cdinfo, se_cdinfo, sw_cdinfo]
xs_arrays = np.array([info[:, 0] for info in info_arrays])
ys_arrays = np.array([info[:, 1] for info in info_arrays])
pa_arrays = np.array([info[:, 2] for info in info_arrays])
sk_arrays = np.array([info[:, 3] for info in info_arrays])
#pa_arrays = [ne_cdinfo[:, 2], nw_cdinfo[:, 2],
#             se_cdinfo[:, 2], sw_cdinfo[:, 2]]
#pa_arrays = [ne_cdinfo[:, 0], nw_cdinfo[:, 0],
#             se_cdinfo[:, 2], sw_cdinfo[:, 2]]

sensor_order = ['NE', 'NW', 'SE', 'SW']

mas_per_deg = 3.6e6
scale_lims = (306.1, 306.45)

## Plot the X-scales:
for ii,(xs,qq) in enumerate(zip(xs_arrays, sensor_order)):
    anaxs[0,0].plot(runid_list, mas_per_deg*xs, label=qq)
    anaxs[0,0].set_ylabel("X-scale")
    anaxs[0,0].grid(True)
    anaxs[0,0].set_ylim(*scale_lims)

## Plot the Y-scales:
for ii,(ys,qq) in enumerate(zip(ys_arrays, sensor_order)):
    anaxs[0,1].plot(runid_list, mas_per_deg*ys, label=qq)
    anaxs[0,1].set_ylabel("Y-scale")
    anaxs[0,1].grid(True)
    anaxs[0,1].set_ylim(*scale_lims)

## Plot the measured position angles:
for ii,(pa,qq) in enumerate(zip(pa_arrays, sensor_order)):
    anaxs[1,0].plot(runid_list, pa, label=qq)
    anaxs[1,0].set_ylabel("Y-axis PA [deg]")
    anaxs[1,0].grid(True)
    anaxs[1,0].set_ylim(-0.35, 1.05)
    anaxs[1,0].legend(loc='upper left')

## Plot the measured axis skews:
for ii,(sk,qq) in enumerate(zip(sk_arrays, sensor_order)):
    anaxs[1,1].plot(runid_list, sk, label=qq)
    anaxs[1,1].set_ylabel("X-Y skew")
    anaxs[1,1].grid(True)

## Plot the measured axis skews:
for ii,(sr,qq) in enumerate(zip(xs_arrays/ys_arrays-1, sensor_order)):
    anaxs[2,0].plot(runid_list, sr, label=qq)
    anaxs[2,0].set_ylabel("(X-scale / Y-scale) - 1")
    anaxs[2,0].grid(True)

## Fix QRUNID labels:
for label in anaxs[-1,0].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8)
for label in anaxs[-1,1].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8)

fg2.align_ylabels()
fg2.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
plot_name = 'scale_PA_skew_by_QRUNID.png'
fg2.savefig(plot_name, bbox_inches='tight')

##--------------------------------------------------------------------------##
##------------------         Some Other Stuff Later         ----------------##
##--------------------------------------------------------------------------##



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
# CHANGELOG (32_params_vs_season.py):
#---------------------------------------------------------------------
#
#  2026-05-12:
#     -- Increased __version__ to 0.1.0.
#     -- First created 32_params_vs_season.py.
#

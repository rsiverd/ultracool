#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Evaluate the stability of various joint fit parameters.
#
# Rob Siverd
# Created:       2026-03-03
# Last modified: 2026-04-02
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
import glob
#import io
import gc
import os
import ast
import sys
import time
import pprint
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
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
np.set_printoptions(suppress=True, linewidth=160)
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
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
#fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig, axs = plt.subplots(nrows=5, ncols=2, num=1, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=True)

for ax in axs.ravel():
    ax.grid(True)

axs[0,0].plot(runid_list, ne_nw_dx)
axs[0,0].set_ylabel("CRPIX1 (NE - NW)")
axs[0,1].plot(runid_list, ne_nw_dy)
axs[0,1].set_ylabel("CRPIX2 (NE - NW)")

axs[1,0].plot(runid_list, ne_se_dx)
axs[1,0].set_ylabel("CRPIX1 (NE - SE)")
axs[1,1].plot(runid_list, ne_se_dy)
axs[1,1].set_ylabel("CRPIX2 (NE - SE)")


axs[2,0].plot(runid_list, se_sw_dx)
axs[2,0].set_ylabel("CRPIX1 (SE - SW)")
axs[2,1].plot(runid_list, se_sw_dy)
axs[2,1].set_ylabel("CRPIX2 (SE - SW)")

axs[3,0].plot(runid_list, nw_sw_dx)
axs[3,0].set_ylabel("CRPIX1 (NW - SW)")
axs[3,1].plot(runid_list, nw_sw_dy)
axs[3,1].set_ylabel("CRPIX2 (NW - SW)")

axs[4,0].plot(runid_list, par_stack[0,0,:])
axs[4,1].plot(runid_list, par_stack[0,3,:])

for label in axs[-1,0].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8) 
for label in axs[-1,1].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8) 

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
plot_name = 'delta_CRPIX_vs_QRUNID.png'
fig.savefig(plot_name, bbox_inches='tight')

##--------------------------------------------------------------------------##
##------------------         Sensor Orientations            ----------------##
##--------------------------------------------------------------------------##

fig_dims = (11, 9)
pafig, paaxs = plt.subplots(nrows=4, ncols=2, num=2, clear=True, figsize=fig_dims,
                        sharex=True, squeeze=True)

pa_arrays = [ne_cdinfo[:, 2], nw_cdinfo[:, 2], se_cdinfo[:, 2], sw_cdinfo[:, 2]]
pa_naming = ['NE', 'NW', 'SE', 'SW']

## Plot the measured position angles:
for ii,(pa,qq) in enumerate(zip(pa_arrays, pa_naming)):
    paaxs[ii,0].plot(runid_list, pa)
    paaxs[ii,0].set_ylabel("PA(%s) [deg]" % qq)
    paaxs[ii,0].grid(True)

## Plot PA differences relative to NE:
for ii,(pa,qq) in enumerate(zip(pa_arrays, pa_naming)):
    pa_diff = pa - pa_arrays[0]
    paaxs[ii,1].plot(runid_list, pa_diff)
    paaxs[ii,1].set_ylabel("PA(%s) - PA(NE) [deg]" % qq)
    paaxs[ii,1].grid(True)

#paaxs[0,0].plot(runid_list, ne_cdinfo[:, 2]) # position angle
#paaxs[0,0].set_ylabel("NE PA [deg]")
#
#paaxs[1,0].plot(runid_list, nw_cdinfo[:, 2]) # position angle
#paaxs[1,0].set_ylabel("NW PA [deg]")
#
#paaxs[2,0].plot(runid_list, se_cdinfo[:, 2]) # position angle
#paaxs[2,0].set_ylabel("SE PA [deg]")
#
#paaxs[3,0].plot(runid_list, sw_cdinfo[:, 2]) # position angle
#paaxs[3,0].set_ylabel("SW PA [deg]")

## PA differences:
#for ii,pa2 in enumerate([ne_cdinfo[:, 2], nw_cdinfo[:, 2],
#                         se_cdinfo[:, 2], sw_cdinfo[:, 2]):
#    delta_PA = 
#paaxs[0,1].plot(runid_list, ne_cdinfo[:, 2] - )
#paaxs[0,1].set_ylabel("CRPIX2 (NE - NW)")

for label in paaxs[-1,0].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8)
for label in paaxs[-1,1].get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(8)

pafig.align_ylabels()
pafig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
plot_name = 'PA_vs_QRUNID.png'
pafig.savefig(plot_name, bbox_inches='tight')

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
# CHANGELOG (31_params_over_time.py):
#---------------------------------------------------------------------
#
#  2026-03-03:
#     -- Increased __version__ to 0.1.0.
#     -- First created 31_params_over_time.py.
#

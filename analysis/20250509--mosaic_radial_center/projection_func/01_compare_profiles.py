#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Compare the radial distortion profile of CFHT to that of a parabolic mirror.
#
# Rob Siverd
# Created:       2025-10-28
# Last modified: 2025-10-28
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

### Python version-agnostic module reloading (cute, 2.7+?):
#import sys
#reload = sys.modules['imp' if 'imp' in sys.modules else 'importlib'].reload

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
import scipy.optimize as opti
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
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import PIL.Image as pli
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
## Config:
_RMAX = 3000.       # maximum radius (bound for chebyshev)

##--------------------------------------------------------------------------##
## Load parabola data:
parab_profile  = 'parab_profiles.csv'
pdata = pd.read_csv(parab_profile, low_memory=False)

pscale = 0.306
offax_mult = 3600.0 / pscale    # guess
#disto_mult = 
pdata['pixsep'] = pdata['offax_deg'] * offax_mult
#pdata['pixerr'] = 

##--------------------------------------------------------------------------##
## Load CFHT diags:
cfht_diag_file = 'best_fit_diags.csv'
all_ddata = pd.read_csv(cfht_diag_file, low_memory=False)
ddata = all_ddata[~all_ddata.masked]

## R-sep and R-shift:
every_rsep = ddata['rdist']
every_rerr = np.hypot(ddata['xerror'] - ddata['xnudge'],
                      ddata['yerror'] - ddata['ynudge'])

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Polynomial helpers:
def poly_eval(r, model):
    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r
    return model[0] + model[1]*r + model[2]*r*r \
            + model[3]*r*r*r + model[4]*r*r*r*r
    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r \
    #        + model[4]*r*r*r*r + model[5]*r*r*r*r*r

#def dumb_poly_eval(r, c0, c1, c2, c3, c4, c5):
#    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))))
#badguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

## 4th order in R, c0 floating:
def dumb_poly_eval(r, c0, c1, c2, c3, c4):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r*c4)))
badguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 

### 4th order in R, c0 fixed to 0.0:
#def dumb_poly_eval(r, c1, c2, c3, c4):
#    return 0.0 + r * (c1 + r * (c2 + r * (c3 + r*c4)))
#badguess = np.array([0.0, 0.0, 0.0, 0.0]) 

## 5th order in R, c0 floating:
def dumb_poly_eval(r, c0, c1, c2, c3, c4, c5):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r*c5))))
badguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

### 5th order in R, c0 fixed to 0.0:
#def dumb_poly_eval(r, c1, c2, c3, c4, c5):
#    return 0.0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r*c5))))
#badguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 

### 6th order in R, c0 floating:
#def dumb_poly_eval(r, c0, c1, c2, c3, c4, c5, c6):
#    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r*c6)))))
#badguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

#def dumb_poly_eval(r, c0, c1, c2, c3, c4, c5, c6):
#    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r))))))
#badguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 


#def dumb_poly_eval(r, c0, c1, c2, c3):
#    return c0 + r * (c1 + r * (c2 + r * c3))
#    #return c0 + r * (c1 + r * c2)
#badguess = np.array([0.0, 0.0, 0.0, 0.0]) 
#
#def dumb_poly_eval(r, c0, c1, c2):
#    return c0 + r * (c1 + r * c2)
#badguess = np.array([0.0, 0.0, 0.0]) 

#def cheby_eval(x, degree, 

def cheb_eval(x, c0, c1, c2, c3, c4):
#def cheb_eval(x, c0, c1, c2, c3):
#def cheb_eval(x, c0, c1, c2):
    T0 = 1.0
    T1 = x
    T2 = 2 * x * T1 - T0
    T3 = 2 * x * T2 - T1
    T4 = 2 * x * T3 - T2
    return c0 + c1 * T1 + c2 * T2 + c3 * T3 + c4 * T4
    #return c0 + c1 * T1 + c2 * T2 + c3 * T3
    #return c0 + c1 * T1 + c2 * T2


## Fit the function as a polynomial:
#badguess = None
#badguess = np.array([0.0, 0.00012424, 0.00000268, 0.0, 0.0])
#bestpar, bestcov = opti.curve_fit(dumb_poly_eval, every_rsep, every_rerr)
fracrad = every_rsep / _RMAX
bestpar, bestcov = opti.curve_fit(dumb_poly_eval, 
                                  every_rsep, every_rerr, badguess)

chebpar, chebcov = opti.curve_fit(cheb_eval, every_rsep, every_rerr)

## Illustrate the best fit:
showme_x = np.linspace(1.0, 3000.)
showme_y = dumb_poly_eval(showme_x, *bestpar)
baddie_y = dumb_poly_eval(showme_x, *badguess)
residuals = every_rerr - dumb_poly_eval(every_rsep, *bestpar)

showch_y = cheb_eval(showme_x, *chebpar)
cheby_res = every_rerr - cheb_eval(every_rsep, *chebpar)

## Print best fit to terminal:
print('\n'.join(['%15e'%x for x in bestpar]))

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
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1)
ax3 = fig.add_subplot(223, sharex=ax1)
ax4 = fig.add_subplot(224, sharex=ax1)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
axs = [ax1, ax2, ax3, ax4]
for ax in axs:
    ax.grid(True)
#ax1.grid(True)
#ax2.grid(True)
#ax1.axis('off')

## Polar scatter:
skw = {'lw':0, 's':15}
#ax1.scatter(azm_rad, zdist_deg, **skw)
#ax1.scatter(every_rsep, every_rerr, **skw)
## 17 @ 2500
#dummyx = np.linspace(0, 2800.)
#dummy = pdata['pixsep'] * 23. / 2800.
#
#mult = 1.0
#mult = 1400.
#ax1.scatter(pdata['pixsep'], mult*pdata['planex'], **skw)
##ax1.plot(dummyx, dummyy)
##ax2.plot(pdata['pixsep'], dummy - mult*pdata['planex'])
##ax2.plot(pdata['pixsep'], pdata['focusx'])
##ax1.scatter(pdata['pixsep'], 1.0/pdata['planex'], **skw)
##ax2.scatter(pdata['pixsep'], pdata['focusx'])
ax1.scatter(every_rsep, every_rerr, **skw)
ax1.plot(showme_x, showme_y, c='r')
ax3.scatter(every_rsep, residuals, **skw)

ax2.scatter(every_rsep, every_rerr, **skw)
ax2.plot(showme_x, showch_y, c='r')
ax4.scatter(every_rsep, cheby_res, **skw)

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

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')


######################################################################
# CHANGELOG (01_compare_profiles.py):
#---------------------------------------------------------------------
#
#  2025-10-28:
#     -- Increased __version__ to 0.0.1.
#     -- First created 01_compare_profiles.py.
#

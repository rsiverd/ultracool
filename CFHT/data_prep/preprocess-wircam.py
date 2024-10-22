#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This script selects a specific (user-specified) WIRCam sensor from a
# processed image and cleans up that image in several ways to ensure 
# good performance in this pipeline. Tasks performed include:
# * select a specific WIRCam quadrant [NE, SE, NW, SW]
# * set saturated pixels to a specific high value
# * interpolate sensible values for certain point defects
# * stack data cubes to produce single useful image
# * mask out image sections affected by guider operation
# * optionally storing a mask/uncertainty image for later use
# 
# This step is necessary to compensate for significant changes in the
# WIRCam pipeline that affect data newer than ~2015B.
# 
# Required inputs are:
# * raw WIRCam image
# * processed WIRCam image
#
# Outputs produced:
# * modified processed 'p' image with uniform properties
#
# Rob Siverd
# Created:       2024-10-14
# Last modified: 2024-10-14
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
#import matplotlib.pyplot as plt
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

## WIRCam config:
import wircam_config
reload(wircam_config)
wcfg = wircam_config

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
    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
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
##------------------         WIRCam Quadrant Names          ----------------##
##--------------------------------------------------------------------------##

quad_exts = {
        'NW'    :   (1, 'HAWAII-2RG-#77'),
        'SW'    :   (2, 'HAWAII-2RG-#52'),
        'SE'    :   (3, 'HAWAII-2RG-#54'),
        'NE'    :   (4, 'HAWAII-2RG-#60'),
}


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
    Clean up WIRCam images before running extraction and astrometry.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    parser.set_defaults(quadrant='NE')
    parser.set_defaults(saturval=65535)
    parser.set_defaults(mthresh=0.9)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('-d', '--dayshift', required=False, default=0,
    #        help='Switch between days (1=tom, 0=today, -1=yest', type=int)
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    quadgroup = parser.add_argument_group('Quadrant Choice')
    quadgroup.add_argument('--NE', dest='quadrant', required=False,
            action='store_const', const='NE',
            help='extract and preprocess NE quadrant [default]')
    quadgroup.add_argument('--NW', dest='quadrant', required=False,
            action='store_const', const='NW',
            help='extract and preprocess NW quadrant')
    quadgroup.add_argument('--SE', dest='quadrant', required=False,
            action='store_const', const='SE',
            help='extract and preprocess SE quadrant')
    quadgroup.add_argument('--SW', dest='quadrant', required=False,
            action='store_const', const='SW',
            help='extract and preprocess SW quadrant')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    procgroup = parser.add_argument_group('Processing Options')
    # FIXME: add compression enable/disable here
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('--proc', default=None, required=True, type=str,
            help="Processed input image (file type 'p')")
    iogroup.add_argument('--raw', default=None, required=True, type=str,
            help="Raw input image (file type 'o')")
    iogroup.add_argument('--mask', default=None, required=True,
            help='Bad pixel mask', type=str)
    iogroup.add_argument('-o', '--output_file', default=None, required=True,
            help='Output image filename', type=str)
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
##--------------------------------------------------------------------------##

## Choose extension name based on quadrant:
sys.stderr.write("Chosen quadrant: %s\n" % context.quadrant)
qnum, use_extname = wcfg.quad_exts.get(context.quadrant)
sys.stderr.write("Using extension: %s\n" % use_extname)

## Load raw 'o' input image:
odata, okeys = pf.getdata(context.raw, header=True, extname=use_extname)

## Load processed 'p' input image:
pdata, pkeys = pf.getdata(context.proc, header=True, extname=use_extname)
median_value = np.median(pdata)  # usually about 7000

## Load bad pixel mask:
mdata = pf.getdata(context.mask)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Guide box size comes from header:
gbox_xsize_key = 'WCSIZEX'
gbox_ysize_key = 'WCSIZEY'
gbox_xsize_pix = pkeys[gbox_xsize_key]
gbox_ysize_pix = pkeys[gbox_ysize_key]

## Guide box header keywords (from sensor number):
gbox_ll_x_key = 'WCPOSX%d' % qnum
gbox_ll_y_key = 'WCPOSY%d' % qnum
gbox_ll_x_pix = pkeys[gbox_ll_x_key]
gbox_ll_y_pix = pkeys[gbox_ll_y_key]

## Transfer saturated pixels from raw to proc:
is_saturated = (odata > 65534)
pdata[is_saturated] = context.saturval

## FIXME: add interpolation for isolated bad pixels:
is_masked = (mdata > context.mthresh)
pdata[is_masked] = median_value

## Columns and rows masked as they would be in the CFH/WIRCam pipeline:
#gcols = range(gbox_ll_x_pix, gbox_ll_x_pix + gbox_xsize_pix)   # match gbox
#grows = range(gbox_ll_y_pix, gbox_ll_y_pix + gbox_ysize_pix)   # offset 1

## Columns and rows that match the black guider box:
#gcols = range(gbox_ll_x_pix, gbox_ll_x_pix + gbox_xsize_pix)
#grows = range(gbox_ll_y_pix - 1, gbox_ll_y_pix + gbox_ysize_pix)

## Cols/rows that cover guide box AND WIRCam pipeline pixels:
#gcols = range(gbox_ll_x_pix, gbox_ll_x_pix + gbox_xsize_pix)
#grows = range(gbox_ll_y_pix - 1, gbox_ll_y_pix + gbox_ysize_pix)

## Cols/rows that cover guide box AND WIRCam pipeline pixels with buffer:
gcols = range(gbox_ll_x_pix - 1, gbox_ll_x_pix + gbox_xsize_pix + 1)
grows = range(gbox_ll_y_pix - 2, gbox_ll_y_pix + gbox_ysize_pix + 1)

## Replace pixels in rows/cols near guide box:
pdata[grows, :] = median_value      # wipe guide rows
pdata[:, gcols] = median_value      # wipe guide cols

# in test image, 
# * row Y=555 to Y=573, inclusive (pixel coords)
# * col X=747 to X=762, inclusive (pixel coords)
gmsk_ylo = grows[ 0] + 1
gmsk_yhi = grows[-1] + 1
gmsk_xlo = gcols[ 0] + 1
gmsk_xhi = gcols[-1] + 1

## Record the rows and columns that were wiped:
pkeys['GMSK_YLO'] = (gmsk_ylo, 'lower row in custom guide mask (bottom)')
pkeys['GMSK_YHI'] = (gmsk_yhi, 'upper row in custom guide mask (top)')
pkeys['GMSK_XLO'] = (gmsk_xlo, 'lower col in custom guide mask (left)')
pkeys['GMSK_XHI'] = (gmsk_xhi, 'upper col in custom guide mask (right)')

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Save result:
qsave(context.output_file, pdata, header=pkeys)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Some debugging assistance:
if context.debug:
    sys.stderr.write("\n%s\n" % fulldiv)
    sys.stderr.write("DEBUGGING COMMANDS\n\n")
    r_base = os.path.basename(context.raw).split('.fz')[0]
    p_base = os.path.basename(context.proc).split('.fz')[0]
    r_save = os.path.join('/tmp', r_base)
    p_save = os.path.join('/tmp', p_base)
    sys.stderr.write("fitsarith -qHi '%s[%s]' -o %s\n"
            % (context.raw,  use_extname, r_save))
    sys.stderr.write("fitsarith -qHi '%s[%s]' -o %s\n"
            % (context.proc, use_extname, p_save))
    # ds9 view commands:
    sys.stderr.write("flztfs %s %s %s\n" % (r_save, p_save, context.output_file))
    sys.stderr.write("\n%s\n" % fulldiv)


######################################################################
# CHANGELOG (preprocess-wircam.py):
#---------------------------------------------------------------------
#
#  2024-10-14:
#     -- Increased __version__ to 0.1.0.
#     -- First created preprocess-wircam.py.
#

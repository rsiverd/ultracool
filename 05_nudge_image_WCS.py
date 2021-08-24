#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Apply catalog offsets to FITS images to simplify visual inspection and
# subsequent processing. This program creates the 'nudge' imtype from 'clean'
# files and their corresponding 'clean' fcat catalogs.
#
# Rob Siverd
# Created:       2021-08-24
# Last modified: 2021-08-24
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
#import gc
import os
import sys
import time
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

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
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
## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers

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
    Propagate Gaia-derived RA/Dec 'nudges' into image WCS.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(imtype='clean') #'cbcd') #'clean')
    parser.set_defaults(gwtype='nudge') # image flavor for results
    parser.set_defaults(cat_type='fcat')
    #parser.set_defaults(thing1='value1', thing2='value2')
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('--overwrite', required=False, dest='skip_existing',
            action='store_false', help='overwrite existing catalogs')
    #iogroup.add_argument('-E', '--ephem_data', default=None, required=True,
    #        help='CSV file with SST ephemeris data', type=str)
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find CBCD images')
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
##------------------         Make Input Image List          ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Listing %s frames ... " % context.imtype)

if context.walk:
    img_files = sfh.get_files_walk(context.input_folder, flavor=context.imtype)
else:
    img_files = sfh.get_files_single(context.input_folder, flavor=context.imtype)
sys.stderr.write("done.\n")

## Abort in case of no input:
if not img_files:
    sys.stderr.write("No input (%s) files found in folder:\n" % context.imtype)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

n_images = len(img_files)


##--------------------------------------------------------------------------##
##------------------        Check for Matching fcats        ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Checking for corresponding fcat catalogs ... ")
for img_ipath in img_files:
    cat_fpath = img_ipath + '.' + context.cat_type
    if not os.path.isfile(cat_fpath):
        sys.stderr.write("failure!\n\nCatalog not found:\n")
        sys.stderr.write("--> %s\n" % cat_fpath)
        sys.exit(1)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------        Nudge Individual Images         ----------------##
##--------------------------------------------------------------------------##

## Required nudge keywords:
_nudge_keys = ['GRADELTA', 'GRASIGMA', 'GDEDELTA', 'GDESIGMA']

ntodo = 0
nproc = 0
for img_ipath in img_files:
    sys.stderr.write("%s\n" % fulldiv)
    ibase = os.path.basename(img_ipath)
    cat_fpath = img_ipath + '.' + context.cat_type
    if not os.path.isfile(cat_fpath):
        sys.stderr.write("\nFile not found (should not see this):\n")
        sys.stderr.write("--> %s\n" % cat_fpath)
    img_npath = img_ipath.replace(context.imtype, context.gwtype)
    sys.stderr.write("Nudged image %s ... " % img_npath)

    # skip existing unless --overwrite requested:
    if context.skip_existing:
        if os.path.isfile(img_npath):
            sys.stderr.write("exists!  Skipping ... \n")
            continue
        sys.stderr.write("not found ... creating ...\n")
    else:
        sys.stderr.write("creating ...\n")
    nproc += 1

    # retrieve nudge parameters from extended catalog:
    chdrs = pf.getheader(cat_fpath, extname='CATALOG')
    if not all([x in chdrs for x in _nudge_keys]):
        sys.stderr.write("Yikes! Unexpected header contents in catalog:\n")
        sys.stderr.write("--> %s\n" % cat_fpath)
        sys.exit(1)
    ra_nudge = chdrs['GRADELTA']
    de_nudge = chdrs['GDEDELTA']

    # load original image:
    idata, ihdrs = pf.getdata(img_ipath, header=True)

    # duplicate headers and nudge CRVALs:
    nhdrs = ihdrs.copy()
    nhdrs['CRVAL1'] += ra_nudge
    nhdrs['CRVAL2'] += de_nudge

    # also propagate the nudge keys:
    for kk in _nudge_keys:
        nhdrs[kk] = chdrs[kk]

    # save nudged image:
    qsave(img_npath, idata, header=nhdrs)

    # stop early upon request:
    if (ntodo > 0) and (nproc >= ntodo):
        break

#--------------------------------------------------------------------------##


######################################################################
# CHANGELOG (05_nudge_image_WCS.py):
#---------------------------------------------------------------------
#
#  2021-08-24:
#     -- Increased __version__ to 0.1.0.
#     -- First created 05_nudge_image_WCS.py.
#

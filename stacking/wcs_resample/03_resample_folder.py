#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Resample a folder of Spitzer data onto a matched grid for stacking.
#
# Rob Siverd
# Created:       2019-10-23
# Last modified: 2019-10-23
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
import signal
import glob
import os
import sys
import time
import numpy as np
import random
#from numpy.lib.recfunctions import append_fields
from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Easy interpolation:
from reproject import reproject_interp
from reproject import reproject_exact

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

## Various from astropy:
try:
#    import astropy.io.ascii as aia
    import astropy.io.fits as pf
#    import astropy.table as apt
#    import astropy.time as astt
    import astropy.wcs as awcs
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
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
## Catch interruption cleanly:
def signal_handler(signum, frame):
    sys.stderr.write("\nInterrupted!\n\n")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

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
    Resample a folder of images onto common RA/DE grid.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(clobber=False, exact=False)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    prgroup = parser.add_argument_group('Processing Options')
    prgroup.add_argument('-E', '--exact', required=False, action='store_true',
            help='exact reprojection (spherical polygon intersection)')
    prgroup.add_argument('-n', '--ntodo', required=False, default=0,
            help='stop processing after N objects', type=int)
    prgroup.add_argument('-r', '--random', default=False, required=False,
            help='process images in random order', action='store_true')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-i', '--input_folder', default=None, required=True,
            help='folder with input images', type=str)
    iogroup.add_argument('-o', '--output_folder', default=None, required=True,
            help='folder for output images', type=str)
    iogroup.add_argument('-R', '--ref_image', default=None, required=True,
            help='image with WCS grid to use')
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
##--------------------------------------------------------------------------##

## Check and load reference image with WCS:
sys.stderr.write("Loading WCS grid ... ")
if not os.path.isfile(context.ref_image):
    sys.stderr.write("\nError: WCS grid image not found: '%s'\n"
            % context.ref_image)
    sys.exit(1)
rdata, rkeys = pf.getdata(context.ref_image, header=True)
grwcs = awcs.WCS(rkeys)
sys.stderr.write("done.\n")

## Ensure input folder exists:
if not os.path.isdir(context.input_folder):
    sys.stderr.write("Error: input folder not found: '%s'\n"
            % context.input_folder)
    sys.exit(1)

## Create output folder if needed:
if not os.path.isdir(context.output_folder):
    sys.stderr.write("Output folder not found ... creating.\n")
    os.mkdir(context.output_folder)


## List of input files:
image_list = sorted(glob.glob("%s/SPITZER*cbcd.fits" % context.input_folder))
if context.random:
    random.shuffle(image_list)

## Summary:
sys.stderr.write("Input folder:  %s\n" % context.input_folder)
sys.stderr.write("Output folder: %s\n" % context.output_folder)
sys.stderr.write("Images found:  %d\n" % len(image_list))

##--------------------------------------------------------------------------##
##------------------    Single-Image Resamling Routine      ----------------##
##--------------------------------------------------------------------------##

#sys.exit(0)
def grid_resample(src_path, new_wcs, exact=False, **kwargs):
    with pf.open(src_path) as hdulist:
        if (len(hdulist) > 1):
            sys.stderr.write("Unexpected HDU count: %d\n" % len(hdulist))
            raise
        src_hdu = hdulist[0]
        new_hdr = hdulist[0].header.copy(strip=True)[:120]  # trash old WCS
        if exact:
            new_img, footprint = reproject_exact(src_hdu, new_wcs,
                    shape_out=new_wcs.pixel_shape, parallel=4)
        else:
            new_img, footprint = reproject_interp(src_hdu, new_wcs,
                    shape_out=new_wcs.pixel_shape, order='bilinear')
                    #shape_out=new_wcs.pixel_shape, order='biquadratic')
                    #shape_out=new_wcs.pixel_shape, order='bicubic')
        new_hdr.update(new_wcs.to_header())
    return new_img, new_hdr, footprint

## resample with current options:
my_resampler = partial(grid_resample, new_wcs=grwcs, exact=context.exact)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Process everything:
sys.stderr.write("\nStarting resampling task ...\n")
#ntodo = 5
nproc = 0
total = len(image_list)
for ii,ipath in enumerate(image_list, 1):
    ibase = os.path.basename(ipath)
    sys.stderr.write("\rResampling %s ... " % ipath)
    opath = os.path.join(context.output_folder, ibase)
    if (not context.clobber) and os.path.isfile(opath):
        sys.stderr.write("already done!  ")
        continue
    sys.stderr.write("needs work ...      \n")
    nproc += 1

    sys.stderr.write("Resampling image ... ")
    tik = time.time()
    pix, hdrs, footprint = my_resampler(ipath)
    #pix, hdrs, footprint = grid_resample(ipath, grwcs, exact=context.exact)
    #pix, hdrs, footprint = grid_resample(ipath, grwcs, exact=False)
    tok = time.time()
    sys.stderr.write("done. (%.3f s)\n" % (tok-tik)) 

    sys.stderr.write("Saving result ... ")
    pf.writeto(opath, pix, header=hdrs, overwrite=True)
    sys.stderr.write("done.\n")

    # Early quit for debug:
    if (context.ntodo > 0) and (nproc >= context.ntodo):
        break

sys.stderr.write("\nProcessing complete!\n")

##--------------------------------------------------------------------------##
##------------------       Multi-Threaded Processing        ----------------##
##--------------------------------------------------------------------------##

### Assemble paired input/output paths:
#def dirswap(path, newdir):
#    return os.path.join(newdir, os.path.basename(path))
#
#io_pairs = [(x, dirswap(x, context.output_folder)
#
### Pre-build list of output images:
#io_pairs = [(x, os.path.join(context.output_folder, os.path.basename(x))) \
#                    for x in image_list]
#
### First, identify remaining images:
#remaining_images = []






######################################################################
# CHANGELOG (03_resample_folder.py):
#---------------------------------------------------------------------
#
#  2019-10-23:
#     -- Increased __version__ to 0.1.0.
#     -- First created 03_resample_folder.py.
#

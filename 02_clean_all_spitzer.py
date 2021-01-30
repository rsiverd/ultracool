#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Clean CBCD Spitzer images by removing cosmic rays and/or removing
# large-scale background.
#
# NOTE: This script assumes a directory structure as created by the
# related fetch_sha_data.py script. Specifically, the images to be
# processed are expected to reside in a structure like:
# object_dir/r<AOR_number>/SPITZER*_cbcd.fits
#
# The object_dir contains data for a specific target/sky position to
# be reduced. <AOR_number> represents a Spitzer AOR (a visit to a sky
# position at which data were obtained).
#
# Rob Siverd
# Created:       2019-10-30
# Last modified: 2021-01-29
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
__version__ = "0.2.0"

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
import random
#import resource
import signal
import glob
#import gc
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## LACOSMIC cosmic ray removal:
try:
    from lacosmic import lacosmic
except ImportError:
    logger.error("failed to import lacosmic module!")
    sys.exit(1)

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

### FITSIO module (provides compression ability):
#try:
#    import fitsio
#except ImportError:
#    logger.error("fitsio module not found!  Install and retry.")
#    sys.stderr.write("Error: fitsio module not found!\n\n")
#    sys.exit(1)

## Star extraction:
try:
    import easy_sep
    reload(easy_sep)
except ImportError:
    logger.error("easy_sep module not found!  Install and retry.")
    sys.stderr.write("Error: easy_sep module not found!\n\n")
    sys.exit(1)
pse = easy_sep.EasySEP()

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
    Clean Spitzer images by removing cosmic rays and/or large-scale
    background light.

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
    iogroup.add_argument('-I', '--image_folder', default=None, required=True,
            help='where to find CBCD images', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find CBCD images')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    jobgroup = parser.add_argument_group('Processing Options')
    jobgroup.add_argument('-r', '--random', default=False, action='store_true',
            help='randomize image processing order\n(for parallel processing)',
            required=False)
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
##------------------      Input Image List Generators       ----------------##
##--------------------------------------------------------------------------##

def get_cbcd_single_folder(dirpath, suffix):
    im_wildpath = '%s/SPITZ*%s' % (dirpath, suffix)
    return sorted(glob.glob(im_wildpath))

def get_cbcd_recursive_walk(targ_root, suffix):
    image_list = []
    for thisdir, subdirs, files in os.walk(targ_root):
        cbcd_files = [x for x in files if x.endswith(suffix)]
        image_list += [os.path.join(thisdir, x) for x in cbcd_files]
    return sorted(image_list)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Get list of CBCD files:
#im_wildpath = '%s/SPITZ*_cbcd.fits' % context.image_folder
#cbcd_files = sorted(glob.glob(im_wildpath))
if context.walk:
    cbcd_files = get_cbcd_recursive_walk(context.image_folder, '_cbcd.fits')
else:
    cbcd_files = get_cbcd_single_folder(context.image_folder)

## Randomize image order on request (for parallel processing):
if context.random:
    random.shuffle(cbcd_files)  # for parallel operation

## Abort with warning if no files identified:
if not cbcd_files:
    sys.stderr.write("\nError: no cbcd files found in specified location:\n")
    sys.stderr.write("--> %s\n\n" % context.image_folder)
    sys.exit(1)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## LA Cosmic config:
def fresh_cr_args():
    return {'contrast': 12.0,
            'cr_threshold':6.0,
            'neighbor_threshold':4.0,}

## Clean up each image:
ntodo = 0
nproc = 0
total = len(cbcd_files)
for ii,img_ipath in enumerate(cbcd_files, 1):
    #sys.stderr.write("%s\n" % fulldiv)
    unc_ipath = img_ipath.replace('cbcd', 'cbunc')
    vst_ipath = img_ipath.replace('cbcd',  'vmed')
    hcf_ipath = img_ipath.replace('cbcd', 'hcfix')
    cln_ipath = img_ipath.replace('cbcd', 'clean')
    msk_ipath = img_ipath.replace('cbcd', 'crmsk')
    sys.stderr.write("\rFile %s (%d of %d) ... " % (cln_ipath, ii, total))
    done_list = [vst_ipath, cln_ipath, msk_ipath]
    if all([os.path.isfile(x) for x in done_list]):
        sys.stderr.write("already done!   ")
        continue
    sys.stderr.write("not found, processing ... \n")
    nproc += 1

    # load data:
    idata, ihdrs = pf.getdata(img_ipath, header=True)
    udata, uhdrs = pf.getdata(unc_ipath, header=True)
    #fdata, fhdrs = fitsio.read(img_ipath, header=True)

    # get median image value:
    ignore = np.isnan(idata) | np.isinf(idata)
    medval = np.median(idata[~ignore])

    # vertical median to fix hot columns:
    itemp  = idata.copy()
    itemp[ignore] = medval
    vstack = np.median(itemp, axis=0)
    qsave(vst_ipath, vstack)
    idata -= vstack[np.newaxis, :]
    qsave(hcf_ipath, idata, header=ihdrs)

    # CR removal:
    lakw = fresh_cr_args()
    lakw['mask'] = np.isnan(idata)
    lakw['error'] = udata
    sys.stderr.write("Running LACOSMIC ... ")
    tik = time.time()
    cleaned, cr_mask = lacosmic(idata, **lakw)
    tok = time.time()
    sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
    
    # save results:
    qsave(cln_ipath, cleaned, header=ihdrs)
    qsave(msk_ipath, cr_mask.astype('uint8'), header=ihdrs)
    #fitsio.write(msk_ipath, cr_mask.astype('uint8'), header=fhdrs, 
    #        clobber=True, compress='RICE')
    sys.stderr.write("%s\n" % fulldiv)

    if (ntodo > 0) and (nproc >= ntodo):
        break
    
sys.stderr.write("\n")






######################################################################
# CHANGELOG (02_clean_all_spitzer.py):
#---------------------------------------------------------------------
#
#  2019-10-30:
#     -- Increased __version__ to 0.1.0.
#     -- First created 02_clean_all_spitzer.py.
#

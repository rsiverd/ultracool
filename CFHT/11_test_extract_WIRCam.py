#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract and save extended object catalogs from the specified data and
# uncertainty images.
#
# Rob Siverd
# Created:       2023-05-31
# Last modified: 2023-05-31
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
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
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
### Spitzer pipeline filesystem helpers:
#try:
#    import spitz_fs_helpers
#    reload(spitz_fs_helpers)
#except ImportError:
#    logger.error("failed to import spitz_fs_helpers module!")
#    sys.exit(1)
#sfh = spitz_fs_helpers

## Spitzer pipeline filesystem helpers:
try:
    import wircam_fs_helpers
    reload(wircam_fs_helpers)
except ImportError:
    logger.error("failed to import wircam_fs_helpers module!")
    sys.exit(1)
wfh = wircam_fs_helpers

## Spitzer star detection routine:
try:
    import spitz_extract
    reload(spitz_extract)
    spf = spitz_extract.SpitzFind()
except ImportError:
    logger.error("spitz_extract module not found!")
    sys.exit(1)


##--------------------------------------------------------------------------##
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

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
    Extract catalogs from the listed Spitzer data/uncertainty images.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt)
                          #formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(imtype=None) # 'cbcd', 'clean')
    parser.set_defaults(imtype='p') # 'cbcd', 'clean')
    parser.set_defaults(sigthresh=2.0)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('--overwrite', required=False, dest='skip_existing',
            action='store_false', help='overwrite existing catalogs')
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=False,
            help='where to save extended catalog outputs', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find WIRCam images')
    imtype = iogroup.add_mutually_exclusive_group()
    imtype.add_argument('--cbcd', required=False, action='store_const',
            dest='imtype', const='cbcd', help='use cbcd images')
    imtype.add_argument('--hcfix', required=False, action='store_const',
            dest='imtype', const='hcfix', help='use hot column-fixed images')
    imtype.add_argument('--clean', required=False, action='store_const',
            dest='imtype', const='clean', help='use clean images')
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

    # Unless otherwise specified, output goes into input folder:
    if not context.output_folder:
        context.output_folder = context.input_folder

    # Ensure an image type is selected:
    if not context.imtype:
        sys.stderr.write("\nNo image type selected!\n\n")
        sys.exit(1)

##--------------------------------------------------------------------------##
##------------------         Make Input Image List          ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Listing %s frames ... " % context.imtype) 

## Ensure presence of input folder:
if not os.path.isdir(context.input_folder):
    sys.stderr.write("error: folder not found:\n")
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

## Look for files:
context.imtype = "p"
if context.walk:
    #img_files = sfh.get_files_walk(context.input_folder, flavor=context.imtype)
    img_files = wfh.get_files_walk(context.input_folder, flavor=context.imtype)
else:
    #img_files = sfh.get_files_single(context.input_folder, flavor=context.imtype)
    img_files = wfh.get_files_single(context.input_folder, flavor=context.imtype)
sys.stderr.write("done.\n")

## Abort in case of no input:
if not img_files:
    sys.stderr.write("No input (%s) files found in folder:\n" % context.imtype)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

n_images = len(img_files)

##--------------------------------------------------------------------------##
##------------------           Process All Images           ----------------##
##--------------------------------------------------------------------------##

ntodo = 0
nproc = 0
for ii,img_ipath in enumerate(img_files, 1):
    sys.stderr.write("%s\n" % fulldiv)
    #unc_ipath = img_ipath.replace(context.imtype, 'cbunc')
    #if not os.path.isfile(unc_ipath):
    #    sys.stderr.write("WARNING: file not found:\n--> %s\n" % unc_ipath)
    #    continue
    img_ibase = os.path.basename(img_ipath)

    # no uncertainty images exist for this test case:
    unc_ipath = None

    # set output folder:
    save_dir = context.output_folder
    if context.walk:
        save_dir = os.path.dirname(img_ipath)

    # set output paths:
    cat_fbase = img_ibase + '.fcat'
    cat_fpath = os.path.join(save_dir, cat_fbase)
    sys.stderr.write("Catalog %s ... " % cat_fpath)
    if context.skip_existing:
        if os.path.isfile(cat_fpath):
            sys.stderr.write("exists!  Skipping ... \n")
            continue
        sys.stderr.write("not found ... creating ...\n")
    else:
        sys.stderr.write("creating ... \n")

    # perform extraction:
    nproc += 1
    spf.use_images(ipath=img_ipath, upath=unc_ipath)
    #result = spf.find_stars(context.sigthresh, include_akp=True)
    result = spf.find_stars(context.sigthresh, include_akp=False)
    result.save_as_fits(cat_fpath, overwrite=True)
    if (ntodo > 0) and (nproc >= ntodo):
        break


######################################################################
# CHANGELOG (11_test_extract_WIRCam.py):
#---------------------------------------------------------------------
#
#  2023-05-31:
#     -- Increased __version__ to 0.1.0.
#     -- First created 11_test_extract_WIRCam.py.
#

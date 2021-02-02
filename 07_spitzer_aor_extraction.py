#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract and save extended object catalogs from the specified data and
# uncertainty images. This version of the script jointly analyzes all
# images from a specific AOR/channel to enable more sophisticated
# analysis.
#
# Rob Siverd
# Created:       2021-02-02
# Last modified: 2021-02-02
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
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##
## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers

## Spitzer pipeline cross-correlation:
try:
    import spitz_ccorr_stacking
    reload(spitz_ccorr_stacking)
except ImportError:
    logger.error("failed to import spitz_ccor_stacking module!")
    sys.exit(1)
sxc = spitz_ccorr_stacking.SpitzerXCorr()

## Spitzer star detection routine:
try:
    import spitz_extract
    reload(spitz_extract)
    spf = spitz_extract.SpitzFind()
except ImportError:
    logger.error("spitz_extract module not found!")
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
    parser.set_defaults(imtype='cbcd') #'clean')
    parser.set_defaults(sigthresh=3.0)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=False,
            help='where to save extended catalog outputs', type=str)
    imtype = iogroup.add_mutually_exclusive_group()
    imtype.add_argument('--cbcd', required=False, action='store_const',
            dest='imtype', const='cbcd', help='use cbcd images')
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

##--------------------------------------------------------------------------##
##------------------         Make Input Image List          ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Listing %s frames ... " % context.imtype) 
#im_wildpath = 'SPITZ*%s.fits' % context.imtype
#im_wildcard = os.path.join(context.input_folder, 'SPIT*'
#_img_types = ['cbcd', 'clean', 'cbunc']
#_type_suff = dict([(x, x+'.fits') for x in _im_types])
#img_list = {}
#for imsuff in suffixes:
#    wpath = '%s/SPITZ*%s.fits' % (context.input_folder, imsuff)
#    img_list[imsuff] = sorted(glob.glob(os.path.join(context.
#img_files = sorted(glob.glob(os.path.join(context.input_folder, im_wildpath)))

img_files = sfh.get_files_single(context.input_folder, flavor='cbcd')
sys.stderr.write("done.\n")

## Abort in case of no input:
if not img_files:
    sys.stderr.write("No input (%s) files found in folder:\n" % context.imtype)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

## List of uncertainty frames (warn if any missing):
#unc_files = [x.replace(context.imtype, 'cbunc') for x in img_files]
#sys.stderr.write("Checking error-images ... ") 
#have_unc = [os.path.isfile(x) for x in unc_files]
#if not all(have_unc):
#    sys.stderr.write("WARNING: some uncertainty frames missing!\n")
#else:
#    sys.stderr.write("done.\n") 

##--------------------------------------------------------------------------##
##------------------       Unique AOR/Channel Combos        ----------------##
##--------------------------------------------------------------------------##


unique_tags = list(set([sfh.get_irac_aor_tag(x) for x in img_files]))
images_by_tag = {x:[] for x in unique_tags}
for ii in img_files:
    images_by_tag[sfh.get_irac_aor_tag(ii)].append(ii)


##--------------------------------------------------------------------------##
##------------------           Process All Images           ----------------##
##--------------------------------------------------------------------------##

ntodo = 0
nproc = 0
ntotal = len(img_files)

for aor_tag,tag_files in images_by_tag.items():
    sys.stderr.write("\n\nProcessing images from %s ...\n" % aor_tag)


    # process individual files with cross-correlation help:
    for ii,img_ipath in enumerate(tag_files, 1):
        sys.stderr.write("%s\n" % fulldiv)
        unc_ipath = img_ipath.replace(context.imtype, 'cbunc')
        if not os.path.isfile(unc_ipath):
            sys.stderr.write("WARNING: file not found:\n--> %s\n" % unc_ipath)
            continue
        img_ibase = os.path.basename(img_ipath)
        cat_ibase = img_ibase.replace(context.imtype, 'fcat')
        cat_ipath = os.path.join(context.output_folder, cat_ibase)
        sys.stderr.write("Catalog %s ... " % cat_ipath)
        if os.path.isfile(cat_ipath):
            sys.stderr.write("exists!  Skipping ... \n")
            continue
        break
        nproc += 1
        sys.stderr.write("not found ... creating ...\n")
        spf.use_images(ipath=img_ipath, upath=unc_ipath)
        result = spf.find_stars(context.sigthresh)
        result.save_as_fits(cat_ipath, overwrite=True)
        if (ntodo > 0) and (nproc >= ntodo):
            break


sys.stderr.write("\n\n\n")
sys.stderr.write("sxc.shift_and_stack(tag_files)\n")
result = sxc.shift_and_stack(tag_files)

sys.stderr.write("\n\n\n")
sys.stderr.write("visual inspection with:\n") 
sys.stderr.write("flztfs %s\n" % ' '.join(tag_files))

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (07_spitzer_aor_extraction.py):
#---------------------------------------------------------------------
#
#  2021-02-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created 07_spitzer_aor_extraction.py.
#

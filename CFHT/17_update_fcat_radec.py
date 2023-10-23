#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Find extracted catalogs with ephemerides (flavor "p_eph") and update their
# RA/DE coordinates ('dra' & 'dde' columns). The current procedure involves
# a few iterations of Gaia matching and optimization.
# save with new flavor.
#
# Rob Siverd
# Created:       2023-07-26
# Last modified: 2023-07-26
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
import random
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

## WIRCam pipeline filesystem helpers:
try:
    import wircam_fs_helpers
    reload(wircam_fs_helpers)
except ImportError:
    logger.error("failed to import wircam_fs_helpers module!")
    sys.exit(1)
wfh = wircam_fs_helpers

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ecl = extended_catalog.ExtendedCatalog()
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## WCS tune-up helpers (beta):
try:
    import wircam_wcs_tuneup
    reload(wircam_wcs_tuneup)
    #ecl = extended_catalog.ExtendedCatalog()
    wwt = wircam_wcs_tuneup
except ImportError:
    logger.error("failed to import wircam_wcs_tuneup module!")
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
    #parser.set_defaults(imtype='p') # 'cbcd', 'clean')
    parser.set_defaults(input_flavor='p_eph')
    parser.set_defaults(output_flavor='p_fixed')
    #parser.set_defaults(sigthresh=2.0)
    parser.set_defaults(ntodo=0)
    parser.set_defaults(skip_existing=True)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('--overwrite', required=False, dest='skip_existing',
            action='store_false', help='overwrite existing catalogs')
    #iogroup.add_argument('-Q', '--wcs_pars_path', required=True, type=str,
    #        help='path to CSV file with companion WCS parameters')
    #iogroup.add_argument('-F', '--failure_list',
    #        default='failures.txt', type=str,
    #        help='output list of processing failures [def: %(default)s]')
    #iogroup.add_argument('-E', '--ephem_data', default=None, required=True,
    #        help='where to find input images', type=str)
    iogroup.add_argument('-G', '--gaia_csv', default=None, required=True,
            help='CSV file with Gaia source list', type=str)
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=False,
            help='where to save extended catalog outputs', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find WIRCam images')
    #imtype = iogroup.add_mutually_exclusive_group()
    #imtype.add_argument('--cbcd', required=False, action='store_const',
    #        dest='imtype', const='cbcd', help='use cbcd images')
    #imtype.add_argument('--hcfix', required=False, action='store_const',
    #        dest='imtype', const='hcfix', help='use hot column-fixed images')
    #imtype.add_argument('--clean', required=False, action='store_const',
    #        dest='imtype', const='clean', help='use clean images')
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Miscellany:
    miscgroup = parser.add_argument_group('Miscellany')
    miscgroup.add_argument('--debug', dest='debug', default=False,
            help='Enable extra debugging messages', action='store_true')
    miscgroup.add_argument('-n', '--ntodo', type=int, required=False,
            help='maximum number of images to process [def: %(default)s]')
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
    if not context.input_flavor:
        sys.stderr.write("\nNo input catalog flavor selected!\n\n")
        sys.exit(1)

##--------------------------------------------------------------------------##
##------------------         Load Gaia CSV Catalog          ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Loading Gaia catalog ... ")
if not os.path.isfile(context.gaia_csv):
    sys.stderr.write("error!\nFile not found: %s\n" % context.gaia_csv)
    sys.exit(1)
wwt.gm.load_sources_csv(context.gaia_csv)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------        Make Input Catalog List         ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Listing %s catalogs ... " % context.input_flavor) 

## Ensure presence of input folder:
if not os.path.isdir(context.input_folder):
    sys.stderr.write("error: folder not found:\n")
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

## Look for files:
#context.imtype = "p"
if context.walk:
    cat_files = wfh.get_catalogs_walk(context.input_folder,
            flavor=context.input_flavor)
else:
    cat_files = wfh.get_catalogs_single(context.input_folder,
            flavor=context.input_flavor)
sys.stderr.write("done.\n")

## Abort in case of no input:
if not cat_files:
    sys.stderr.write("No input (%s) catalogs found in folder:\n" % context.input_flavor)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)
n_catalogs = len(cat_files)

## Optionally randomize the file processing order:
#random.shuffle(cat_files)

##--------------------------------------------------------------------------##
##------------------           Load Ephemeris Data          ----------------##
##--------------------------------------------------------------------------##

#sys.stderr.write("Loading ephemeris data ... ")
#if not os.path.isfile(context.ephem_data):
#    sys.stderr.write("error: file not found!\n")
#    sys.stderr.write("--> %s\n\n" % context.ephem_data)
#    sys.exit(1)
#eee.load(context.ephem_data)
#sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------           Process All Images           ----------------##
##--------------------------------------------------------------------------##

ntodo = context.ntodo
nproc = 0
for ii,fcat_path in enumerate(cat_files, 1):
    sys.stderr.write("%s\n" % fulldiv)
    #unc_ipath = img_ipath.replace(context.imtype, 'cbunc')
    #if not os.path.isfile(unc_ipath):
    #    sys.stderr.write("WARNING: file not found:\n--> %s\n" % unc_ipath)
    #    continue
    fcat_base = os.path.basename(fcat_path)
    sys.stderr.write("Processing catalog: %s\n" % fcat_base)

    # output catalog name:
    save_base = wfh.change_catalog_flavor(fcat_base, 
            context.input_flavor, context.output_flavor)
    if not save_base:
        sys.stderr.write("No dice!\n")
        sys.exit(1)

    # set output folder and file:
    save_dir = context.output_folder
    if context.walk:
        save_dir = os.path.dirname(fcat_path)
    save_path = os.path.join(save_dir, save_base)
    gmst_path = save_path + '.gmatch'
    wpar_path = save_path + '.wcspar'
    pix_reg_1 = save_path + '.pix1.reg'
    sky_reg_1 = save_path + '.sky1.reg'
    pix_reg_2 = save_path + '.pix2.reg'
    sky_reg_2 = save_path + '.sky2.reg'
    pix_reg_3 = save_path + '.pix3.reg'
    sky_reg_3 = save_path + '.sky3.reg'

    # announce things:
    sys.stderr.write("Have output folder:\n")
    sys.stderr.write("--> %s\n" % save_dir)
    sys.stderr.write("--> %s\n" % save_path)

    # skip catalogs that already exist:
    sys.stderr.write("Output catalog %s ... " % save_base)
    if context.skip_existing:
        if os.path.isfile(save_path):
            sys.stderr.write("exists!  Skipping ...\n")
            continue
        sys.stderr.write("not found ... creating ...\n")
    else:
        sys.stderr.write("creating ... \n")

    # If we get here, we need to do processing:
    nproc += 1

    # fetch ephemeris using partial filename:
    #ftag = fcat_base.split('.')[0]
    #this_eph = eee.get_eph_by_name(ftag)

    # load the catalog:
    ecl.load_from_fits(fcat_path)
    stars  = ecl.get_catalog()
    header = ecl.get_header()

    # perform the tune-up:
    #new_stars = wwt.wcs_tuneup(stars, header)
    new_stars = wwt.wcs_tuneup(stars, header, 
            save_matches=gmst_path, save_wcspars=wpar_path,
            pixreg1=pix_reg_1, skyreg1=sky_reg_1,
            pixreg2=pix_reg_2, skyreg2=sky_reg_2,
            pixreg3=pix_reg_3, skyreg3=sky_reg_3)

    # update the catalog:
    ecl.set_catalog(new_stars)

    # save the result:
    ecl.save_as_fits(save_path, overwrite=True)

    # stop early if requested:
    if (ntodo > 0) and (nproc >= ntodo):
        break


######################################################################
# CHANGELOG (17_update_fcat_radec.py):
#---------------------------------------------------------------------
#
#  2023-07-26:
#     -- Increased __version__ to 0.1.0.
#     -- First created 17_update_fcat_radec.py.
#

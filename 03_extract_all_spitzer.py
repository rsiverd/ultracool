#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract and save extended object catalogs from the specified data and
# uncertainty images.
#
# Rob Siverd
# Created:       2021-02-01
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
## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers


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
    parser.set_defaults(imtype=None) # 'cbcd', 'clean')
    parser.set_defaults(sigthresh=2.0)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=True,
            help='where to save extended catalog outputs', type=str)
    imtype = iogroup.add_mutually_exclusive_group()
    imtype.add_argument('--cbcd', required=False, action='store_const',
            dest='imtype', const='cbcd', help='use cbcd images')
    imtype.add_argument('--hcfix', required=False, action='store_const',
            dest='imtype', const='hcfix', help='use clean images')
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

##--------------------------------------------------------------------------##
##------------------         Make Input Image List          ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Listing %s frames ... " % context.imtype) 
im_wildpath = 'SPITZ*%s.fits' % context.imtype
#im_wildcard = os.path.join(context.input_folder, 'SPIT*'
#_img_types = ['cbcd', 'clean', 'cbunc']
#_type_suff = dict([(x, x+'.fits') for x in _im_types])
#img_list = {}
#for imsuff in suffixes:
#    wpath = '%s/SPITZ*%s.fits' % (context.input_folder, imsuff)
#    img_list[imsuff] = sorted(glob.glob(os.path.join(context.
img_files = sorted(glob.glob(os.path.join(context.input_folder, im_wildpath)))
sys.stderr.write("done.\n")

## List of uncertainty frames (warn if any missing):
#unc_files = [x.replace(context.imtype, 'cbunc') for x in img_files]
#sys.stderr.write("Checking error-images ... ") 
#have_unc = [os.path.isfile(x) for x in unc_files]
#if not all(have_unc):
#    sys.stderr.write("WARNING: some uncertainty frames missing!\n")
#else:
#    sys.stderr.write("done.\n") 

##--------------------------------------------------------------------------##
##------------------         Make Output File List          ----------------##
##--------------------------------------------------------------------------##

## Assemble path:
#cat_list =
## Check for existing output frames:
#sys.stderr.write("Checking for existing output ... ")
#cat_list = 

##--------------------------------------------------------------------------##
##------------------           Process All Images           ----------------##
##--------------------------------------------------------------------------##

ntotal = len(img_files)
#for ii,(ipath,upath) in enumerate(zip(img_list, unc_list), 1):
#    sys.stderr.write("\rImage %d of %d ... \n" % (ii, ntotal))

ntodo = 0
nproc = 0
for ii,img_ipath in enumerate(img_files, 1):
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
    nproc += 1
    sys.stderr.write("not found ... creating ...\n")
    spf.use_images(ipath=img_ipath, upath=unc_ipath)
    result = spf.find_stars(context.sigthresh)
    result.save_as_fits(cat_ipath, overwrite=True)
    if (ntodo > 0) and (nproc >= ntodo):
        break



##--------------------------------------------------------------------------##
## Process images in order:

##--------------------------------------------------------------------------##
## Process images in order:
##--------------------------------------------------------------------------##
## Quick ASCII I/O:
#data_file = 'data.txt'
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = aia.read(data_file)

#all_data = pd.read_csv(data_file)
#all_data = pd.read_table(data_file, delim_whitespace=True)
#all_data = pd.read_table(data_file, sep='|')
#fields = all_data.dtype.names
#if not fields:
#    x = all_data[:, 0]
#    y = all_data[:, 1]
#else:
#    x = all_data[fields[0]]
#    y = all_data[fields[1]]

#vot_file = 'neato.xml'
#vot_data = av.parse_single_table(vot_file)
#vot_data = av.parse_single_table(vot_file).to_table()




######################################################################
# CHANGELOG (05_extract_all_spitzer.py):
#---------------------------------------------------------------------
#
#  2019-10-29:
#     -- Increased __version__ to 0.1.0.
#     -- First created 05_extract_all_spitzer.py.
#

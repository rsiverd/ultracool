#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Create a bad pixel mask from a list of WIRCam images. The bad pixel mask
# is the set of pixels that are set to zero in all of the processed frames
# for a specific RUNID.
#
# Rob Siverd
# Created:       2024-10-21
# Last modified: 2024-10-21
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
#import pickle
#import vaex
#import calendar
#import ephem
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## WIRCam config:
import wircam_config
reload(wircam_config)
wcfg = wircam_config

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
def qsave(iname, idata, header=None, padkeys=1000, vlevel=1, **kwargs):
    this_func = sys._getframe().f_code.co_name
    parent_func = sys._getframe(1).f_code.co_name
    if vlevel >= 1:
        sys.stderr.write("Writing to '%s' ... " % iname)
    if header:
        while (len(header) < padkeys):
            header.append() # pad header
    if os.path.isfile(iname):
        os.remove(iname)
    pf.writeto(iname, idata, header=header, **kwargs)
    if vlevel >= 1:
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
    Generate bad pixel masks by RUNID for WIRCam processing.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    #parser.set_defaults(quadrant='NE')
    parser.set_defaults(quadrant=None)
    parser.set_defaults(saturval=65535)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
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
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-i', '--hdrdata', default=None, required=True,
            help='CSV file with RUNID listing', type=str)
    iogroup.add_argument('-o', '--outdir', default=None, required=True,
            help='Output folder for bad pixel masks', type=str)
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

## Abort if no quadrant specified:
if not context.quadrant:
    sys.stderr.write("Error: no quadrant specified!\n")
    sys.exit(1)

## Choose extension name based on quadrant:
sys.stderr.write("Chosen quadrant: %s\n" % context.quadrant)
qnum, use_extname = wcfg.quad_exts.get(context.quadrant)
sys.stderr.write("Using extension: %s\n" % use_extname)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Load header data from file:
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
##pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
pdkwargs.update({'delim_whitespace':True}) #, 'sep':'|', 'escapechar':'#'})
hdata = pd.read_csv(context.hdrdata, **pdkwargs)

## Create output folder:
sys.stderr.write("Output folder: %s\n" % context.outdir)
if not os.path.isdir(context.outdir):
    os.mkdir(context.outdir)

## List of unique QRUNIDs:
every_QRUNID = sorted(list(set(hdata.QRUNID)))

## Group by QRUNID, iterate:
chunks = hdata.groupby('QRUNID')
for qrunid,subset in chunks:
    sys.stderr.write("qrunid: %s\n" % qrunid)
    nimgs = len(subset)
    sys.stderr.write("Have %d 'p' images with QRUNID=%s.\n" % (nimgs, qrunid))

    # make output folder:
    save_dir = os.path.join(context.outdir, qrunid)
    sys.stderr.write("save_dir: %s\n" % save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, 'badpix.fits')

    # skip if mask already exists:
    if os.path.isfile(save_file):
        sys.stderr.write("Mask already exists: %s\n" % save_file)
        continue

    sys.stderr.write("Loading ...\n")
    #frame_data = []
    pixel_mask = []
    mask_stack = []
    for ii,ipath in enumerate(subset.FILENAME, 1):
        sys.stderr.write("\rLoading image %d of %d ... " % (ii, nimgs))
        try:
            pdata = pf.getdata(ipath, extname=use_extname)
        except:
            sys.stderr.write("Failed to load image: %s\n" % ipath)
            sys.stderr.write("extension: %s\n" % use_extname)
            sys.exit(1)
        #frame_data.append(pf.getdata(ipath, extname=use_extname))
        #pixel_mask.append((pdata == 0))
        mask_stack.append(np.int_(pdata == 0))
        pass
    sys.stderr.write("done.\n")
    #sys.stderr.write("Making median ... ")
    #mask = np.median(mask_stack, axis=0)
    sys.stderr.write("Taking average ... ")
    mask = np.mean(mask_stack, axis=0)
    sys.stderr.write("saving ... ")
    qsave(save_file, mask, vlevel=0, overwrite=True)
    sys.stderr.write("done.\n")
    #break
    pass



######################################################################
# CHANGELOG (make-wircam-pixmask.py):
#---------------------------------------------------------------------
#
#  2024-10-21:
#     -- Increased __version__ to 0.1.0.
#     -- First created make-wircam-pixmasks.py.
#

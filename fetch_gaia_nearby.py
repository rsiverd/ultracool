#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Retrieve Gaia sources near the specified target.
#
# Rob Siverd
# Created:       2019-09-05
# Last modified: 2025-01-14
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.4.0"

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
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#from matplotlib import colors
#import matplotlib.colors as mplcolors
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
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

## Fast FITS I/O:
#try:
#    import fitsio
#except ImportError:
#    sys.stderr.write("\nError: fitsio module not found!\n")
#    sys.exit(1)

## FITS I/O:
#try:
#    import astropy.io.fits as pf
#except ImportError:
#    try:
#       import pyfits as pf
#    except ImportError:
#        sys.stderr.write("\nError!  No FITS I/O module found!\n"
#               "Install either astropy.io.fits or pyfits and try again!\n\n")
#        sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
    from astropy import coordinates as coord
    from astropy import units as uu
    from astroquery.gaia import Gaia
except ImportError:
    sys.stderr.write("\nError: astropy/astroquery module not found!\n")
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
#def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    if header:
#        while (len(header) < padkeys):
#            header.append() # pad header
#    if os.path.isfile(iname):
#        os.remove(iname)
#    pf.writeto(iname, idata, header=header, **kwargs)
#    sys.stderr.write("done.\n")

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
    Retrieve nearby Gaia sources using cone search.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(cone_rad_deg=0.5)
    parser.set_defaults(overwrite=False)
    #parser.set_defaults(row_limit=0)
    parser.set_defaults(data_release=None)
    # ------------------------------------------------------------------
    parser.add_argument('RA_deg', help='target RA in degrees', type=float)
    parser.add_argument('DE_deg', help='target RA in degrees', type=float)
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    parser.add_argument('-o', '--output_file', required=True, default=None,
            help='output file for Gaia sources (CSV)', type=str)
    parser.add_argument('--overwrite', required=False, action='store_true',
            help='enable overwrite of existing output file')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    vgroup = parser.add_argument_group('Data Release')
    vgroup.add_argument('--DR2', required=False, dest='data_release',
            action='store_const', const='DR2', help='use Gaia Data Release 2')
    vgroup.add_argument('--DR3', required=False, dest='data_release',
            action='store_const', const='DR3', help='use Gaia Data Release 3')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    qgroup = parser.add_argument_group('Query Parameters')
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    qgroup.add_argument('-R', '--radius', default=0.5, required=False,
            help='radius of search cone in DEGREES')
    qgroup.add_argument('--rowlimit', required=False, default=0,
            help='row limit for query [def: %(default)s]', type=int)
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

## Abort in case no data release is specified:
if not context.data_release:
    sys.stderr.write("\nNo data release specified!\n\n")
    parser.print_help()
    sys.exit(0)

## Stop early in case of existing output unless --overwrite given:
if not context.overwrite:
    if os.path.isfile(context.output_file):
        sys.stderr.write("Found existing output file:\n")
        sys.stderr.write("--> %s\n" % context.output_file)
        sys.stderr.write("Gaia data already retrieved!\n")
        sys.exit(0)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Search cone setup:
tgtcoo = coord.SkyCoord(ra=context.RA_deg, dec=context.DE_deg,
            unit=(uu.deg, uu.deg), frame='icrs')
cone_radius = uu.Quantity(context.radius, uu.deg)

## Choose data release:
main_table = {'DR2':'gaiadr2.gaia_source', 'DR3':'gaiadr3.gaia_source'}
#Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"    # does this work?
Gaia.MAIN_GAIA_TABLE = main_table[context.data_release]

## Perform query:
sys.stderr.write("Running query ... \n")
tik  = time.time()
Gaia.ROW_LIMIT = context.rowlimit
qobj = Gaia.cone_search_async(tgtcoo, cone_radius)
hits = qobj.get_results()
tok  = time.time()
sys.stderr.write("Query time: %.3f seconds.\n" % (tok-tik))

## Force column names to lower-case (with dupe check):
have_cols = hits.keys()
want_cols = [x.lower() for x in have_cols]
if len(set(want_cols)) != len(have_cols):
    # this should never happen, but it's a big problem if it does ...
    sys.stderr.write("Column names appear to be duplicated ...\n")
    sys.stderr.write("You should never see this!  Manual rename needed ...\n")
    sys.exit(1)
for old,new in zip(have_cols, want_cols):
    hits.rename_column(old, new)

## Save results:
if context.output_file:
    sys.stderr.write("Saving to %s ... " % context.output_file)
    hits.write(context.output_file, format='ascii.csv', overwrite=True)
    sys.stderr.write("done.\n")



######################################################################
# CHANGELOG (fetch_gaia_nearby.py):
#---------------------------------------------------------------------
#
#  2025-01-14:
#     -- Increased __version__ to 0.4.0.
#     -- Added --rowlimit, --DR2, and --DR3 options.
#     -- Added DR3 support. Data release
#     -- Removed some big comment blocks with unused boilerplate.
#
#  2024-05-29:
#     -- Increased __version__ to 0.3.0.
#     -- Now force all Gaia column names to lower-case.
#     -- Added overwrite=True to results output to silence warnings.
#     -- Fixed missing store_true in --overwrite argparse spec.
#
#  2021-12-16:
#     -- Increased __version__ to 0.2.0.
#     -- Various changes.
#
#  2019-09-05:
#     -- Increased __version__ to 0.1.0.
#     -- First created fetch_gaia_nearby.py.
#

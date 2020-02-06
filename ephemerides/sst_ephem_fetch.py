#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Get the Spitzer barycentric positions and velocities for all images
# in my dataset.
#
# Rob Siverd
# Created:       2020-02-06
# Last modified: 2020-02-06
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
import glob
import gc
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

##--------------------------------------------------------------------------##

## Fast FITS I/O:
try:
    import fitsio
except ImportError:
    logger.error("fitsio module not found!  Install and retry.")
    sys.stderr.write("\nError: fitsio module not found!\n")
    sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

## HORIZONS queries:
from astroquery.jplhorizons import Horizons

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
    Fetch Spitzer barycentric positions and velocities for
    images in the specified folder.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    parser.set_defaults(max_todo=0, qmax=100)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-I', '--image_dir', default=None, required=True,
            help='folder with Spitzer images')
    iogroup.add_argument('-o', '--output_file', default=None, required=True,
            help='CSV output file for retrieved ephemeris', type=str)
    #fmtparse.set_defaults(output_mode='pydict')
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
#context.max_todo = 500

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Look for cbcd files in directory provided:
if not os.path.isdir(context.image_dir):
    sys.stderr.write("Folder not found: '%s'\n" % context.image_dir)
    sys.exit(1)
image_list = sorted(glob.glob("%s/SPITZER*cbcd.fits" % context.image_dir))
if not image_list:
    sys.stderr.write("No viable images found in %s!\n" % context.image_dir)
    sys.exit(1)
total = len(image_list)
sys.stderr.write("Found %d SPITZER images.\n" % len(image_list))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Retrieve headers from all images:
tik = time.time()
hdr_data = []
for ii,ipath in enumerate(image_list, 1):
    sys.stderr.write("\rLoading image %d of %d ... " % (ii, total))
    sst_hdr = fitsio.read_header(ipath)
    sst_hdr['IMGPATH'] = os.path.basename(ipath)
    hdr_data.append(sst_hdr)
    sys.stderr.write("done.   ")
    if (context.max_todo > 0) and (ii >= context.max_todo):
        break
sys.stderr.write("\n")
tok = time.time()
sys.stderr.write("Loaded header(s) in %.3f sec.\n" % (tok-tik))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Assemble TDB timestamps:
obs_dates = [x['DATE_OBS'] for x in hdr_data]
timestamps = astt.Time(obs_dates, scale='utc', format='isot')
names_used = [x['IMGPATH'] for x in hdr_data]

## Query HORIZONS piecewise (avoids 2000-char URL length limit):
sys.stderr.write("Querying HORIZONS ...\n")
tik = time.time()
loc_ssb = {'location':'@0'}    # solar system barycenter
spitzkw = {'id':'Spitzer Space Telescope', 'id_type':'id'}
#sst_query = Horizons(**spitzkw, **loc_ssb, epochs=timestamps.tdb.jd.tolist())
#sst_query.TIMEOUT = 1000
nchunks = (timestamps.tdb.jd.size // context.qmax) + 1
batches = np.array_split(timestamps.tdb.jd, nchunks)
results = []
for ii,batch in enumerate(batches, 1):
    sys.stderr.write("\rQuery batch %d of %d ... " % (ii, nchunks))
    sst_query = Horizons(**spitzkw, **loc_ssb, epochs=batch.tolist())
    batch_eph = sst_query.vectors()
    results.append(batch_eph)

## Combine into single table:
horiz_eph = apt.vstack(results)
#try:
#    horiz_eph = sst_query.vectors()
#except:
#    tok = time.time()
#    sys.stderr.write("\nCrapped out after %.3f sec..\n" % (tok-tik))
#    sys.exit(1)
tok = time.time()
sys.stderr.write("done. %.3f sec\n" % (tok-tik))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Column config:
iname_col = 'filename'
drop_cols = ['targetname', 'datetime_str']

## Make adjustments to a copy of the results:
sst_table = horiz_eph.copy()
for cc in drop_cols:
    if cc in sst_table.keys():
        sst_table.remove_column(cc)

## Columns to be saved in CSV:
want_cols = [iname_col]
want_cols.extend(sst_table.keys())
#want_cols.extend(sst_table.keys()[1:])   # drops 'targetname'

## Attach file names and re-order columns:
imlist_col = apt.Column(data=names_used, name='filename')
sst_table.add_column(imlist_col)
sst_table = sst_table[want_cols]

## Save result as CSV:
sys.stderr.write("Saving to %s ... " % context.output_file)
sst_table.write(context.output_file, format='csv', overwrite=True)
sys.stderr.write("done.\n")


######################################################################
# CHANGELOG (sst_ephem_fetch.py):
#---------------------------------------------------------------------
#
#  2020-02-06:
#     -- Increased __version__ to 0.1.0.
#     -- First created sst_ephem_fetch.py.
#

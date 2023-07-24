#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This script retrieves barycentric XYZ coordinates from JPL HORIZONS for the 
# Canada-France-Hawaii Telescope (CFHT) at times relevant for the available
# images of a single target (folder and subfolders).
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
#import glob
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

## HORIZONS ephemeris interaction:
try:
    import jpl_eph_helpers
    reload(jpl_eph_helpers)
except ImportError:
    logger.error("failed to import jpl_eph_helpers module!")
    sys.exit(1)
fhe = jpl_eph_helpers.FetchHorizEphem()

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
    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.exit(1)

## HORIZONS queries:
try:
    from astroquery.jplhorizons import Horizons
except ImportError:
    logger.error("Unable to load astroquery/Horizons module!")
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
    Retrieve HORIZONS ephemeris positions for CFHT WIRCam images.

    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(gather_headers=True)
    parser.set_defaults(qmax=50)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-I', '--image_folder', default=None, required=True,
            help='where to find WIRCam images', type=str)
    iogroup.add_argument('-o', '--output_file', required=True, default=None,
            help='where to save retrieved ephemeris data (CSV)', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find WIRCam images')
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

## Ensure input folder exists:
if not os.path.isdir(context.image_folder):
    sys.stderr.write("\nError! Folder not found:\n")
    sys.stderr.write("--> %s\n\n" % context.image_folder)
    sys.exit(1)

## Get list of CBCD files:
iflav = 'p'
if context.walk:
    all_cfht_files = wfh.get_files_walk(context.image_folder, flavor=iflav)
else:
    all_cfht_files = wfh.get_files_single(context.image_folder, flavor=iflav)
n_images = len(all_cfht_files)
sys.stderr.write("Identified %d '%s' FITS images.\n" % (n_images, iflav))

## Retrieve FITS headers:
if context.gather_headers:
    sys.stderr.write("Loading FITS headers for all files ...\n")
    #tik = time.time()
    #all_cfht_headers = [pf.getheader(x, extname='HAWAII-2RG-#60') \
    #        for x in all_cfht_files]
    #tok = time.time()

    tik = time.time()
    all_cfht_headers = []
    for i,ipath in enumerate(all_cfht_files, 1):
        sys.stderr.write("\rLoading header %d of %d ... " % (i, n_images))
        ihdr = pf.getheader(ipath, extname='HAWAII-2RG-#60')
        all_cfht_headers.append(ihdr)
    tok = time.time()

    sys.stderr.write("done.\n")
    sys.stderr.write("Header load took %.3f seconds.\n" % (tok - tik))

## Make list of base names for storage:
img_bases  = [os.path.basename(x) for x in all_cfht_files]


## Extract observation timestamps:
#obs_dates  = [x['DATE-OBS'] for x in all_cfht_headers]
#obs_dates  = [x['DATE-OBS'] for x in all_cfht_headers]
mjd_start  = [x['MJD-OBS']  for x in all_cfht_headers]
exp_times  = [x['EXPTIME']  for x in all_cfht_headers]
timestamps = astt.Time(mjd_start, scale='utc', format='mjd') \
                + 0.5 * astt.TimeDelta(exp_times, format='sec')

## Sort bases and timestamps by timestamp:
order = np.argsort(timestamps.tdb.jd)
timestamps = timestamps[order]
img_bases = [img_bases[x] for x in order]

#ntodo = 30
#timestamps = timestamps[:ntodo]
#img_bases  = img_bases[:ntodo]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## For more information about how to specify the observing site, visit:
## https://ssd.jpl.nasa.gov/horizons/manual.html#center
## https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html

## From the latter:

## Horizons always interprets longitude values as eastward. However, there are
## two major gotchas in this: 1. For most prograde rotators, which is to say
## most major bodies, Horizons interprets west-longitude as positive and
## east-longitude as negative. However, values must still be entered in
## east-longitude, which means they must be negative; Horizons will raise an
## error if given any positive longitude value for such bodies. Instead enter
## the west-longitude - 360. For instance, a site on Mars (id code 499) at 30
## degrees longitude, 30 degrees latitude, 0 km elevation should be specified
## as {'body': 499, 'elevation': 0, 'lon': -330, 'lat': 30}. 2. This does not
## apply to the Earth, Moon, and Sun.  Although they are prograde, Horizons
## interprets east-longitude as positive and west-longitude as negative for
## these bodies.



#sys.exit(0)
## Retrieve Spitzer ephemeris:
#spitzkw = {'id':'Spitzer Space Telescope', 'id_type':'id'}
#cfht_kw = {'id':'Spitzer Space Telescope', 'id_type':'id'}
#cfht_kw = {'id':'CFH@399', 'id_type':None}
#cfht_kw = {'id':'399', 'id_type':None}
#cfht_kw = {'id':'T14@399', 'id_type':None}
cfh_kw = {     'body':   399,           # Earth
                'lon':  -155.47166667,
                'lat':   +19.82829444,
          'elevation':     4.2094}
ssb_kw = {'id':'T14', 'id_type':None}
#cfht_kw = {'id':'267@399', 'id_type':None}
#cfht_kw = {'id':'CFH', 'id_type':None}
#fhe.set_target(ssb_kw)
fhe.set_target(cfh_kw)
fhe.set_imdata(img_bases, timestamps)
sst_table = fhe.get_ephdata()

## Save result as CSV:
sys.stderr.write("Saving to %s ... " % context.output_file)
sst_table.write(context.output_file, format='csv', overwrite=True)
sys.stderr.write("done.\n")


######################################################################
# CHANGELOG (01_get_SST_ephemeris.py):
#---------------------------------------------------------------------
#
#  2021-03-16:
#     -- Increased __version__ to 0.1.0.
#     -- First created 01_get_SST_ephemeris.py.
#

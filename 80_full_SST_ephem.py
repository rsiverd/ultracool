#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This script retrieves barycentric XYZ coordinates from JPL HORIZONS for the 
# Spitzer Space Telescope at daily intervals for the duration of the mission.
# These data points are intended to be used for visualization tasks such as
# drawing RA(t) and DE(t) for given astrometric parameters.
#
# This script is effectively identical to the one used for individual image
# ephemeris retrieval except the image names are dummy files that do not
# exist on disk.
#
# Rob Siverd
# Created:       2021-08-23
# Last modified: 2021-08-23
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
import os
import sys
import time
import numpy as np
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers

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
    Retrieve HORIZONS ephemeris positions for Spitzer images.
    
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
    #iogroup.add_argument('-I', '--image_folder', default=None, required=True,
    #        help='where to find CBCD images', type=str)
    iogroup.add_argument('-o', '--output_file', required=True, default=None,
            help='where to save retrieved ephemeris data', type=str)
    #iogroup.add_argument('-W', '--walk', default=False, action='store_true',
    #        help='recursively walk subfolders to find CBCD images')
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

## Spitzer Space Telescope launched on 2003-08-25. Mission concludes in 2020.
t_launch = astt.Time('2003-09-01T00:00:00', scale='tdb', format='isot')
t_retire = astt.Time('2020-02-01T00:00:00', scale='tdb', format='isot')

## Create a range of TDB times:
duration_days = (t_retire - t_launch).jd
offsets_d  = astt.TimeDelta(np.arange(duration_days), format='jd')
timestamps = t_launch + offsets_d

## Fabricate corresponding image strings:
date_strings = [x.split('T')[0].replace('-', '') for x in timestamps.isot]
img_bases = ['dummy_%s.fits'%x for x in date_strings]
#sys.exit(0)

### Extract observation timestamps:
#obs_dates  = [x['DATE_OBS'] for x in all_cbcd_headers]
#exp_times  = [x['EXPTIME']  for x in all_cbcd_headers]
#timestamps = astt.Time(obs_dates, scale='utc', format='isot') \
#                + 0.5 * astt.TimeDelta(exp_times, format='sec')
#
### Sort bases and timestamps by timestamp:
#order = np.argsort(timestamps.tdb.jd)
#timestamps = timestamps[order]
#img_bases = [img_bases[x] for x in order]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Retrieve Spitzer ephemeris:
spitzkw = {'id':'Spitzer Space Telescope', 'id_type':'id'}
fhe.set_target(spitzkw)
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
#  2021-08-23:
#     -- Increased __version__ to 0.1.0.
#     -- First created 80_full_SST_ephem.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Find stars on the specified image, extract X/Y/flux, and perform
# initial match against Gaia source list using image-provided WCS.
#
# Rob Siverd
# Created:       2019-09-09
# Last modified: 2019-09-09
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
__version__ = "0.1.5"

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
import gc
import os
import sys
import time
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
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Easy Gaia source matching:
try:
    import gaia_match
    reload(gaia_match)
    gm = gaia_match.GaiaMatch()
except ImportError:
    logger.error("failed to import gaia_match module!")
    sys.exit(1)

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

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

## Home-brew robust statistics:
try:
    import robust_stats
    reload(robust_stats)
    rs = robust_stats
except ImportError:
    logger.error("module robust_stats not found!  Install and retry.")
    sys.exit(1)

## Fast FITS I/O:
#try:
#    import fitsio
#except ImportError:
#    sys.stderr.write("\nError: fitsio module not found!\n")
#    sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
#    import astropy.table as apt
#    import astropy.time as astt
    import astropy.io.fits as pf
    import astropy.wcs as awcs
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.exit(1)

## Star extraction:
try:
    import easy_sep
    reload(easy_sep)
except ImportError:
    logger.error("easy_sep module not found!  Install and retry.")
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
    Star identification tasks:
    1) Extract sources with SEP
    2) Determine approximate source coordinates from image WCS
    3) Match extracted sources to Gaia 

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
    #parser.add_argument('-n', '--number_of_days', default=1,
    #        help='Number of days of data to retrieve.')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-g', '--gaia_csv', default=None, required=True,
            help='CSV file with Gaia source list', type=str)
    iogroup.add_argument('-i', '--input_image', default=None, required=True,
            help='input FITS image to analyze', type=str)
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
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
##------------------       load Gaia sources from CSV       ----------------##
##--------------------------------------------------------------------------##

try:
    logger.info("Loading sources from %s" % context.gaia_csv)
    gm.load_sources_csv(context.gaia_csv)
except:
    logger.error("Yikes ...")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------       load image / initialize WCS      ----------------##
##--------------------------------------------------------------------------##

try:
    logger.info("Loading image %s" % context.input_image)
    rdata, rhdrs = pf.getdata(context.input_image, header=True)
    ihdrs = rhdrs.copy(strip=True)
except:
    logger.error("Unable to load image: %s" % context.input_image)
    sys.exit(1)

## Working copy of image data:
idata = rdata.copy().astype('float32')

## Initialize WCS:
imwcs = awcs.WCS(ihdrs)


##--------------------------------------------------------------------------##
##------------------   as-needed image unit conversion      ----------------##
##--------------------------------------------------------------------------##

## Spitzer config:
unit_key = 'BUNIT'
conv_key = 'FLUXCONV'
gain_key = 'GAIN'
tele_key = 'TELESCOP'

## Telescope-specific stuff:
this_tele = ihdrs.get('TELESCOP', 'UNKNOWN')
match_tol = 0.1
if (this_tele == 'Spitzer'):
    # Conversion below adopted from IRAC Instrument Handbook, p. 105:
    # BUNIT gives the units (MJy/sr) of the images. For reference, 1 MJy/sr =
    # 10–17 erg s–1 cm–2 Hz–1 sr–1. FLUXCONV is the calibration factor derived
    # from standard star observations; its units are (MJy/sr)/(DN/s). The raw
    # files are in “data numbers” (DN). To convert from MJy/sr back to DN,
    # divide by FLUXCONV and multiply by EXPTIME. To convert DN to electrons,
    # multiply by GAIN.
    logger.info("Detected Spitzer image!")
    idata *= float(ihdrs['EXPTIME']) / float(ihdrs['FLUXCONV']) # now in DN
    idata *= float(ihdrs['GAIN'])       # now in electrons
    match_tol = 10.0 / 3600.0           # 10 arcseconds in degrees

## Mask hot/bad pixels and estimate background:
tik = time.time()
#bright_pixels = (raw_vals >= 50000)
#pse.set_image(raw_vals, gain=gain)
pse.set_image(idata, gain=1.0)
pse.set_options(minpixels=5)
#pse.set_mask(bright_pixels)

## Extract stars:
pix_origin = 1.0
#useobjs = pse.analyze(sigthresh=3.0)
useobjs = pse.analyze(sigthresh=3.0)
badobjs = pse.badobjs
allobjs = pse.allobjs
ssub_data = pse.sub_data

## X,Y,mag for astrometry:
ccd_xx  = useobjs['x'] + pix_origin
ccd_yy  = useobjs['y'] + pix_origin
#ccd_mag = flux2mag(useobjs['flux'])
tok = time.time()
sys.stderr.write("SEP star extraction time: %.3f sec\n" % (tok-tik))
logger.info("SEP star extraction time: %.3f sec\n" % (tok-tik))

## Convert to RA/Dec using WCS:
ccd_ra, ccd_de = imwcs.all_pix2world(ccd_xx, ccd_yy, pix_origin)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Cross-match all detections against Gaia:
#gaia_hits = []
#for tra,tde in zip(ccd_ra, ccd_de):
#    gaia_hits.append(
#    pass
sys.stderr.write("Initial match to Gaia sources ... ")
tik = time.time()
results = [gm.nearest_star(tra, tde, match_tol) \
                for tra,tde in zip(ccd_ra, ccd_de)]
matched = np.array([x['match'] for x in results])
#records = [x['record'] for x in results if x['match']]
#gaia_hits = pd.concat(records, ignore_index=True)

### FIXME
### the concatenation below should be replaced with a single batch
### selection of rows from the parent table using indexes of objects
### found to match. This will avoid a lot of overhead. MORE IMPORTANTLY,
### that selection and construction logic should be hidden away inside
### the source-matching module!!!
gaia_hits = pd.concat([x['record'] for x in results if x['match']],
                ignore_index=True)
gaia_hits['xpix'] = ccd_xx[matched]
gaia_hits['ypix'] = ccd_yy[matched]
tok = time.time()
sys.stderr.write("done. Found %d matches in %.3f seconds.\n"
        % (len(gaia_hits), tok-tik))

sys.exit(0)

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




######################################################################
# CHANGELOG (extract_and_match_gaia.py):
#---------------------------------------------------------------------
#
#  2019-09-10:
#     -- Increased __version__ to 0.1.5.
#     -- Gaia source loading and nearest-neighbor lookup completed.
#
#  2019-09-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created extract_and_match_gaia.py.
#
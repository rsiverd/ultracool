#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Driver script for Spitzer image cross-correlation. Intended for use with
# images from the same AOR/visit.
#
# Rob Siverd
# Created:       2021-01-14
# Last modified: 2021-01-14
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
import gc
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
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Fresh start (for ipython):
gc.collect()

##--------------------------------------------------------------------------##
## Disable buffering on stdout/stderr:
#class Unbuffered(object):
#   def __init__(self, stream):
#       self.stream = stream
#   def write(self, data):
#       self.stream.write(data)
#       self.stream.flush()
#   def __getattr__(self, attr):
#       return getattr(self.stream, attr)
#
#sys.stdout = Unbuffered(sys.stdout)
#sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##

## Home-brew robust statistics:
try:
    import robust_stats
    reload(robust_stats)
    rs = robust_stats
except ImportError:
    logger.error("module robust_stats not found!  Install and retry.")
    sys.stderr.write("\nError!  robust_stats module not found!\n"
           "Please install and try again ...\n\n")
    sys.exit(1)

## Fast FITS I/O:
try:
    import fitsio
except ImportError:
    logger.error("fitsio module not found!  Install and retry.")
    sys.stderr.write("\nError: fitsio module not found!\n")
    sys.exit(1)

## Various from astropy:
#try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
#except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
#    sys.exit(1)

## Star extraction:
#try:
#    import easy_sep
#    reload(easy_sep)
#except ImportError:
#    logger.error("easy_sep module not found!  Install and retry.")
#    sys.stderr.write("Error: easy_sep module not found!\n\n")
#    sys.exit(1)
#pse = easy_sep.EasySEP()

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
    Cross-correlate Spitzer images. This script independently determines X and
    Y shifts between input images using FFT methods. Images are assumed to be
    related by a simple XY shift (no rotation), which is the case for data
    from a single AOR.
    
    Bright pixels corresponding to sources are correlated to maximize SNR.
    A specific member of the set may be chosen as reference. If none specified,
    the first image in the set will be taken as reference.

    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(thing1='value1', thing2='value2')
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-n', '--number_of_days', default=1,
    #        help='Number of days of data to retrieve.')
    parser.add_argument('-o', '--output_file', required=True,
            default=None, help='Output filename.')
    parser.add_argument('imlist', nargs='*',
            help='full paths to FITS images')
    parser.add_argument('-T', '--tight', default=False, action='store_true',
            dest='tight_pad', help='enable tight padding')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #iogroup = parser.add_argument_group('File I/O')
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
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

tstart = time.time()

## Load input images:
imdata = [fitsio.read(ff) for ff in context.imlist]
ny, nx = imdata[0].shape

## Padding:
ypads = int(0.5 * ny)
xpads = int(0.5 * nx)
yxpads = (ypads, xpads)

##--------------------------------------------------------------------------##
## Replace NaN/Inf values prior to cross-correlation attempt:
for frame in imdata:
    which = np.isnan(frame) | np.isinf(frame)
    replacement = np.median(frame[~which])
    frame[which] = replacement

## Create bright pixel masks representing source locations:
bpmask = []
thresh = 20.0
for frame in imdata:
    pix_med, pix_iqrn = rs.calc_ls_med_IQR(frame)
    bright = (frame - pix_med >= thresh * pix_iqrn)
    bpmask.append(bright)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## FFT-assisted cross-correlation:
def ccalc(rowcol1, rowcol2):
    cft1 = np.fft.fft(rowcol1)
    cft2 = np.fft.fft(rowcol2)
    cft2.imag *= -1.0
    corr = np.fft.ifft(cft1 * cft2)
    return corr

def qcorr(rowcol1, rowcol2):
    npix = rowcol1.size
    corr = ccalc(rowcol1, rowcol2)
    #cft1 = np.fft.fft(rowcol1)
    #cft2 = np.fft.fft(rowcol2)
    #cft2.imag *= -1.0
    #corr = np.fft.ifft(cft1 * cft2)
    nshift = corr.argmax()
    sys.stderr.write("--------------------------------\n")
    if (nshift > 0):
        sys.stderr.write("corr[%d]: %10.5f\n" % (nshift-1, corr[nshift-1]))
    sys.stderr.write("corr[%d]: %10.5f\n" % (nshift+0, corr[nshift+0]))
    if (nshift < npix - 1):
        sys.stderr.write("corr[%d]: %10.5f\n" % (nshift+1, corr[nshift+1]))
    sys.stderr.write("--------------------------------\n")
    if (nshift > 0.5*npix):
        nshift -= npix
    return nshift


## UNPADDED cross-correlation to find pixel shifts:
## Sum across rows to produce average column, along columns for average row:
xsmashed = [np.sum(im, axis=1) for im in bpmask]    # sum each row
ysmashed = [np.sum(im, axis=0) for im in bpmask]    # sum each col

## Cross-correlate to find pixel shifts:
xnudges = [qcorr(ysmashed[0], rr) for rr in ysmashed]
ynudges = [qcorr(xsmashed[0], cc) for cc in xsmashed]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Implement 'tight' padding if requested:
if context.tight_pad:
    tight_ypad = np.abs(ynudges).max()
    tight_xpad = np.abs(xnudges).max()
    yxpads = (tight_ypad, tight_xpad)

#sys.exit(0)
# FIXME: should roll a NaN-padded image so stacking works properly ...
layers = []
for ff,im,dx,dy in zip(context.imlist, imdata, xnudges, ynudges):
    npdata = np.pad(im, yxpads, constant_values=np.nan)
    r2data = np.roll(np.roll(npdata, dx, axis=1), dy, axis=0)
    layers.append(r2data)
    fitsio.write('r' + ff, r2data, clobber=True)
imcube = np.array(layers)
lshape = r2data.shape
del layers
tstop = time.time()
ttook = tstop - tstart
sys.stderr.write("Shifted images in %.3f seconds.\n" % ttook)
sys.stderr.write("Using layer shape: %s\n" % str(lshape))
#sys.exit(0)

## Reshape and stack:
tstart = time.time()
old_shape = imcube.shape
istack = np.zeros_like(r2data) * np.nan
for jj,ii in itt.product(*[range(x) for x in lshape]):
    #sys.stderr.write("jj,ii: %3d,%3d\n" % (jj,ii))
    usepix = imcube[:, jj, ii]
    ignore = np.isnan(usepix) | np.isinf(usepix)
    if not np.all(ignore):
        istack[jj, ii] = np.median(usepix[~ignore])
    #istack[jj, ii] = np.median(imcube[:, jj, ii])
#imtemp = 
fitsio.write(context.output_file, istack, clobber=True)
tstop = time.time()
ttook = tstop - tstart
sys.stderr.write("Shifted images in %.3f seconds.\n" % ttook)


##--------------------------------------------------------------------------##
## Quick FITS I/O:
#data_file = 'image.fits'
#img_vals = pf.getdata(data_file)
#hdr_keys = pf.getheader(data_file)
#img_vals, hdr_keys = pf.getdata(data_file, header=True)
#img_vals, hdr_keys = pf.getdata(data_file, header=True, uint=True) # USHORT
#img_vals, hdr_keys = fitsio.read(data_file, header=True)

#date_obs = hdr_keys['DATE-OBS']
#site_lat = hdr_keys['LATITUDE']
#site_lon = hdr_keys['LONGITUD']

## Initialize time:
#img_time = astt.Time(hdr_keys['DATE-OBS'], scale='utc', format='isot')
#img_time += astt.TimeDelta(0.5 * hdr_keys['EXPTIME'], format='sec')
#jd_image = img_time.jd

## Initialize location:
#observer = ephem.Observer()
#observer.lat = np.radians(site_lat)
#observer.lon = np.radians(site_lon)
#observer.date = img_time.datetime

#pf.writeto('new.fits', img_vals)
#qsave('new.fits', img_vals)
#qsave('new.fits', img_vals, header=hdr_keys)

## Star extraction:
#pse.set_image(img_vals, gain=3.6)
#objlist = pse.analyze(sigthresh=5.0)



######################################################################
# CHANGELOG (sst_xcorr.py):
#---------------------------------------------------------------------
#
#  2021-01-14:
#     -- Increased __version__ to 0.1.0.
#     -- First created sst_xcorr.py.
#

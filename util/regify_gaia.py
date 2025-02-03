#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Create a region file containing Gaia sources for convenient overlay.
#
# Rob Siverd
# Created:       2025-02-03
# Last modified: 2025-02-03
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Modules:
import argparse
#import shutil
#import glob
import gc
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
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
#import scipy.stats as scst
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
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
    Generate a DS9 region file from Gaia CSV data.
    
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
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-i', '--input_file', default=None, required=True,
            help='input Gaia catalog CSV path', type=str)
    iogroup.add_argument('-o', '--output_file', default=None, required=True,
            help='output region file path', type=str)
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    gsgroup = parser.add_argument_group('Source Selection')
    gsgroup.add_argument('--minmag', default=0, required=False,
            help='minimum Gaia mag [def: %(default)s]', type=float)
    gsgroup.add_argument('--maxmag', default=99, required=False,
            help='maximum Gaia mag [def: %(default)s]', type=float)
    gsgroup.add_argument('--magcol', default='phot_g_mean_mag', required=False,
            help='magnitude column to use [def: %(default)s]', type=str)
    gsgroup.add_argument('--racol', default='ra', required=False,
            help='RA column to use [def: %(default)s]', type=str)
    gsgroup.add_argument('--decol', default='dec', required=False,
            help='DE column to use [def: %(default)s]', type=str)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    rgroup = parser.add_argument_group('Region Config')
    rgroup.add_argument('--r1arcsec', default=1.2, required=False, type=float,
            help='annulus inner radius in arcsec [def: %(default)s]')
    rgroup.add_argument('--r2arcsec', default=3.6, required=False, type=float,
            help='annulus outer radius in arcsec [def: %(default)s]')
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
## Derived config:
need_cols = [context.racol, context.decol, context.magcol]
r1deg = context.r1arcsec / 3600.0
r2deg = context.r2arcsec / 3600.0

##--------------------------------------------------------------------------##
## Load the catalog:
sys.stdout.write("Loading %s ... " % context.input_file)
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
gsrcs = pd.read_csv(context.input_file, **pdkwargs)
sys.stdout.write("done.\n")

## Ensure required columns:
for cc in need_cols:
    if not cc in gsrcs.keys():
        sys.stderr.write("\nFile %s is missing required column: %s\n" 
                % (context.input_file, cc))
        sys.stderr.write("Fix and retry ...\n")
        sys.exit(1)
sys.stdout.write("Found all required columns: %s\n" % str(need_cols))

## Some basic stats:
nsrcs = len(gsrcs)
sys.stdout.write("Source count: %d\n" % nsrcs)
dec_diam = gsrcs[context.decol].max() - gsrcs[context.decol].min()
sys.stderr.write("Dec diameter: %.3f deg (max-min)\n" % dec_diam)

## Ensure chosen column is legit:
if context.magcol in gsrcs.keys():
    sys.stdout.write("Found column %s, proceed!\n" % context.magcol)
else:
    sys.stdout.write("\nError: column '%s' not found in file: %s\n\n"
            % (context.magcol, context.input_file))
    sys.exit(1)

magok = (context.minmag <= gsrcs[context.magcol]) & \
        (gsrcs[context.magcol] <= context.maxmag)

gkeep = gsrcs[magok]
nkept = len(gkeep)
sys.stdout.write("Keeping %d of %d sources with %f <= mag <= %f.\n"
        % (nkept, nsrcs, context.minmag, context.maxmag))

##--------------------------------------------------------------------------##
## Write output file:
sys.stderr.write("Saving output to %s ... " % context.output_file)
ntodo = 0
count = 0
with open(context.output_file, 'w') as rfile:
    rfile.write("global color=green")
    for rr,dd in zip(gkeep[context.racol], gkeep[context.decol]):
        count += 1
        rfile.write("fk5; annulus(%10.6fd, %10.6fd, %.5fd, %.5fd)\n" \
                % (rr, dd, r1deg, r2deg))
        if (ntodo > 0) and (count >= ntodo):
            break
sys.stderr.write("done.\n")



######################################################################
# CHANGELOG (regify_gaia.py):
#---------------------------------------------------------------------
#
#  2025-02-03:
#     -- Increased __version__ to 0.1.0.
#     -- First created regify_gaia.py.
#

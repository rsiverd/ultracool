#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This script produces a 'master' list of sources detected on the individual
# frames. Inclusion in this list determines whether or not data for a
# particular target is analyzed in subsequent pipeline stages. Each list
# entry includes at a minimum an identifier (constructed from coordinates)
# and fiducial RA/Dec that can be used for cross-matching.
#
# Rob Siverd
# Created:       2021-04-14
# Last modified: 2021-04-14
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

## Angular math tools:
try:
    import angle
    reload(angle)
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    logger.error("failed to import extended_catalog module!")
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

##--------------------------------------------------------------------------##

## Home-brew robust statistics:
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

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

##--------------------------------------------------------------------------##
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()

## Read ASCII file to list:
def read_column(filename, column=0, delim=' ', strip=True):
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.split(delim)[column] for x in content]
    if strip:
        content = [x.strip() for x in content]
    return content

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
    Generate master list of detected sources for further analysis.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    parser.set_defaults(min_src_hits=3)
    parser.set_defaults(gaia_tol_arcsec=2.0)
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-C', '--cat_list', default=None, required=True,
            help='ASCII file with list of catalog paths in column 1')
    iogroup.add_argument('-o', '--output_file', default=None, required=True,
            help='master source list output file', type=str)
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
##------------------          catalog config (FIXME)        ----------------##
##--------------------------------------------------------------------------##

## RA/DE coordinate keys for various methods:
centroid_colmap = {
        'simple'    :   ('dra', 'dde'),
        'window'    :   ('wdra', 'wdde'),
        'pp_fix'    :   ('ppdra', 'ppdde'),
        }

centroid_method = 'simple'
#centroid_method = 'window'
#centroid_method = 'pp_fix'
_ra_key, _de_key = centroid_colmap[centroid_method]

##--------------------------------------------------------------------------##
##------------------      load listed ExtendedCatalogs      ----------------##
##--------------------------------------------------------------------------##

## Read list of ExCat files:
cat_files = read_column(context.cat_list)

## Load listed ExCat files from disk:
tik = time.time()
cdata_all = []
total = len(cat_files)
for ii,fname in enumerate(cat_files, 1):
    sys.stderr.write("\rLoading catalog %d of %d ... " % (ii, total))
    ccc = ec.ExtendedCatalog()
    ccc.load_from_fits(fname)
    cdata_all.append(ccc)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

## Keep only subset (why would this be needed?):
cdata = [x for x in cdata_all]  # everything

## Useful summary data:
cbcd_name = [x.get_imname() for x in cdata]
#irac_band = np.array([irac_channel_from_filename(x) for x in cbcd_name])
expo_time = np.array([x.get_header()['EXPTIME'] for x in cdata])
n_sources = np.array([len(x.get_catalog()) for x in cdata])

##--------------------------------------------------------------------------##
##-----------------     Reasonably Complete Source List    -----------------##
##--------------------------------------------------------------------------##

#rfile = 'nifty.reg'
#r_sec = 2.0
#with open(rfile, 'w') as f:
#    r_deg = r_sec / 3600.0
#    for rr,dd in zip(every_dra, every_dde):
#        f.write("fk5; circle(%10.6fd, %10.6fd, %.6fd)\n" % (rr,dd, r_deg))

## Create identifiers based on coordinates:
def make_coo_string(dra, dde):
    return '%.7f%+.7f' % (dra, dde)

## Use most populous catalog as 'master' source list (simple):
largest_catalog = cdata[n_sources.argmax()].get_catalog()
n_master = len(largest_catalog)

## Build up source list:
sldata = {'dra':largest_catalog[_ra_key], 'dde':largest_catalog[_de_key]}

sldata['srcid'] = [make_coo_string(rr,dd) for rr,dd in \
                        zip(sldata['dra'], sldata['dde'])]

##--------------------------------------------------------------------------##
##-----------------     Save Resulting Sources to Disk     -----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Saving detections to file ... ")
columns = ('srcid', 'dra', 'dde')
with open(context.output_file, 'w') as of:
    # add header line:
    of.write(' '.join(columns) + '\n')
    # dump names and coordinates:
    for things in zip(*[sldata[cc] for cc in columns]):
        of.write("%s %15.9f %15.9f\n" % things)
sys.stderr.write("done.\n")



######################################################################
# CHANGELOG (10_make_master_source_list.py):
#---------------------------------------------------------------------
#
#  2021-04-14:
#     -- Increased __version__ to 0.1.0.
#     -- First created 10_make_master_source_list.py.
#

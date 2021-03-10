#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Given a list of FITS files with ExtendedCatalog contents, this script will:
# * cross-match EC sources to Gaia and record correspondences
# * extract data points corresponding to target ephemeris and record
#
# The subset of target and neighbor data points are stored in a pickle object
# for further analysis and astrometric fitting.
#
# Rob Siverd
# Created:       2021-03-09
# Last modified: 2021-03-09
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
import ast
import argparse
import pickle
import gc
import os
import sys
import time
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
#import theil_sen as ts
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))


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

## Angular math tools:
try:
    import angle
    reload(angle)
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Easy Gaia source matching:
try:
    import gaia_match
    reload(gaia_match)
    gm = gaia_match.GaiaMatch()
except ImportError:
    logger.error("failed to import gaia_match module!")
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
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

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
    Match extracted sources to Gaia catalog. Separately identify data
    points corresponding to specified target.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    parser.set_defaults(gaia_tol_arcsec=3.0)
    parser.set_defaults(target_param_files=[])
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-C', '--cat_list', default=None, required=True,
            help='ASCII file with list of catalog paths in column 1')
    iogroup.add_argument('-G', '--gaia_csv', default=None, required=True,
            help='CSV file with Gaia source list', type=str)
    iogroup.add_argument('-T', '--targfile', required=False,
            action='append', dest='target_param_files', type=str,
            help='target of interest parameter guess')
    # ------------------------------------------------------------------
    parser.add_argument('-o', '--output_file', default=None, required=True,
            help='pickled results output file', type=str)
    #parser.add_argument('-d', '--dayshift', required=False, default=0,
    #        help='Switch between days (1=tom, 0=today, -1=yest', type=int)
    #parser.add_argument('-e', '--encl', nargs=1, required=False,
    #        help='Encl to make URL for', choices=all_encls, default=all_encls)
    #parser.add_argument('-s', '--site', nargs=1, required=False,
    #        help='Site to make URL for', choices=all_sites, default=all_sites)
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #iogroup = parser.add_argument_group('File I/O')
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #ofgroup = parser.add_argument_group('Output format')
    #fmtparse = ofgroup.add_mutually_exclusive_group()
    #fmtparse.add_argument('--python', required=False, dest='output_mode',
    #        help='Return Python dictionary with results [default]',
    #        default='pydict', action='store_const', const='pydict')
    #bash_var = 'ARRAY_NAME'
    #bash_msg = 'output Bash code snippet (use with eval) to declare '
    #bash_msg += 'an associative array %s containing results' % bash_var
    #fmtparse.add_argument('--bash', required=False, default=None,
    #        help=bash_msg, dest='bash_array', metavar=bash_var)
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

## Abort if no targets given:
if not context.target_param_files:
    logger.error("No targets specified!")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------       load target 5-parameter guess    ----------------##
##--------------------------------------------------------------------------##

## Ensure files exist:
for tpath in context.target_param_files:
    if not os.path.isfile(tpath):
        logger.error("File not found: %s" % tpath)
        sys.exit(1)

## Load parameter files:
targ_pars = []
for tpath in context.target_param_files:
    with open(tpath, 'r') as tp:
        targ_pars.append(ast.literal_eval(tp.read()))


##--------------------------------------------------------------------------##
##------------------          catalog config (FIXME)        ----------------##
##--------------------------------------------------------------------------##

## RA/DE coordinate keys for various methods:
centroid_colmap = {
        'simple'    :   ('dra', 'dde'),
        'window'    :   ('wdra', 'wdde'),
        'pp_fix'    :   ('ppdra', 'ppdde'),
        }

##--------------------------------------------------------------------------##
##------------------       load Gaia sources from CSV       ----------------##
##--------------------------------------------------------------------------##

if context.gaia_csv:
    try:
        logger.info("Loading sources from %s" % context.gaia_csv)
        gm.load_sources_csv(context.gaia_csv)
    except:
        logger.error("failed to load from %s" % context.gaia_csv)
        sys.exit(1)

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
sip_order = np.array([x.get_header()['AP_ORDER'] for x in cdata])
timestamp = astt.Time([x.get_header()['DATE_OBS'] for x in cdata],
        format='isot', scale='utc')
jdutc = timestamp.jd
#jdutc = ['%.6f'%x for x in timestamp.jd]
jd2im = {kk:vv for kk,vv in zip(jdutc, cbcd_name)}
im2jd = {kk:vv for kk,vv in zip(cbcd_name, jdutc)}
im2ex = {kk:vv for kk,vv in zip(cbcd_name, expo_time)}

##--------------------------------------------------------------------------##
## Concatenated list of RA/Dec coordinates:

centroid_method = 'simple'
#centroid_method = 'window'
#centroid_method = 'pp_fix'
_ra_key, _de_key = centroid_colmap[centroid_method]
every_dra = np.concatenate([x._imcat[_ra_key] for x in cdata])
every_dde = np.concatenate([x._imcat[_de_key] for x in cdata])
every_jdutc = np.concatenate([n*[jd] for n,jd in zip(n_sources, jdutc)])
#every_jdutc = np.float_(every_jdutc)
gc.collect()

##--------------------------------------------------------------------------##
##-----------------   Cross-Match to Gaia, Extract Target  -----------------##
##--------------------------------------------------------------------------##

#ntodo = 100
#toler_sec = 3.0
gcounter = {x:0 for x in gm._srcdata.source_id}
n_gaia = len(gm._srcdata)

## First, check which Gaia sources might get used:
tik = time.time()
for ii,(index, gsrc) in enumerate(gm._srcdata.iterrows(), 1):
    sys.stderr.write("\rChecking Gaia source %d of %d ... " % (ii, n_gaia))
    sep_sec = 3600. * angle.dAngSep(gsrc.ra, gsrc.dec, every_dra, every_dde)
    gcounter[gsrc.source_id] += np.sum(sep_sec <= context.gaia_tol_arcsec)
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Collect subset of useful Gaia objects:
need_srcs = 3
useful_ids = [kk for kk,vv in gcounter.items() if vv>need_srcs]
use_gaia = gm._srcdata[gm._srcdata.source_id.isin(useful_ids)]
n_useful = len(use_gaia)
sys.stderr.write("Found possible matches to %d of %d Gaia sources.\n"
        % (n_useful, len(gm._srcdata)))
gc.collect()
if n_useful < 5:
    sys.stderr.write("Gaia match error: found %d useful objects\n" % n_useful)
    sys.exit(1)

## Total Gaia-detected PM in surviving object set:
use_gaia = use_gaia.assign(pmtot=np.hypot(use_gaia.pmra, use_gaia.pmdec))
gaia_pmsrt = use_gaia.sort_values(by='pmtot', ascending=False)

## Robust (non-double-counted) matching of Gaia sources using slimmed list:
sys.stderr.write("Associating catalog objects with Gaia sources:\n")
tik = time.time()
gmatches = {x:[] for x in use_gaia.source_id}
for ci,extcat in enumerate(cdata, 1):
#for ci,extcat in enumerate(cdata[:10], 1):
    #sys.stderr.write("\n------------------------------\n")
    sys.stderr.write("\rChecking image %d of %d ... " % (ci, len(cdata)))
    #ccat = extcat._imcat
    ccat = extcat.get_catalog()
    #cat_jd = jdutc[ci]
    jd_info = {'jd':jdutc[ci-1], 'iname':extcat.get_imname()}
    for gi,(gix, gsrc) in enumerate(use_gaia.iterrows(), 1):
        #sys.stderr.write("Checking Gaia source %d of %d ... " % (gi, n_useful))
        sep_sec = 3600.0 * angle.dAngSep(gsrc.ra, gsrc.dec,
                                    ccat[_ra_key], ccat[_de_key])
                                    #ccat['dra'], ccat['dde'])
        matches = sep_sec <= context.gaia_tol_arcsec
        nhits = np.sum(matches)
        if (nhits == 0):
            #sys.stderr.write("no match!\n")
            continue
        else:
            #sys.stderr.write("got %d match(es).  " % nhits)
            hit_sep = sep_sec[matches]
            hit_cat = ccat[matches]
            sepcheck = 3600.0 * angle.dAngSep(gsrc.ra, gsrc.dec,
                    hit_cat[_ra_key], hit_cat[_de_key])
                    #hit_cat['dra'], hit_cat['dde'])
            #sys.stderr.write("sepcheck: %.4f\n" % sepcheck)
            nearest = hit_sep.argmin()
            m_info = {}
            m_info.update(jd_info)
            m_info['sep'] = hit_sep[nearest]
            m_info['cat'] = hit_cat[nearest]
            #import pdb; pdb.set_trace()
            #sys.exit(1)
            gmatches[gsrc.source_id].append(m_info)
    pass
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Stop here if no Gaia matches:
if not gmatches:
    sys.stderr.write("No matches to Gaia detected! Something is wrong ...\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##-----------------     Extract Target with Ephemeris      -----------------##
##--------------------------------------------------------------------------##




##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##


######################################################################
# CHANGELOG (11_match_and_group_sources.py):
#---------------------------------------------------------------------
#
#  2021-03-09:
#     -- Increased __version__ to 0.1.0.
#     -- First created 11_match_and_group_sources.py.
#

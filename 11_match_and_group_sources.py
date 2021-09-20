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
# Last modified: 2021-09-19
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
__version__ = "0.2.5"

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
from numpy.lib.recfunctions import append_fields
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

### HORIZONS ephemeris tools:
#try:
#    import jpl_eph_helpers
#    reload(jpl_eph_helpers)
#except ImportError:
#    logger.error("failed to import jpl_eph_helpers module!")
#    sys.exit(1)
#eee = jpl_eph_helpers.EphTool()

## Astrometry support routines:
#try:
#    import astrom_test
#    reload(astrom_test)
#except ImportError:
#    logger.error("failed to import astrom_test module!")
#    sys.exit(1)
#af = astrom_test.AstFit()


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
    #parser.set_defaults(gaia_tol_arcsec=3.0)
    parser.set_defaults(min_src_hits=3)
    parser.set_defaults(gaia_tol_arcsec=2.0)
    parser.set_defaults(target_param_files=[])
    parser.set_defaults(min_detections=10)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-C', '--cat_list', default=None, required=True,
            help='ASCII file with list of catalog paths in column 1')
    iogroup.add_argument('-D', '--det_list', default=None, required=True,
            help='ASCII file with master detections list', type=str)
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
##------------------       miscellaneous helper routines    ----------------##
##--------------------------------------------------------------------------##

## Handy routine to append scalar quantities to existing record/recarray:
def scalar_append(recarray, colname, colvalue):
    return append_fields(recarray, colname, np.array([colvalue]), usemask=False)

## How to repackage matched data points:
def repack_matches(match_infos):
    ccat = np.vstack([x['cat'] for x in match_infos])
    jtmp = np.array([x['jd'] for x in match_infos])
    itmp = np.array([x['iname'] for x in match_infos])
    etmp = np.array([x['expt'] for x in match_infos])
    return append_fields(ccat, ('jdutc', 'iname', 'exptime'),
            (jtmp, itmp, etmp), usemask=False)


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

## Validate parameter files (FIXME: should be in a module):
need_keys = ['name', 'ra_deg', 'de_deg', 'pmra_cosdec_asyr', 'pmde_asyr',
                'epoch_jdutc']
for tp in targ_pars:
    if not all([x in tp.keys() for x in need_keys]):
        logger.error("Required keywords missing from %s" % str(tp))
        sys.exit(1)


## Augment parameters with astropy Time object (FIXME: in same module):
for tp in targ_pars:
    tp['astt_epoch'] = astt.Time(tp['epoch_jdutc'], scale='utc', format='jd')
    cosine_dec = np.cos(np.radians(tp['de_deg']))
    tp['pmra_asyr'] = tp['pmra_cosdec_asyr'] / cosine_dec

## Epoch-correct position calculation:
def corrected_targpos(tpars, obstime):
    _asec_per_deg = 3600.0
    dt_yrs = (obstime - tpars['astt_epoch']).jd / 365.25
    fix_ra = tpars['ra_deg'] + (tpars['pmra_asyr'] / _asec_per_deg * dt_yrs)
    fix_de = tpars['de_deg'] + (tpars['pmde_asyr'] / _asec_per_deg * dt_yrs)
    return fix_ra, fix_de


#sys.exit(0)

##--------------------------------------------------------------------------##
##------------------       load master detections list      ----------------##
##--------------------------------------------------------------------------##

if not os.path.isfile(context.det_list):
    logger.error("File not found: %s" % context.det_list)
    sys.stderr.write("Master detection list not found!\n\n")
    sys.exit(1)

sys.stderr.write("Loading detections list ... ")
gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
gftkw.update({'names':True, 'autostrip':True})
try:
    det_data = np.genfromtxt(context.det_list, dtype=None, **gftkw)
except:
    sys.stderr.write("FAILED!\n")
    sys.stderr.write("Missing or empty file?\n")
    sys.stderr.write("--> %s\n" % context.det_list)
    sys.exit(1)
sys.stderr.write("done.\n")

## Ensure sufficient length:
if (len(det_data) < context.min_detections):
    sys.stderr.write("Insufficient detections loaded!\n")
    sys.stderr.write("min_detections: %d\n" % context.min_detections)
    sys.stderr.write("Loaded targets: %d\n" % len(det_data))
    sys.exit(1)

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
##-----------------     Reasonably Complete Source List    -----------------##
##--------------------------------------------------------------------------##

#rfile = 'nifty.reg'
#r_sec = 2.0
#with open(rfile, 'w') as f:
#    r_deg = r_sec / 3600.0
#    for rr,dd in zip(every_dra, every_dde):
#        f.write("fk5; circle(%10.6fd, %10.6fd, %.6fd)\n" % (rr,dd, r_deg))

## Use most populous catalog as 'master' source list (simple):
#master_list = cdata[n_sources.argmax()].get_catalog()
#n_master = len(master_list)
#
### Create identifiers based on coordinates:
#def make_coo_string(dra, dde):
#    return '%.7f%+.7f' % (dra, dde)
#
#master_cc_string = [make_coo_string(rr,dd) for rr,dd in \
#                        zip(master_list[_ra_key], master_list[_de_key])]
#master_ra_string = ['%.7f'%x for x in master_list[_ra_key]]
#master_de_string = ['%+.7f'%x for x in master_list[_de_key]]
#master_cc_string = [x+y for x,y in zip(master_ra_string, master_de_string)]

#sys.stderr.write("Looking for repeating sources ... \n")
#tik = time.time()
#for rtrial,dtrial in zip(every_dra, every_dde):
#    sep_sec = 3600.0 * angle.dAngSep(rtrial, dtrial, every_dra, every_dde)
#    matches = sep_sec <= context.gaia_tol_arcsec
#tok = time.time()
#sys.stderr.write("Each-against-all took %.3f sec\n" % (tok-tik))

## In first pass, count matches to each ID:
sys.stderr.write("Checking which master list sources are used ...\n")
tik = time.time()
n_detect = len(det_data)
scounter = {x:0 for x in det_data['srcid']}
for ii,sdata in enumerate(det_data, 1):
    sys.stderr.write("\rChecking detection %d of %d ... " % (ii, n_detect))
    sep_sec = 3600. * angle.dAngSep(sdata['dra'], sdata['dde'],
                                            every_dra, every_dde)
    scounter[sdata['srcid']] += np.sum(sep_sec <= context.gaia_tol_arcsec)
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Collect subset of useful detections:
useful = np.array([scounter[x]>context.min_src_hits for x in det_data['srcid']])
use_dets = det_data[useful]

## Self-associate sources:
sys.stderr.write("Associating catalog objects:\n")
tik = time.time()
smatches = {x:[] for x in use_dets['srcid']}
for ci,extcat in enumerate(cdata, 1):
    sys.stderr.write("\rChecking image %d of %d ... " % (ci, len(cdata)))
    ccat = extcat.get_catalog()
    jd_info = {'jd':jdutc[ci-1], 'iname':extcat.get_imname(),
            'expt':expo_time[ci-1]}
    this_imname = extcat.get_imname()
    for dobj in use_dets:
        _ra, _de = dobj['dra'], dobj['dde']
        mlid = dobj['srcid']
        sep_sec = 3600.0 * angle.dAngSep(_ra, _de, ccat[_ra_key], ccat[_de_key])
        matches = sep_sec <= context.gaia_tol_arcsec
        nhits = np.sum(matches)
        if (nhits == 0):
            #sys.stderr.write("no match!\n")
            continue
        else:
            #sys.stderr.write("got %d match(es).  " % nhits)
            hit_sep = sep_sec[matches]
            hit_cat = ccat[matches]
            #sepcheck = 3600.0 * angle.dAngSep(_ra, _de,
            #        hit_cat[_ra_key], hit_cat[_de_key])
            #sys.stderr.write("sepcheck: %.4f\n" % sepcheck)
            nearest = hit_sep.argmin()
            m_info = {}
            #sep_asec = np.array([hit_sep[nearest]])
            #cat_data = np.atleast_1d(hit_cat[nearest])
            #cat_data = scalar_append(cat_data, 'sep', hit_sep[nearest])
            #cat_data = scalar_append(cat_data, 'iname', this_imname)
            #cat_data = append_fields(cat_data, 'sep', sep_asec, usemask=False)
            m_info.update(jd_info)
            m_info['sep'] = hit_sep[nearest]
            m_info['cat'] = hit_cat[nearest]
            #m_info['cat'] = cat_data
            #import pdb; pdb.set_trace()
            #sys.exit(1)
            smatches[mlid].append(m_info)
    pass
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Collect data sets by Gaia source for analysis:
stargets = {}
for ii,sid in enumerate(smatches.keys(), 1):
    sys.stderr.write("\rGathering ML source %d of %d ..." % (ii, len(use_dets)))
    stargets[sid] = repack_matches(smatches[sid])
sys.stderr.write("done.\n")
stg_npts = {ss:len(cc) for ss,cc in stargets.items()}
#npts_100 = [gg for gg,nn in gtg_npts.items() if nn>100]
#gid_list = sorted(gtg_npts.keys())

#sys.exit(0)

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
#need_srcs = 3
useful_ids = [kk for kk,vv in gcounter.items() if vv>context.min_src_hits]
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
    jd_info = {'jd':jdutc[ci-1], 'iname':extcat.get_imname(),
            'expt':expo_time[ci-1]}
    this_imname = extcat.get_imname()
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
            #sep_asec = np.array([hit_sep[nearest]])
            #cat_data = np.atleast_1d(hit_cat[nearest])
            #cat_data = scalar_append(cat_data, 'sep', hit_sep[nearest])
            #cat_data = scalar_append(cat_data, 'iname', this_imname)
            #cat_data = append_fields(cat_data, 'sep', sep_asec, usemask=False)
            m_info.update(jd_info)
            m_info['sep'] = hit_sep[nearest]
            m_info['cat'] = hit_cat[nearest]
            #m_info['cat'] = cat_data
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
##-----------------     Extract Targets by Ephemerides     -----------------##
##--------------------------------------------------------------------------##

#for targ in targ_pars:
#    tname = targ['name']
#    sys.stderr.write("Extracting moving target '%s' ... \n" % tname)
    
tpars = targ_pars[0]
sys.stderr.write("Extracting %s data ... " % tpars['name'])
tik = time.time()
tgt_data = []
tgt_tol_asec = 3.
for ci,extcat in enumerate(cdata, 1):
    ccat = extcat.get_catalog()
    jd_info = {'jd':jdutc[ci-1], 'iname':extcat.get_imname(),
            'expt':expo_time[ci-1]}
    # FIXME:
    # date_obs = extcat.get_header()['DATE_OBS']
    obsdate = astt.Time(extcat.get_header()['DATE_OBS'], scale='utc')
    #elapsed_yr = (jd_info['jd'] - j2000_epoch.utc.jd) / 365.25
    #elapsed_yr = (jd_info['jd'] - trent_epoch_jd) / 365.25
    #_ra, _de = targpos(elapsed_yr, trent_pars)
    _ra, _de = corrected_targpos(tpars, obsdate)
    sep_sec = 3600. * angle.dAngSep(_ra, _de, ccat[_ra_key], ccat[_de_key])
    matches = sep_sec <= tgt_tol_asec
    nhits = np.sum(matches)

    for match in ccat[matches]:
        m_info = {}
        m_info.update(jd_info)
        m_info['cat'] = match
        tgt_data.append(m_info)
    #nhits = np.sum(which)
    #if (nhits > 1):
    #    sys.stderr.write("Found multiple in-box sources in catalog %d!\n" % ci)
    #    sys.exit(1)
    #if np.any(which):
    #    m_info = {}
    #    m_info.update(jd_info)
    #    m_info['cat'] = ccat[which]
    #    tgt_data.append(m_info)
    pass
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Stop here in case of matching failure(s):
if not tgt_data:
    sys.stderr.write("No matches for target data??  Please investigate ...\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------      Repackage Results for Export      ----------------##
##--------------------------------------------------------------------------##

## How to repackage matched data points:
#def repack_matches(match_infos):
#    ccat = np.vstack([x['cat'] for x in match_infos])
#    jtmp = np.array([x['jd'] for x in match_infos])
#    itmp = np.array([x['iname'] for x in match_infos])
#    return append_fields(ccat, ('jdutc', 'iname'), (jtmp, itmp), usemask=False)

## Collect and export target data set:
tgt_ccat = repack_matches(tgt_data)

## Collect data sets by Gaia source for analysis:
gtargets = {}
for ii,gid in enumerate(gmatches.keys(), 1):
    sys.stderr.write("\rGathering gaia source %d of %d ..." % (ii, n_useful))
    gtargets[gid] = repack_matches(gmatches[gid])
sys.stderr.write("done.\n")
gtg_npts = {gg:len(cc) for gg,cc in gtargets.items()}
npts_100 = [gg for gg,nn in gtg_npts.items() if nn>100]
gid_list = sorted(gtg_npts.keys())

##--------------------------------------------------------------------------##
##------------------    Export Grouped Data for Analysis    ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Saving groupings to file: %s\n" % context.output_file)
with open(context.output_file, 'wb') as of:
    pickle.dump((tgt_ccat, stargets, gtargets), of)

sys.exit(0)

##--------------------------------------------------------------------------##
## Saving the match count:
save_count = False
if save_count:
    savefile = os.path.basename(context.cat_list) + '.npts'
    sys.stderr.write("Saving data point count to: %s\n" % savefile)
    with open(savefile, 'w') as f:
        for gg in gid_list:
            f.write("%d %d\n" % (gg, gtg_npts[gg]))
            pass
        pass

##--------------------------------------------------------------------------##
## Gaia source data for matched stars:
gaia_full = gm._srcdata
gaia_hits = gaia_full[gaia_full['source_id'].isin(npts_100)]

## Look at star brightnesses:

gaia_bands = ['g', 'bp', 'rp']
bp_columns = ['phot_%s_mean_mag'%b for b in gaia_bands]
#for bb in gaia_bands:

##--------------------------------------------------------------------------##
## Dictify gaia astrometric parameters:
def dictify_gaia(params):
    use_keys = ['ref_epoch', 'ra', 'ra_error', 'dec', 'dec_error', 
            'parallax', 'parallax_error', 'parallax_over_error',
            'pmra', 'pmra_error', 'pmdec', 'pmdec_error']
    return {kk:np.atleast_1d(params[kk])[0] for kk in use_keys}

## Evaluate gaia ephem:
def eval_gaia(jdtdb, pars):
    ref_epoch = pars['ref_epoch']
    if (ref_epoch != 2015.5):
        sys.stderr.write("Unhandled Gaia epoch: %f\n" % ref_epoch)
        raise
    gpoch = astt.Time(2457206.375, format='jd', scale='tcb')
    tdiff = astt.Time(jdtdb, format='jd', scale='tdb') - gpoch
    years = tdiff.jd / 365.25
    use_pmra = pars['pmra'] / np.cos(np.radians(pars['dec']))
    calc_ra = pars['ra'] + (years * use_pmra / 3.6e6)
    calc_de = pars['dec'] + (years * pars['pmdec'] / 3.6e6)
    return (calc_ra, calc_de)

## Save data for external plotting:
save_dir = 'pdata'
#if not os.path.isdir(save_dir):
#    os.mkdir(save_dir)

## This third version allows the user to specify keyword choice/order if
## desired but defaults to the keys provided in the first dictionary.
def recarray_from_dicts(list_of_dicts, use_keys=None):
    keys = use_keys if use_keys else list_of_dicts[0].keys()
    data = [np.array([d[k] for d in list_of_dicts]) for k in keys]
    return np.core.records.fromarrays(data, names=','.join(keys))

## Per-image storage of x,y and residuals:
resid_data = {cc.get_imname():[] for cc in cdata}

j2000_epoch = astt.Time('2000-01-01T12:00:00', scale='tt', format='isot')
for gid in npts_100:
    sys.stderr.write("gid: %d\n" % gid)
    gdata = gtargets[gid]   # all measurements for this gaia source
    tjd = astt.Time(gdata['jdtdb'], format='jd', scale='tdb')
    gpars = use_gaia[use_gaia.source_id == gid]
    dpars = dictify_gaia(gpars)
    gaia_ra, gaia_de = eval_gaia(tjd, dpars)
    cos_dec = np.cos(np.radians(dpars['dec']))
    delta_ra_mas = 3.6e6 * (gdata['wdra'] - gaia_ra) * cos_dec
    delta_de_mas = 3.6e6 * (gdata['wdde'] - gaia_de)
    year = 2000. + ((tjd.tt.jd - j2000_epoch.tt.jd) / 365.25)
    for iname,xpix,ypix,rmiss,dmiss in zip(gdata['iname'],
            gdata['x'], gdata['y'], delta_ra_mas, delta_de_mas):
        resid_data[iname].append({'x':xpix, 'y':ypix,
            'ra_err':rmiss, 'de_err':dmiss})

## Promote dictionary lists to recarrays:
resid_data = {kk:recarray_from_dicts(vv) for kk,vv in resid_data.items()}

##--------------------------------------------------------------------------##
#import matplotlib.pyplot as plt
#
##fig = plt.figure(3)
##fig.clf()
#plt.gcf().clf()
#fig, axs = plt.subplots(2, 3, sharex=True, num=3)
#for ii,cc in enumerate(bp_columns):
#    axs[0, ii].hist(gaia_hits[cc], bins=20, range=(10,25))
#for ii,cc in enumerate(bp_columns):
#    axs[1, ii].hist(gaia_full[cc], bins=20, range=(10,25))

##--------------------------------------------------------------------------##
sys.exit(0)
import matplotlib.pyplot as plt

fig = plt.figure(3)
total = len(resid_data)
for ii,(imname,iresid) in enumerate(resid_data.items(), 1):
    sys.stderr.write("%s (image %d of %d) ...   \n" % (imname, ii, total))
    fig.clf()
    ax1 = fig.add_subplot(111, aspect='equal')
    ax1.grid(True)
    #ax1.scatter(iresid['x'], iresid['y'])
    ax1.quiver(iresid['x'], iresid['y'], iresid['ra_err'], iresid['de_err'])
    ax1.set_xlim(0, 260)
    ax1.set_ylim(0, 260)
    fig.tight_layout()
    plt.draw()
    sys.stderr.write("press ENTER to continue ...\n")
    response = input()
    #break


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

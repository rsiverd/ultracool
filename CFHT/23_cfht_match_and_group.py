#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Bespoke match and group sources for CFHT/WIRCam data.
#
# Rob Siverd
# Created:       2023-07-27
# Last modified: 2023-07-27
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
__version__ = "0.0.1"

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
import numpy as np
from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

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
        'simple'    :   (    'dra',     'dde'),
        'window'    :   (   'wdra',    'wdde'),
        'pp_fix'    :   (  'ppdra',   'ppdde'),
        'tpcalc'    :   ('calc_ra', 'calc_de'),
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

## Summary data (used later):
jdtdb     = np.array([x._imeta['jdtdb'] for x in cdata])
obs_times = astt.Time(jdtdb, format='jd', scale='tdb')
jdutc     = obs_times.utc.jd
expo_time = np.array([x.get_header()['EXPTIME'] for x in cdata])

## Concatenated list of RA/Dec coordinates:
centroid_method = 'tpcalc'
_ra_key, _de_key = centroid_colmap[centroid_method]
every_dra = np.concatenate([x._imcat[_ra_key] for x in cdata])
every_dde = np.concatenate([x._imcat[_de_key] for x in cdata])
every_idx = np.arange(len(every_dra))
gc.collect()

##--------------------------------------------------------------------------##
##-----------------     Reasonably Complete Source List    -----------------##
##--------------------------------------------------------------------------##

## Multi-stage cross-matching. This routine checks a test RA/DE position against
## a pair of RA/DE vectors. It returns a list of indexes to elements in the
## vectors that match to within the specified tolerance.
## All RA/DE positions are expected to be in DEGREES.
def multi_step_match(test_ra, test_de, ra_vec, de_vec, tol_deg):
    orig_idx = np.arange(len(ra_vec))
    de_which = np.abs(test_de - de_vec) < tol_deg
    near_dra =   ra_vec[de_which]
    near_dde =   de_vec[de_which]
    near_idx = orig_idx[de_which]
    asep_deg = angle.dAngSep(test_ra, test_de, near_dra, near_dde)
    is_match = asep_deg < tol_deg
    return near_idx[is_match], asep_deg[is_match]

## In first pass, count matches to each ID:
sys.stderr.write("Checking which master list sources are used ...\n")
tik = time.time()
n_detect = len(det_data)
scounter = {x:0 for x in det_data['srcid']}
tol_deg  = context.gaia_tol_arcsec / 3600.0
for ii,sdata in enumerate(det_data, 1):
    sys.stderr.write("\rChecking detection %d of %d ... " % (ii, n_detect))
    #with warnings.catch_warnings():
    #    warnings.simplefilter('error')
    #    sep_sec = 3600. * angle.dAngSep(sdata['dra'], sdata['dde'],
    #                                            every_dra, every_dde)
    #    scounter[sdata['srcid']] += np.sum(sep_sec <= context.gaia_tol_arcsec)
    # Brute-force method:
    #sep_sec = 3600. * angle.dAngSep(sdata['dra'], sdata['dde'],
    #                                        every_dra, every_dde)

    ## Check in Dec first:
    #de_which = np.abs(sdata['dde'] - every_dde) < tol_deg
    #near_dra = every_dra[de_which]
    #near_dde = every_dde[de_which]
    #sep_sec = 3600. * angle.dAngSep(sdata['dra'], sdata['dde'],
    #                                        near_dra, near_dde)
    #scounter[sdata['srcid']] += np.sum(sep_sec <= context.gaia_tol_arcsec)

    # Use multi-step matcher:
    hits, seps = multi_step_match(sdata['dra'], sdata['dde'], 
            every_dra, every_dde, tol_deg)
    scounter[sdata['srcid']] += hits.size
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Collect subset of useful detections:
useful = np.array([scounter[x]>context.min_src_hits for x in det_data['srcid']])
use_dets = det_data[useful]

## Self-associate sources:
sys.stderr.write("Associating catalog objects:\n")
tik = time.time()
tol_deg  = context.gaia_tol_arcsec / 3600.0
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
        #sep_sec = 3600.0 * angle.dAngSep(_ra, _de, ccat[_ra_key], ccat[_de_key])
        #matches = sep_sec <= context.gaia_tol_arcsec
        #nhits = np.sum(matches)
        matches, sep_deg = multi_step_match(_ra, _de,
                ccat[_ra_key], ccat[_de_key], tol_deg)
        #sep_sec = 3600.0 * sep_deg
        nhits = matches.size
        if (nhits == 0):
            #sys.stderr.write("no match!\n")
            continue
        else:
            #sys.stderr.write("got %d match(es).  " % nhits)
            #hit_sep = sep_sec[matches]
            hit_sep = sep_deg * 3600.0
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

##--------------------------------------------------------------------------##
##-----------------     Extract Targets by Ephemerides     -----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Beginning moving-target extraction!\n")
n_moving = len(targ_pars)
sys.stderr.write("Moving target count: %d\n" % n_moving)
#extraction of moving targets \n")

tik = time.time()
#tgt_data = []
tgt_tol_asec = 3.0
tgt_tol_asec = 2.0

m_matches = {}
m_targets = {}
warn_missing = True

for ti,tpars in enumerate(targ_pars, 1):
    mover_name = tpars['name']
    sys.stderr.write("Extracting %s (target %d of %d) ...\n"
            % (mover_name, ti, n_moving))

    tgt_data = []
    for ci,extcat in enumerate(cdata, 0):
        ccat = extcat.get_catalog()
        im_name = extcat.get_imname()
        jd_info = {'jd':jdutc[ci], 'iname':im_name, 'expt':expo_time[ci]}
        obsdate = obs_times[ci]
        _ra, _de = corrected_targpos(tpars, obsdate)
        sep_sec = 3600. * angle.dAngSep(_ra, _de, ccat[_ra_key], ccat[_de_key])
        matches = (sep_sec <= tgt_tol_asec)
        nhits = np.sum(matches)
        if nhits > 0:
            for match in ccat[matches]:
                m_info = {}
                m_info.update(jd_info)
                m_info['cat'] = match
                tgt_data.append(m_info)
        else:
            if warn_missing:
                sys.stderr.write("No detection from %s ...\n" % im_name)
        pass

    m_matches[mover_name] = tgt_data
    m_targets[mover_name] = repack_matches(tgt_data)
    pass
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

##--------------------------------------------------------------------------##
##------------------      Repackage Results for Export      ----------------##
##--------------------------------------------------------------------------##

s_keys = sorted(list(stargets.keys()))
s_npts = np.array([len(stargets[kk]) for kk in s_keys])

## Create a master list of targets to keep:
need_targ_pts = 100
keep_targets = {}
for kk,vv in stargets.items():
    npts = len(vv)
    sys.stderr.write("Target '%s' has %d data points.\n" % (kk, npts))
    if npts >= need_targ_pts:
        keep_targets[kk] = vv
    pass

keep_targets.update(m_targets)

##--------------------------------------------------------------------------##
##------------------    Export Grouped Data for Analysis    ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Saving groupings to file: %s\n" % context.output_file)
with open(context.output_file, 'wb') as of:
    #pickle.dump((tgt_ccat, stargets, gtargets), of)
    pickle.dump(keep_targets, of)

sys.exit(0)


######################################################################
# CHANGELOG (23_cfht_match_and_group.py):
#---------------------------------------------------------------------
#
#  2023-07-27:
#     -- Increased __version__ to 0.0.1.
#     -- First created 23_cfht_match_and_group.py.
#

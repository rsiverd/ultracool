#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract and save extended object catalogs from the specified data and
# uncertainty images. This version of the script jointly analyzes all
# images from a specific AOR/channel to enable more sophisticated
# analysis.
#
# Rob Siverd
# Created:       2023-01-25
# Last modified: 2023-02-20
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
import shutil
#import resource
#import signal
import glob
import copy
#import gc
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
import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Angular math tools:
try:
    import angle
    reload(angle)
except ImportError:
    logger.error("failed to import angle module!")
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
## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers

## Spitzer pipeline cross-correlation:
try:
    import spitz_xcorr_stacking
    reload(spitz_xcorr_stacking)
except ImportError:
    logger.error("failed to import spitz_xcor_stacking module!")
    sys.exit(1)
sxc = spitz_xcorr_stacking.SpitzerXCorr()

## Catalog pruning helpers:
try:
    import catalog_tools
    reload(catalog_tools)
except ImportError:
    logger.error("failed to import catalog_tools module!")
    sys.exit(1)
xcp = catalog_tools.XCorrPruner()

## Spitzer star detection routine:
try:
    import spitz_extract
    reload(spitz_extract)
    spf = spitz_extract.SpitzFind()
except ImportError:
    logger.error("spitz_extract module not found!")
    sys.exit(1)

## Hybrid stack+individual position calculator:
try:
    import spitz_stack_astrom
    reload(spitz_stack_astrom)
    ha = spitz_stack_astrom.HybridAstrom()
except ImportError:
    logger.error("failed to import spitz_stack_astrom module!")
    sys.exit(1)

## HORIZONS ephemeris tools:
try:
    import jpl_eph_helpers
    reload(jpl_eph_helpers)
except ImportError:
    logger.error("failed to import jpl_eph_helpers module!")
    sys.exit(1)
eee = jpl_eph_helpers.EphTool()

## Adam Kraus polynomial routines:
try:
    import akspoly
    reload(akspoly)
    akp = akspoly.AKSPoly()
except ImportError:
    logger.error("failed to import akspoly module!")
    sys.exit(1)

##--------------------------------------------------------------------------##
## Fast FITS I/O:
try:
    import fitsio
except ImportError:
    logger.error("fitsio module not found!  Install and retry.")
    sys.stderr.write("\nError: fitsio module not found!\n")
    sys.exit(1)

## Save FITS image with clobber (fitsio):
def qsave(iname, idata, header=None, **kwargs):
    this_func = sys._getframe().f_code.co_name
    parent_func = sys._getframe(1).f_code.co_name
    sys.stderr.write("Writing to '%s' ... " % iname)
    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

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
    Extract catalogs from the listed Spitzer data/uncertainty images.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt)
                          #formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(imtype=None) #'cbcd') #'clean')
    #parser.set_defaults(sigthresh=3.0)
    parser.set_defaults(sigthresh=2.0)
    parser.set_defaults(skip_existing=True)
    parser.set_defaults(save_registered=True)
    parser.set_defaults(gaia_tol_arcsec=2.0)
    parser.set_defaults(min_gaia_matches=10)
    #parser.set_defaults(save_reg_subdir=None)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('--overwrite', required=False, dest='skip_existing',
            action='store_false', help='overwrite existing catalogs')
    #iogroup.add_argument('-E', '--ephem_data', default=None, required=True,
    #        help='CSV file with SST ephemeris data', type=str)
    iogroup.add_argument('-G', '--gaia_csv', default=None, required=True,
            help='CSV file with Gaia source list', type=str)
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=False,
            help='where to save extended catalog outputs', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find CBCD images')
    imtype = iogroup.add_mutually_exclusive_group()
    #imtype.add_argument('--cbcd', required=False, action='store_const',
    #        dest='imtype', const='cbcd', help='use cbcd images')
    imtype.add_argument('--hcfix', required=False, action='store_const',
            dest='imtype', const='hcfix', help='use hcfix images')
    imtype.add_argument('--clean', required=False, action='store_const',
            dest='imtype', const='clean', help='use clean images')
    imtype.add_argument('--nudge', required=False, action='store_const',
            dest='imtype', const='nudge', help='use nudge images')
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

    # Unless otherwise specified, output goes into input folder:
    if not context.output_folder:
        context.output_folder = context.input_folder

    # Ensure an image type is selected:
    if not context.imtype:
        sys.stderr.write("\nNo image type selected!\n\n")
        sys.exit(1)

    ## Use imtype-specific folder for registered file output:
    #if not context.save_reg_subdir:
    #    context.save_reg_subdir = 'aligned_%s' % context.imtype

if context.gaia_csv:
    try:
        logger.info("Loading sources from %s" % context.gaia_csv)
        gm.load_sources_csv(context.gaia_csv)
    except:
        logger.error("failed to load from %s" % context.gaia_csv)
        sys.exit(1)

##--------------------------------------------------------------------------##
##------------------      Gaia-based offset calculator      ----------------##
##--------------------------------------------------------------------------##

def find_gaia_matches(stars, tol_arcsec, ra_col='wdra', de_col='wdde', 
        xx_col='xdw', yy_col='ydw'):
    tol_deg = tol_arcsec / 3600.0
    matches = []
    for target in stars:
        sra, sde = target[ra_col], target[de_col]
        sxx, syy = target[xx_col], target[yy_col]
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            #sys.stderr.write("got one!\n")
            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
            #matches.append((sra, sde, *gcoords))
            matches.append((sxx, syy, *gcoords))
            pass
        pass
    return matches
    #have_ra, have_de, gaia_ra, gaia_de = np.array(matches).T

def compute_offset(match_list):
    have_ra, have_de, gaia_ra, gaia_de = np.array(match_list).T
    ra_diffs = gaia_ra - have_ra
    de_diffs = gaia_de - have_de
    delta_ra, delta_ra_sig = rs.calc_ls_med_IQR(ra_diffs)
    delta_de, delta_de_sig = rs.calc_ls_med_IQR(de_diffs)
    ratio_ra = delta_ra / delta_ra_sig
    ratio_de = delta_de / delta_de_sig
    sys.stderr.write("delta_ra: %8.3f (%6.3f)\n" % (3600.*delta_ra, ratio_ra))
    sys.stderr.write("delta_de: %8.3f (%6.3f)\n" % (3600.*delta_de, ratio_de))
    return {'gradelta':delta_ra,    'grasigma':delta_ra_sig,
            'gdedelta':delta_de,    'gdesigma':delta_de_sig,}
    #return (delta_ra, delta_ra_sig, delta_de, delta_de_sig)

##--------------------------------------------------------------------------##
##------------------         Make Input Image List          ----------------##
##--------------------------------------------------------------------------##

tstart = time.time()
sys.stderr.write("Listing %s frames ... " % context.imtype) 
#im_wildpath = 'SPITZ*%s.fits' % context.imtype
#im_wildcard = os.path.join(context.input_folder, 'SPIT*'
#_img_types = ['cbcd', 'clean', 'cbunc']
#_type_suff = dict([(x, x+'.fits') for x in _im_types])
#img_list = {}
#for imsuff in suffixes:
#    wpath = '%s/SPITZ*%s.fits' % (context.input_folder, imsuff)
#    img_list[imsuff] = sorted(glob.glob(os.path.join(context.
#img_files = sorted(glob.glob(os.path.join(context.input_folder, im_wildpath)))

if not os.path.isdir(context.input_folder):
    sys.stderr.write("error: folder not found:\n")
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

if context.walk:
    img_files = sfh.get_files_walk(context.input_folder, flavor=context.imtype)
else:
    img_files = sfh.get_files_single(context.input_folder, flavor=context.imtype)
sys.stderr.write("done.\n")

## Abort in case of no input:
if not img_files:
    sys.stderr.write("No input (%s) files found in folder:\n" % context.imtype)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

n_images = len(img_files)

## List of uncertainty frames (warn if any missing):
#unc_files = [x.replace(context.imtype, 'cbunc') for x in img_files]
#sys.stderr.write("Checking error-images ... ") 
#have_unc = [os.path.isfile(x) for x in unc_files]
#if not all(have_unc):
#    sys.stderr.write("WARNING: some uncertainty frames missing!\n")
#else:
#    sys.stderr.write("done.\n") 

##--------------------------------------------------------------------------##
##------------------         Load SST Ephemeris Data        ----------------##
##--------------------------------------------------------------------------##

### Ephemeris data file must exist:
#if not context.ephem_data:
#    logger.error("context.ephem_data not set?!?!")
#    sys.exit(1)
#if not os.path.isfile(context.ephem_data):
#    logger.error("Ephemeris file not found: %s" % context.ephem_data)
#    sys.exit(1)
#
### Load ephemeris data:
#eee.load(context.ephem_data)

##--------------------------------------------------------------------------##
##------------------       Unique AOR/Channel Combos        ----------------##
##--------------------------------------------------------------------------##

unique_tags = sorted(list(set([sfh.get_irac_aor_tag(x) for x in img_files])))
images_by_tag = {x:[] for x in unique_tags}
for ii in img_files:
    images_by_tag[sfh.get_irac_aor_tag(ii)].append(ii)


##--------------------------------------------------------------------------##
##------------------        Diagnostic Region Files         ----------------##
##--------------------------------------------------------------------------##

def regify_excat_pix(data, rpath, win=False, rr=2.0):
    colnames = ('wx', 'wy') if win else ('x', 'y')
    xpix, ypix = [data[x] for x in colnames]
    with open(rpath, 'w') as rfile:
        for xx,yy in zip(xpix, ypix):
            rfile.write("image; circle(%8.3f, %8.3f, %8.3f)\n" % (xx, yy, rr))
    return

##--------------------------------------------------------------------------##
##------------------      ExtendedCatalog Ephem Format      ----------------##
##--------------------------------------------------------------------------##

#def reformat_ephem(edata):


##--------------------------------------------------------------------------##
##------------------         Stack/Image Comparison         ----------------##
##--------------------------------------------------------------------------##

#def xcheck(idata, sdata):
#    nstack = len(sdata)
#    nimage = len(idata)
#    sys.stderr.write("nstack: %d\n" % nstack)
#    sys.stderr.write("nimage: %d\n" % nimage)
#    return

def evaluator(params, imkeys, matches):
    cdmat = params[:4]
    crval = params[4:].reshape(-1, 2)

    # compute test RA/DE for the given X,Y positions from each image. Tally
    # angular separations between catalog and computed positions:
    deltas = []
    for ii,kk in enumerate(imkeys):
        cv1, cv2 = crval[ii]
        #sys.stderr.write("Checking %s (%f, %f) ... \n" % (kk, cv1, cv2))
        cxx, cyy, gra, gde = (np.array(x) for x in zip(*matches[kk]))
        cra, cde = akspoly.xycd2radec(cdmat, cxx, cyy, cv1, cv2)
        deltas += angle.dAngSep(cra, cde, gra, gde).tolist()

    # compute figure of merit:
    return np.sum(deltas)       # total absolute separations

##--------------------------------------------------------------------------##
##------------------           Process All Images           ----------------##
##--------------------------------------------------------------------------##

## Catalog object:
ccc = ec.ExtendedCatalog()


ntodo = 0
nproc = 0
ntotal = len(img_files)
min_sobj = 10       # bark if fewer than this many found in stack

skip_stuff = False

#context.save_registered = False
#context.skip_existing = False

## Reduce bright pixel threshold:
#sxc.set_bp_thresh(10.0)
#sxc.set_bp_thresh(5.0)
sxc.set_bp_thresh(10.0)
#sxc.set_vlevel(10)
sxc.set_roi_rfrac(0.90)
sxc.set_roi_rfrac(2.00)
#sys.exit(0)

## How to save results:
save_cols = ['iname', 'aor_tag', 'jdtdb', 'padeg',
    'old_crval1', 'old_crval2', 'old_cd11', 'old_cd12', 'old_cd21', 'old_cd22',
    'new_crval1', 'new_crval2', 'new_cd11', 'new_cd12', 'new_cd21', 'new_cd22']

## Check whether a header row is needed:
def file_is_empty(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            content = f.readlines()
        if len(content) >= 1:
            return False
    return True

## Stats file updater:
def dump_to_file(stat_savefile, cols, data, delimiter=','):
    if file_is_empty(stat_savefile):
        sys.stderr.write("No stats file!\n")
        with open(stat_savefile, 'w') as ff:
            ff.write("%s\n" % delimiter.join(cols))
    with open(stat_savefile, 'a') as ff:
        ff.write("%s\n" % delimiter.join([str(x) for x in data]))

save_file = 'results_joint.csv'

## WCS keywords to fetch from the header:
cdm_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
wcs_keys = ['CRVAL1', 'CRVAL2'] + cdm_keys

#for aor_tag,tag_files in images_by_tag.items():
for aor_tag in unique_tags:
    sys.stderr.write("\n\nProcessing images from %s ...\n" % aor_tag)
    tag_files = images_by_tag[aor_tag]
    n_tagged  = len(tag_files)

    if n_tagged < 2:
        sys.stderr.write("WARNING: only %d images with tag %s\n"
                % (n_tagged, aor_tag))
        sys.stderr.write("This case is not currently handled ...\n")
        sys.exit(1)

    # File/folder paths:
    aor_dir = os.path.dirname(tag_files[0])
    stack_ibase = '%s_%s_stack.fits' % (aor_tag, context.imtype)
    stack_cbase = '%s_%s_stack.fcat' % (aor_tag, context.imtype)
    #medze_ibase = '%s_%s_medze.fits' % (aor_tag, context.imtype)
    stack_ipath = os.path.join(aor_dir, stack_ibase)
    stack_cpath = os.path.join(aor_dir, stack_cbase)
    #medze_ipath = os.path.join(aor_dir, medze_ibase)
    #sys.stderr.write("stack_ibase: %s\n" % stack_ibase)

    #sys.stderr.write("As of this point ...\n")
    #sys.stderr.write("sxc._roi_rfrac: %.5f\n" % sxc._roi_rfrac)


    tag_pcat_files = [x+'.pcat' for x in tag_files]
    if not all([os.path.isfile(x) for x in tag_pcat_files]):
        sys.stderr.write("Missing some/all pcat files! Run 07 first ...\n")
        sys.exit(1)

    # Load all the pcat files:
    aor_cats = {}
    aor_hdrs = {}
    for tpf in tag_pcat_files:
        cbase = os.path.basename(tpf)
        sys.stderr.write("Loading %s ...\n" % tpf)
        ccc.load_from_fits(tpf)
        stars = ccc.get_catalog()
        #match_list = find_gaia_matches(stars, context.gaia_tol_arcsec)
        aor_cats[cbase] = find_gaia_matches(stars, context.gaia_tol_arcsec)
        aor_hdrs[cbase] = ccc.get_header().copy()


    # This order will be used for array indexing:
    imkeys = list(sorted(aor_cats.keys()))


    wstuff = {}
    paravg = {}
    for kk in wcs_keys:
        wstuff[kk] = {ii:hh[kk] for ii,hh in aor_hdrs.items()}
        paravg[kk] = np.average(list(wstuff[kk].values()))

    param_guess = [paravg['CD1_1'], paravg['CD1_2'],
                    paravg['CD2_1'], paravg['CD2_2']]
    for kk in imkeys:
        param_guess.append(wstuff['CRVAL1'][kk])
        param_guess.append(wstuff['CRVAL2'][kk])

    param_guess = np.array(param_guess)
    initial_guess = param_guess.copy()

    fitme = partial(evaluator, imkeys=imkeys, matches=aor_cats)

    sys.stderr.write("Fitting global CD matrix and individual CRVALs ...\n")
    bestfit_pars = opti.fmin(fitme, param_guess)
    bestfit_cdmat = bestfit_pars[:4]
    bestfit_crval = bestfit_pars[4:].reshape(-1, 2)
    guessed_cdmat = param_guess[:4]
    guessed_crval = param_guess[4:].reshape(-1, 2)

    for ii,kk in enumerate(imkeys):
        content = [kk, aor_tag, aor_hdrs[kk]['OBS_TIME'], aor_hdrs[kk]['PA']]
        content += guessed_crval[ii].tolist() + guessed_cdmat.tolist()
        content += bestfit_crval[ii].tolist() + bestfit_cdmat.tolist()
        dump_to_file(save_file, save_cols, content)

    #sys.exit(0)

    if (ntodo > 0) and (nproc >= ntodo):
        break

tstop = time.time()
ttook = tstop - tstart
sys.stderr.write("Extraction completed in %.3f seconds.\n" % ttook)


#import astropy.io.fits as pf
#
#imra = np.array([hh['CRVAL1'] for hh in sxc._im_hdrs])
#imde = np.array([hh['CRVAL2'] for hh in sxc._im_hdrs])
#
##sys.stderr.write("\n\n\n")
##sys.stderr.write("sxc.shift_and_stack(tag_files)\n")
##result = sxc.shift_and_stack(tag_files)
#sys.exit(0)
#
#layers = sxc.pad_and_shift(sxc._im_data, sxc._x_shifts, sxc._y_shifts)
#tstack = sxc.dumb_stack(layers)
#pf.writeto('tstack.fits', tstack, overwrite=True)
#
#tdir = 'zzz'
#if not os.path.isdir(tdir):
#    os.mkdir(tdir)
#
##tag_bases = [os.path.basename(x) for x in tag_files]
##for ibase,idata in zip(tag_bases, layers):
##    tsave = os.path.join(tdir, 'r' + ibase)
##    sys.stderr.write("Saving %s ... \n" % tsave) 
##    pf.writeto(tsave, idata, overwrite=True)
#
#sys.stderr.write("\n\n\n")
#sys.stderr.write("visual inspection with:\n") 
#sys.stderr.write("flztfs %s\n" % ' '.join(tag_files))

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (07_spitzer_aor_extraction.py):
#---------------------------------------------------------------------
#
#  2021-02-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created 07_spitzer_aor_extraction.py.
#

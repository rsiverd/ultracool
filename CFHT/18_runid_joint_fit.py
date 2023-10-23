#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Find extracted catalogs with ephemerides (flavor "p_eph") and update their
# RA/DE coordinates ('dra' & 'dde' columns). The current procedure involves
# a few iterations of Gaia matching and optimization.
# save with new flavor.
#
# Rob Siverd
# Created:       2023-09-08
# Last modified: 2023-09-08
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
#import resource
#import signal
#import glob
#import gc
import os
import sys
import time
import numpy as np
import random
import pandas as pd
import scipy.optimize as opti
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
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

## WIRCam pipeline filesystem helpers:
try:
    import wircam_fs_helpers
    reload(wircam_fs_helpers)
except ImportError:
    logger.error("failed to import wircam_fs_helpers module!")
    sys.exit(1)
wfh = wircam_fs_helpers

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ecl = extended_catalog.ExtendedCatalog()
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## WCS tune-up helpers (beta):
try:
    import wircam_wcs_tuneup
    reload(wircam_wcs_tuneup)
    #ecl = extended_catalog.ExtendedCatalog()
    wwt = wircam_wcs_tuneup
except ImportError:
    logger.error("failed to import wircam_wcs_tuneup module!")
    sys.exit(1)

## WIRCam polynomial routines:
try:
    import wircam_poly
    reload(wircam_poly)
    wcp = wircam_poly.WIRCamPoly()
except ImportError:
    logger.error("failed to import wircam_poly module!")
    sys.exit(1)

## Region-file creation tools:
_have_region_utils = False
try:
    import region_utils
    reload(region_utils)
    rfy = region_utils
    _have_region_utils = True
except ImportError:
    sys.stderr.write(
            "\nWARNING: region_utils not found, DS9 regions disabled!\n")

## Tangent projection helper:
import tangent_proj
reload(tangent_proj)
tp = tangent_proj

## Angular math routines:
import angle
reload(angle)

## Custom polynomial fitter:
import custom_polyfit
reload(custom_polyfit)
cpf2_dx = custom_polyfit.CustomPolyFit2D()
cpf2_dy = custom_polyfit.CustomPolyFit2D()
cpf2_dx.set_degree(2, 3)
cpf2_dy.set_degree(2, 3)

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
    Perform per-run fit of on-sky PA and distortion parameters and
    recompute detection RA/DE positions using improved parameters.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt)
                          #formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(imtype=None) # 'cbcd', 'clean')
    #parser.set_defaults(imtype='p') # 'cbcd', 'clean')
    #parser.set_defaults(input_flavor='p_eph')
    #parser.set_defaults(output_flavor='p_fixed')
    parser.set_defaults(input_flavor='p_fixed')
    parser.set_defaults(output_flavor='p_nudge')
    #parser.set_defaults(sigthresh=2.0)
    parser.set_defaults(ntodo=0)
    parser.set_defaults(skip_existing=True)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('--overwrite', required=False, dest='skip_existing',
            action='store_false', help='overwrite existing catalogs')
    #iogroup.add_argument('-Q', '--wcs_pars_path', required=True, type=str,
    #        help='path to CSV file with companion WCS parameters')
    #iogroup.add_argument('-F', '--failure_list',
    #        default='failures.txt', type=str,
    #        help='output list of processing failures [def: %(default)s]')
    #iogroup.add_argument('-E', '--ephem_data', default=None, required=True,
    #        help='where to find input images', type=str)
    iogroup.add_argument('-G', '--gaia_csv', default=None, required=True,
            help='CSV file with Gaia source list', type=str)
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=False,
            help='where to save extended catalog outputs', type=str)
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find WIRCam images')
    #imtype = iogroup.add_mutually_exclusive_group()
    #imtype.add_argument('--cbcd', required=False, action='store_const',
    #        dest='imtype', const='cbcd', help='use cbcd images')
    #imtype.add_argument('--hcfix', required=False, action='store_const',
    #        dest='imtype', const='hcfix', help='use hot column-fixed images')
    #imtype.add_argument('--clean', required=False, action='store_const',
    #        dest='imtype', const='clean', help='use clean images')
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Miscellany:
    miscgroup = parser.add_argument_group('Miscellany')
    miscgroup.add_argument('--debug', dest='debug', default=False,
            help='Enable extra debugging messages', action='store_true')
    miscgroup.add_argument('-n', '--ntodo', type=int, required=False,
            help='maximum number of images to process [def: %(default)s]')
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
    if not context.input_flavor:
        sys.stderr.write("\nNo input catalog flavor selected!\n\n")
        sys.exit(1)

##--------------------------------------------------------------------------##
##------------------         Load Gaia CSV Catalog          ----------------##
##--------------------------------------------------------------------------##

#sys.stderr.write("Loading Gaia catalog ... ")
#if not os.path.isfile(context.gaia_csv):
#    sys.stderr.write("error!\nFile not found: %s\n" % context.gaia_csv)
#    sys.exit(1)
#wwt.gm.load_sources_csv(context.gaia_csv)
#sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##------------------        Make Input Catalog List         ----------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Listing %s catalogs ... " % context.input_flavor) 

## Ensure presence of input folder:
if not os.path.isdir(context.input_folder):
    sys.stderr.write("error: folder not found:\n")
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

## Look for files:
#context.imtype = "p"
if context.walk:
    cat_files = wfh.get_catalogs_walk(context.input_folder,
            flavor=context.input_flavor)
else:
    cat_files = wfh.get_catalogs_single(context.input_folder,
            flavor=context.input_flavor)
sys.stderr.write("done.\n")

## Abort in case of no input:
if not cat_files:
    sys.stderr.write("No input (%s) catalogs found in folder:\n" % context.input_flavor)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)
n_catalogs = len(cat_files)

##--------------------------------------------------------------------------##
##------------------           Extract Image Runids         ----------------##
##--------------------------------------------------------------------------##

## Get the runid from a file path. By convention, the runid is the name
## of the parent folder of each fcat file.
def get_runid_from_path(fcat_path):
    return os.path.basename(os.path.dirname(fcat_path))

## Get a list of unique runids. These are the subfolders that contain the
## listed catalogs. By convention, the runid is the name of the parent
## folder for each fcat found above.
unique_runids = sorted(list(set([get_runid_from_path(x) for x in cat_files])))
n_runids = len(unique_runids)
sys.stderr.write("Unique runids found: %d\n" % n_runids)

fcats_by_runid = {x:[] for x in unique_runids}
for fcat in cat_files:
    fcats_by_runid[get_runid_from_path(fcat)].append(fcat)

## Optionally randomize the file processing order:
#random.shuffle(cat_files)

#sys.exit(0)

##--------------------------------------------------------------------------##
##------------------            Match Data Loader           ----------------##
##--------------------------------------------------------------------------##

def load_gmatches_data(filename, tag):
    data = pd.read_csv(filename)
    data['tag'] = tag
    return data

def load_wcs_pars_data(filename, tag):
    data = pd.read_csv(filename)
    data['tag'] = tag
    #if 'fit3_crval2.1' in data.keys():
    #    data.rename(columns={'fit2_crval2':'fit2_crval1', 
    #                         'fit3_crval2':'fit3_crval1'}, inplace=True)
    #    data.rename(columns={'fit2_crval2.1':'fit2_crval2', 
    #                         'fit3_crval2.1':'fit3_crval2'}, inplace=True)
    return data

## Compute match xrel/yrel positions:
def update_rel_xy(data):
    data['xrel'] = data['x'] - wwt.crpix1
    data['yrel'] = data['y'] - wwt.crpix2
    return data

##--------------------------------------------------------------------------##
##------------------         Polynomial Coeff Saving        ----------------##
##--------------------------------------------------------------------------##

def save_coeffs_to_file(filename, runid, xcoeffs, ycoeffs):
    with open(filename, 'w') as cf:
        cf.write("runid,dim,c0,c1,c2,c3,c4,c5,c6\n")
        xdata = ','.join(['%e'%x for x in xcoeffs])
        ydata = ','.join(['%e'%y for y in ycoeffs])
        cf.write("%s,x,%s\n" % (runid, xdata))
        cf.write("%s,y,%s\n" % (runid, ydata))
        pass
    return

##--------------------------------------------------------------------------##
##------------------            Fitting Routines            ----------------##
##--------------------------------------------------------------------------##

n_coeffs = 7
xcoeffs_guess = np.zeros(n_coeffs)
ycoeffs_guess = np.zeros(n_coeffs)

def residual(params, imdata):
    return None

def scale_pa_resid(params, imdata):
    pscale = params[0]
    pa_deg = params[1]
    acoeff = params[2:9]                    # X coefficients (7 total)
    bcoeff = params[9:16]                   # Y coefficients (7 total)
    crvals = params[16:].reshape(-1, 2)     # (CRVAL1, CRVAL2) per image

    deltas = []
    for ipars,subset in zip(crvals, imdata):
        cv1, cv2 = ipars
        xnudge, ynudge = wcp._calc_nudges(acoeff, bcoeff,
                subset['xrel'].values, subset['yrel'].values)
        xcorr = subset['xrel'] + xnudge
        ycorr = subset['yrel'] + ynudge
        #sys.stderr.write("xnudge: %s\n" % str(xnudge))
        this_cdmat = tp.make_cdmat(pa_deg, pscale)
        tra, tdec = tp.xycd2radec(this_cdmat, xcorr, ycorr, cv1, cv2)
        deltas += angle.dAngSep(tra, tdec, subset['gra'], subset['gde']).tolist()
        pass
    return np.array(deltas)

def scale_pa_sum_abs_sep(params, imdata):
    return np.sum(scale_pa_resid(params, imdata))

def scale_pa_sum_squared_sep(params, imdata):
    return np.sum(scale_pa_resid(params, imdata)**2)


def calc_radec_offset(xrel, yrel, gra, gde, pscale, pa_deg, crval1, crval2):
    this_cdmat = tp.make_cdmat(pa_deg, pscale)
    tra, tde   = tp.xycd2radec(this_cdmat, xrel, yrel, crval1, crval2)
    tra = tra % 360.0
    #sys.stderr.write("tra: %s\n" % str(tra))
    #sys.stderr.write("tde: %s\n" % str(tde))
    ra_err     = gra - tra
    de_err     = gde - tde
    return ra_err, de_err

##--------------------------------------------------------------------------##
##------------------        Make Grid for Quiver Plot       ----------------##
##--------------------------------------------------------------------------##

npix = 2048
nbin = 32
#nbin = 16
bhalfsize = npix / nbin * 0.5
bctr = (np.arange(nbin) + 0.5) / float(nbin)
x_list = bctr * npix
y_list = bctr * npix
xx, yy = np.meshgrid(x_list, y_list)    # cell centers (abs x,y)
rel_xx = xx - wwt.crpix1                # cell center  (rel x)
rel_yy = yy - wwt.crpix2                # cell center  (rel y)


##--------------------------------------------------------------------------##
##------------------           Process All Runids           ----------------##
##--------------------------------------------------------------------------##

ntodo = context.ntodo
nproc = 0
for runid in unique_runids:
    sys.stderr.write("%s\n" % fulldiv)
    #unc_ipath = img_ipath.replace(context.imtype, 'cbunc')
    #if not os.path.isfile(unc_ipath):
    #    sys.stderr.write("WARNING: file not found:\n--> %s\n" % unc_ipath)
    #    continue
    run_fcats = sorted(fcats_by_runid[runid])
    run_gcats = [x+'.gmatch' for x in run_fcats]
    run_wpars = [x+'.wcspar' for x in run_fcats]
    have_gmcs = [os.path.isfile(x) for x in run_gcats]
    if not all(have_gmcs):
        sys.stderr.write("Runid %s is missing one or more gmatch files!\n" % runid)
        continue
    have_pars = [os.path.isfile(x) for x in run_wpars]
    if not all(have_pars):
        sys.stderr.write("Runid %s is missing one or more wcspar files!\n" % runid)
        continue

    # Coefficient save files:
    runid_dir    = os.path.dirname(run_fcats[0])
    coeff_save_1 = os.path.join(runid_dir, 'run_coeffs_1.csv')
    coeff_save_2 = os.path.join(runid_dir, 'run_coeffs_2.csv')
    subset_save  = os.path.join(runid_dir, 'run_match_subset.csv')

    # Load per-image match sets and WCS fits:
    sys.stderr.write("Loading Gaia match and WCS param files ... ")
    #per_img_matches = {}
    #per_img_wcspars = {}
    per_img_matches = []
    per_img_wcspars = []
    for fcat,gcat,wcat in zip(run_fcats, run_gcats, run_wpars):
        tag = os.path.basename(fcat)
        #per_img_matches[tag] = load_gmatches_data(gcat, tag)
        #per_img_wcspars[tag] = load_wcs_pars_data(wcat, tag)
        wtemp = load_wcs_pars_data(wcat, tag)
        gtemp = update_rel_xy(load_gmatches_data(gcat, tag))
        per_img_matches.append(gtemp)
        per_img_wcspars.append(wtemp)
    sys.stderr.write("done.\n")

    # Merge run fitted WCS params, get averages:
    run_wparams  = pd.concat(per_img_wcspars).reset_index()
    pa_deg_guess = np.median(run_wparams['fit3_pa_deg'])    # median PA
    run_cdmatrix = tp.make_cdmat(pa_deg_guess, wwt.pscale)
    cd11, cd12, cd21, cd22 = run_cdmatrix.flatten()
    fitted_crvals = run_wparams[['fit3_crval1', 'fit3_crval2']].values.flatten()


    # Convert Gaia RA/DE back to X,Y:
    for pp,mm in zip(per_img_wcspars, per_img_matches):
        #pa_deg = pp['fit3_pa_deg'][0]
        crval1 = pp['fit3_crval1'][0]
        crval2 = pp['fit3_crval2'][0]
        inv_xrel, inv_yrel = tp.sky2xy_cd(run_cdmatrix, 
                mm['gra'], mm['gde'], crval1, crval2)
        mm['inv_xrel'] = inv_xrel
        mm['inv_yrel'] = inv_yrel
        pass

    # Merge match data after augmentation (inverse X,Y added):
    run_matches = pd.concat(per_img_matches).reset_index()

    x_deltas = run_matches['inv_xrel'] - run_matches['xrel']
    y_deltas = run_matches['inv_yrel'] - run_matches['yrel']

    # Fit deltas with polynomial:
    cpf2_dx.fit(run_matches['xrel'].values, run_matches['yrel'].values, x_deltas.values)
    cpf2_dy.fit(run_matches['xrel'].values, run_matches['yrel'].values, y_deltas.values)
    xcoeffs_1 = cpf2_dx.get_model().copy()
    ycoeffs_1 = cpf2_dy.get_model().copy()

    # Eval deltas:
    predicted_dx = cpf2_dx.eval(rel_xx, rel_yy)
    predicted_dy = cpf2_dy.eval(rel_xx, rel_yy)
    #plt.clf()
    #plt.quiver(xx, yy, predicted_dx, predicted_dy)
    #plt.iver(xx, yy, predicted_dx, predicted_dy, scale=0.3, units='xy')
    #fig = plt.gcf()
    #ax = plt.gca()
    #ax.set_aspect('equal')
    #ax.quiver(xx, yy, predicted_dx, predicted_dy, scale=0.3, units='xy')
    #ax.set_xlabel("X Pixel")
    #ax.set_ylabel("Y Pixel")
    #plt.gcf().tight_layout()
    #plt.gcf().savefig('fitted_distortion.png')

    # Show differences:
    #quiver(xx, yy, predicted_dx-cs23_dx, predicted_dy-cs23_dy)


    # Compare to cs23 solution:
    #cs23_dx, cs23_dy = wcp.calc_xy_nudges(xx, yy, 'cs23')


    # Calculate distortion nudges according to best fit:
    calc_dx = cpf2_dx.eval(run_matches['xrel'].values, run_matches['yrel'].values)
    calc_dy = cpf2_dy.eval(run_matches['xrel'].values, run_matches['yrel'].values)

    # Calculate differences between nudged detection positions and Gaia-inferred
    # positions:
    x_errs = run_matches['xrel'] + calc_dx - run_matches['inv_xrel']
    y_errs = run_matches['yrel'] + calc_dy - run_matches['inv_yrel']
    #calc_xrel = run_matches['xrel'] + calc_dx
    #calc_yrel = run_matches['yrel'] + calc_dy
    r_errs = np.hypot(x_errs, y_errs)
    r_MAD  = np.median(r_errs)      # median absolute XY error
    sig_thresh = 5.0
    #calc_dr = np.hypot(x_deltas - calc_dx, y_deltas - calc_dy)
    #deviant = (calc_dr > 2.0)   # spurious match
    bad_matches = (r_errs > sig_thresh * r_MAD)
    bad_subset = run_matches[bad_matches].copy()
    chk_tag = 'wircam_J_1838732p_fixed.fits.fz.fcat'
    reg_subset = bad_subset.loc[bad_subset['tag'] == chk_tag]
    rfy.regify_ccd('bad_matches.reg', reg_subset['x'].values, reg_subset['y'].values,
            colors=['red'])

    use_subset = run_matches[~bad_matches].copy()
    reg_subset = use_subset.loc[use_subset['tag'] == chk_tag]
    rfy.regify_ccd('use_matches.reg', reg_subset['x'].values, reg_subset['y'].values,
            colors=['green'])

    # Re-fit good match subset deltas with polynomial:
    cpf2_dx.fit(use_subset['xrel'].values, use_subset['yrel'].values,
            x_deltas.values[~bad_matches])
    cpf2_dy.fit(use_subset['xrel'].values, use_subset['yrel'].values,
            y_deltas.values[~bad_matches])
    xcoeffs_2 = cpf2_dx.get_model().copy()
    ycoeffs_2 = cpf2_dy.get_model().copy()

    # Calculate distortion nudges according to best fit:
    calc_dx = cpf2_dx.eval(use_subset['xrel'].values, use_subset['yrel'].values)
    calc_dy = cpf2_dy.eval(use_subset['xrel'].values, use_subset['yrel'].values)

    # Save coefficients to file:
    save_coeffs_to_file(coeff_save_1, runid, xcoeffs_1, ycoeffs_1)
    save_coeffs_to_file(coeff_save_2, runid, xcoeffs_2, ycoeffs_2)


    # Calculate differences between nudged detection positions and Gaia-inferred
    # positions:
    x_errs_2 = use_subset['xrel'] + calc_dx - use_subset['inv_xrel']
    y_errs_2 = use_subset['yrel'] + calc_dy - use_subset['inv_yrel']
    #calc_xrel = run_matches['xrel'] + calc_dx
    #calc_yrel = run_matches['yrel'] + calc_dy
    r_errs_2 = np.hypot(x_errs, y_errs)
    r_MAD_2  = np.median(r_errs)      # median absolute XY error

    use_subset['x_err'] = x_errs_2
    use_subset['y_err'] = y_errs_2
    use_subset['r_err'] = r_errs_2
    sys.stderr.write("Saving CSV ... ")
    use_subset.to_csv(subset_save)
    sys.stderr.write("done.\n")
    #continue

    # Make resid image:
    res_image = np.zeros_like(xx)
    for ix,xctr in enumerate(x_list):
        xmin = xctr - bhalfsize
        xmax = xctr + bhalfsize
        in_col = (xmin <= use_subset['x']) & (use_subset['x'] < xmax)
        col_subset = use_subset[in_col]
        for iy,yctr in enumerate(y_list):
            ymin = yctr - bhalfsize
            ymax = yctr + bhalfsize
            in_row = (ymin <= col_subset['y']) & (col_subset['y'] < ymax)
            cell_data = col_subset[in_row]
            res_image[iy, ix] = np.median(cell_data['r_err'])
            pass
        
    
    #sys.exit(0)

    # Check whether every image is shifted such that the lower-right corner
    # has small RA/DE offsets when used without distortion correction. This
    # is needed to anchor the distortion solution ...
    image_ra_errs = []
    image_de_errs = []
    pixmax = 500.0
    #pixmax = 250.0
    #pixmax = 125.0
    pixmax = 750.0
    #pixmax = 3000.0
    divtxt = 75 * '='
    for fcat,pp,mm in zip(run_fcats, per_img_wcspars, per_img_matches):
        if (fcat != '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid/15BQ09/wircam_J_1838732p_fixed.fits.fz.fcat'):
            continue
        sys.stderr.write("%s\n" % divtxt)
        sys.stderr.write("%s\n" % divtxt)
        imcat = mm['tag'][0]
        #pa_deg = pp['fit3_pa_deg'][0]
        crval1 = pp['fit3_crval1'][0]
        crval2 = pp['fit3_crval2'][0]
        ra_err, de_err = \
                calc_radec_offset(mm['xrel'], mm['yrel'], mm['gra'], mm['gde'],
                wwt.pscale, pa_deg_guess, crval1, crval2)
                            
        mm['ra_err'] = ra_err
        mm['de_err'] = de_err
        mm['tot_err'] = 3600.0 * np.hypot(ra_err*np.cos(np.radians(mm['gde'])), de_err)

        lr_which = (np.hypot(mm.xrel, mm.yrel) < pixmax)
        lr_stars = mm[lr_which]
        lr_ra_err = 3600.0 * np.median(lr_stars['ra_err'])
        lr_de_err = 3600.0 * np.median(lr_stars['de_err'])
        sys.stderr.write("%s RA,DE shift: %.2f, %.2f\n" % (imcat, lr_ra_err, lr_de_err))
        image_ra_errs.append(lr_ra_err)
        image_de_errs.append(lr_de_err)

        # region files:
        imgpath = fcat.replace('fz.fcat', 'fz').replace('p_fixed.', 'p.')
        adjpath = imgpath.replace('fits.fz', 'fits.adj')
        pix_reg = fcat + '.pix_reg'
        sky_reg = fcat + '.sky_reg'
        # sky region (gaia):
        rfy.regify_sky(sky_reg, lr_stars['gra'], lr_stars['gde'],
                rdeg=0.0005, colors=['green'])
        # pix region (fcat):
        rfy.regify_ccd(pix_reg, lr_stars['x'], lr_stars['y'], colors=['red'])
        rfy.reg_announce(imcat, imgpath, [pix_reg, sky_reg])
        # modified FITS image:
        idata, imhdr = wwt.pf.getdata(imgpath, header=True)
        imhdr['CD1_1'] = cd11
        imhdr['CD1_2'] = cd12
        imhdr['CD2_1'] = cd21
        imhdr['CD2_2'] = cd22
        imhdr['CRVAL1'] = crval1 + (lr_ra_err / 3600.0)
        imhdr['CRVAL2'] = crval2 + (lr_de_err / 3600.0)
        wwt.qsave(adjpath, idata, header=imhdr, overwrite=True)
        rfy.reg_announce(imcat, adjpath, [pix_reg, sky_reg])
        pass

    avg_ra_err = np.average(image_ra_errs)
    avg_de_err = np.average(image_de_errs)
    sys.stderr.write("RA err (<%.0f pix): %.2f\n" % (pixmax, avg_ra_err))
    sys.stderr.write("DE err (<%.0f pix): %.2f\n" % (pixmax, avg_de_err))
    

    continue
    # Best-fit for varying scale and PA:
    sys.stderr.write("Running minimization ... ")
    tik = time.time()
    initial_guess = [wwt.pscale, pa_deg_guess]
    initial_guess.extend(xcoeffs_guess)
    initial_guess.extend(ycoeffs_guess)
    initial_guess.extend(fitted_crvals)
    initial_params = np.array(initial_guess)
    minimize_this = partial(scale_pa_sum_abs_sep, imdata=per_img_matches)

    resid_0_deg = scale_pa_resid(initial_params, imdata=per_img_matches)
    resid_0_sec = resid_0_deg * 3600.0
    resid_0_pix = resid_0_sec / wwt.pscale
    #answer = opti.fmin(minimize_this, initial_params, maxiter=10)
    tok = time.time()
    sys.stderr.write("Minimum found in %.3f seconds.\n" % (tok-tik))

    #fcat_base = os.path.basename(fcat_path)
    #sys.stderr.write("Processing catalog: %s\n" % fcat_base)

    ## output catalog name:
    #save_base = wfh.change_catalog_flavor(fcat_base, 
    #        context.input_flavor, context.output_flavor)
    #if not save_base:
    #    sys.stderr.write("No dice!\n")
    #    sys.exit(1)

    ## set output folder and file:
    #save_dir = context.output_folder
    #if context.walk:
    #    save_dir = os.path.dirname(fcat_path)
    #save_path = os.path.join(save_dir, save_base)

    ## announce things:
    #sys.stderr.write("Have output folder:\n")
    #sys.stderr.write("--> %s\n" % save_dir)
    #sys.stderr.write("--> %s\n" % save_path)

    ## skip catalogs that already exist:
    #sys.stderr.write("Output catalog %s ... " % save_base)
    #if context.skip_existing:
    #    if os.path.isfile(save_path):
    #        sys.stderr.write("exists!  Skipping ...\n")
    #        continue
    #    sys.stderr.write("not found ... creating ...\n")
    #else:
    #    sys.stderr.write("creating ... \n")

    ## If we get here, we need to do processing:
    #nproc += 1

    ## fetch ephemeris using partial filename:
    ##ftag = fcat_base.split('.')[0]
    ##this_eph = eee.get_eph_by_name(ftag)

    ## load the catalog:
    #ecl.load_from_fits(fcat_path)
    #stars  = ecl.get_catalog()
    #header = ecl.get_header()

    ## perform the tune-up:
    #new_stars = wwt.wcs_tuneup(stars, header)

    ## update the catalog:
    #ecl.set_catalog(new_stars)

    ## save the result:
    #ecl.save_as_fits(save_path, overwrite=True)

    ## stop early if requested:
    #if (ntodo > 0) and (nproc >= ntodo):
    #    break


######################################################################
# CHANGELOG (18_runid_joint_fit.py):
#---------------------------------------------------------------------
#
#  2023-09-08:
#     -- Increased __version__ to 0.1.0.
#     -- First created 18_runid_joint_fit.py.
#

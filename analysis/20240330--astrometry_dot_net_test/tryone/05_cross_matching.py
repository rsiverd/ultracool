#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Cross match SExtractor catalogs and ast.net solutions to Gaia. Save
# consolidated results for further analysis.
#
# Rob Siverd
# Created:       2025-02-03
# Last modified: 2025-02-03
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
#import argparse
#import shutil
#import resource
#import signal
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
import scipy.optimize as opti
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
from functools import partial
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

## Angular math:
import angle
reload(angle)

## Gaia catalog matching:
import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()

## Helpers for this investigation:
import helpers
reload(helpers)

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

## Pickle store routine:
def stash_as_pickle(filename, thing):
    with open(filename, 'wb') as sapf:
        pickle.dump(thing, sapf)
    return

## Pickle load routine:
def load_pickled_object(filename):
    with open(filename, 'rb') as lpof:
        thing = pickle.load(lpof)
    return thing

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

## Home-brew KDE:
#try:
#    import my_kde
#    reload(my_kde)
#    mk = my_kde
#except ImportError:
#    logger.error("module my_kde not found!  Install and retry.")
#    sys.stderr.write("\nError!  my_kde module not found!\n"
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
    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
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




##--------------------------------------------------------------------------##
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Parse arguments and run script:
#class MyParser(argparse.ArgumentParser):
#    def error(self, message):
#        sys.stderr.write('error: %s\n' % message)
#        self.print_help()
#        sys.exit(2)
#
### Enable raw text AND display of defaults:
#class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
#                        argparse.RawDescriptionHelpFormatter):
#    pass
#
### Parse the command line:
#if __name__ == '__main__':
#
#    # ------------------------------------------------------------------
#    prog_name = os.path.basename(__file__)
#    descr_txt = """
#    PUT DESCRIPTION HERE.
#    
#    Version: %s
#    """ % __version__
#    parser = argparse.ArgumentParser(
#            prog='PROGRAM_NAME_HERE',
#            prog=os.path.basename(__file__),
#            #formatter_class=argparse.RawTextHelpFormatter)
#            description='PUT DESCRIPTION HERE.')
#            #description=descr_txt)
#    parser = MyParser(prog=prog_name, description=descr_txt)
#                          #formatter_class=argparse.RawTextHelpFormatter)
#    # ------------------------------------------------------------------
#    parser.set_defaults(thing1='value1', thing2='value2')
#    # ------------------------------------------------------------------
#    parser.add_argument('firstpos', help='first positional argument')
#    parser.add_argument('-w', '--whatever', required=False, default=5.0,
#            help='some option with default [def: %(default)s]', type=float)
#    parser.add_argument('-s', '--site',
#            help='Site to retrieve data for', required=True)
#    parser.add_argument('-n', '--number_of_days', default=1,
#            help='Number of days of data to retrieve.')
#    parser.add_argument('-o', '--output_file', 
#            default='observations.csv', help='Output filename.')
#    parser.add_argument('--start', type=str, default=None, 
#            help="Start time for date range query.")
#    parser.add_argument('--end', type=str, default=None,
#            help="End time for date range query.")
#    parser.add_argument('-d', '--dayshift', required=False, default=0,
#            help='Switch between days (1=tom, 0=today, -1=yest', type=int)
#    parser.add_argument('-e', '--encl', nargs=1, required=False,
#            help='Encl to make URL for', choices=all_encls, default=all_encls)
#    parser.add_argument('-s', '--site', nargs=1, required=False,
#            help='Site to make URL for', choices=all_sites, default=all_sites)
#    parser.add_argument('remainder', help='other stuff', nargs='*')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    #iogroup = parser.add_argument_group('File I/O')
#    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
#    #        help='Output filename', type=str)
#    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
#    #        help='KELT image with WCS')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    ofgroup = parser.add_argument_group('Output format')
#    fmtparse = ofgroup.add_mutually_exclusive_group()
#    fmtparse.add_argument('--python', required=False, dest='output_mode',
#            help='Return Python dictionary with results [default]',
#            default='pydict', action='store_const', const='pydict')
#    bash_var = 'ARRAY_NAME'
#    bash_msg = 'output Bash code snippet (use with eval) to declare '
#    bash_msg += 'an associative array %s containing results' % bash_var
#    fmtparse.add_argument('--bash', required=False, default=None,
#            help=bash_msg, dest='bash_array', metavar=bash_var)
#    fmtparse.set_defaults(output_mode='pydict')
#    # ------------------------------------------------------------------
#    # Miscellany:
#    miscgroup = parser.add_argument_group('Miscellany')
#    miscgroup.add_argument('--debug', dest='debug', default=False,
#            help='Enable extra debugging messages', action='store_true')
#    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
#            help='less progress/status reporting')
#    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
#            help='more progress/status reporting')
#    # ------------------------------------------------------------------
#
#    context = parser.parse_args()
#    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
#    context.prog_name = prog_name
#
##--------------------------------------------------------------------------##

## Path converters:
def catalog_from_ipath(ipath):
    ibase = os.path.basename(ipath)
    return os.path.join('qcatalogs', ibase + '.ast')

def quadrant_from_ipath(ipath):
    return os.path.basename(ipath).split('_')[2]

##--------------------------------------------------------------------------##

## Gaia CSV:
#gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d3.csv'
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d4.csv'

## Load Gaia catalog:
sys.stderr.write("Loading Gaia ... ")
gm.load_sources_csv(gaia_csv_path)
sys.stderr.write("done.\n")


##--------------------------------------------------------------------------##



##--------------------------------------------------------------------------##

## Get list of images:
with open('keepers.txt', 'r') as kf:
    ipaths = [x.strip() for x in kf.readlines() if not x.startswith('#')]
    q_ipath = {quadrant_from_ipath(x):x for x in ipaths}
#ibases = [os.path.basename(x) for x in ipaths]
#ipaths = {x:y for x,y in zip(ibases, ipaths)}

#quads = [quadrant_from_ipath(x) for x in ipaths]

##--------------------------------------------------------------------------##

## The list of related catalogs:
#imcats = {x:catalog_from_ipath(x) for x in ipaths}
q_imcat = {x:catalog_from_ipath(y) for x,y in q_ipath.items()}

## Sanity check:
if not all([os.path.isfile(x) for x in q_imcat.values()]):
    sys.stderr.write("One or more catalogs missing!\n")
    sys.exit(1)

## Observation time of each image:
q_itime = {}
q_imwcs = {}
for qq,ipath in q_ipath.items():
    sys.stderr.write("Getting timestamp from %s ..." % ipath)
    _, hdrs = pf.getdata(ipath, header=True)
    q_itime[qq] = helpers.wircam_timestamp_from_header(hdrs)
    #obs_time = helpers.wircam_timestamp_from_header(hdrs)
    q_imwcs[qq] = awcs.WCS(hdrs)
    sys.stderr.write("done.\n")
obs_time = q_itime[qq]
gm.set_epoch(obs_time)

## Load catalogs and match to Gaia:
#q_cdata = {x:helpers.load_skypix_output(y) for x,y in q_imcat.items()}
mtol_arcsec = 0.3
q_cdata = {}
for qq,cpath in q_imcat.items():
    # Real X,Y position, correspond ast.net RA/DE:
    sys.stderr.write("Loading %s ... " % cpath)
    xx, yy, ant_ra, ant_de = helpers.load_skypix_output(cpath)
    nsrcs = len(xx)
    sys.stderr.write("done.\n")
    # Match to Gaia:
    sys.stderr.write("Matching to Gaia ... ")
    matches = gm.twoway_gaia_matches(ant_ra, ant_de, mtol_arcsec)
    idx, gra, gde, gid = matches
    nhits = len(idx)
    sys.stderr.write("done. Matched %d of %d sources.\n" % (nhits, nsrcs))
    q_cdata[qq] = {'axx':xx, 'ayy':yy, 'ara':ant_ra, 'ade':ant_de, 'midx':idx,
            'gid':gid, 'gxx':xx[idx], 'gyy':yy[idx], 'gra':gra, 'gde':gde}

##--------------------------------------------------------------------------##
## Symmetric-about-focal-plane pixel scale measurements. Start with a helper
## routine:
def center_pscale_checker(ww):
    x1, x2 = 1024., 1025.
    y1, y2 = 1024., 1025.
    ra11, de11 = ww.all_pix2world(x1, y1, 1)
    ra12, de12 = ww.all_pix2world(x1, y2, 1)
    ra21, de21 = ww.all_pix2world(x2, y1, 1)
    ra22, de22 = ww.all_pix2world(x2, y2, 1)
    dx1 = angle.dAngSep(ra11, de11, ra21, de21)   # x2-x1 @ y1
    dx2 = angle.dAngSep(ra12, de12, ra22, de22)   # x2-x1 @ y2
    dy1 = angle.dAngSep(ra11, de11, ra12, de12)   # y2-y1 @ x1
    dy2 = angle.dAngSep(ra21, de21, ra22, de22)   # y2-y1 @ x2
    return dx1, dx2, dy1, dy2


##--------------------------------------------------------------------------##
## Image edges:
edge = np.arange(2048) + 1.
L_x, L_y = np.ones_like(edge), edge
R_x, R_y = np.ones_like(edge) * 2048., edge
T_x, T_y = edge, np.ones_like(edge) * 2048.
B_x, B_y = edge, np.ones_like(edge)

##--------------------------------------------------------------------------##

## NE sensor, right edge RA/DE and scale:
ne_rm0 = q_imwcs['NE'].all_pix2world(R_x - 0.0, R_y, 1)
ne_rm1 = q_imwcs['NE'].all_pix2world(R_x - 1.0, R_y, 1)
ne_rsc = angle.dAngSep(*ne_rm0, *ne_rm1)

## NE sensor, bottom edge RA/DE and scale:
ne_bm0 = q_imwcs['NE'].all_pix2world(B_x, B_y + 0.0, 1)
ne_bm1 = q_imwcs['NE'].all_pix2world(B_x, B_y + 1.0, 1)
ne_bsc = angle.dAngSep(*ne_bm0, *ne_bm1)

## NW sensor, left edge RA/DE and scale:
nw_lm0 = q_imwcs['NW'].all_pix2world(L_x - 0.0, L_y, 1)
nw_lm1 = q_imwcs['NW'].all_pix2world(L_x + 1.0, L_y, 1)
nw_lsc = angle.dAngSep(*nw_lm0, *nw_lm1)

## NW sensor, bottom edge RA/DE and scale:
nw_bm0 = q_imwcs['NW'].all_pix2world(B_x, B_y + 0.0, 1)
nw_bm1 = q_imwcs['NW'].all_pix2world(B_x, B_y + 1.0, 1)
nw_bsc = angle.dAngSep(*nw_bm0, *nw_bm1)

## SE sensor, right edge RA/DE and scale:
se_rm0 = q_imwcs['SE'].all_pix2world(R_x - 0.0, R_y, 1)
se_rm1 = q_imwcs['SE'].all_pix2world(R_x - 1.0, R_y, 1)
se_rsc = angle.dAngSep(*se_rm0, *se_rm1)

## SE sensor, top edge RA/DE and scale:
se_tm0 = q_imwcs['SE'].all_pix2world(T_x, T_y - 0.0, 1)
se_tm1 = q_imwcs['SE'].all_pix2world(T_x, T_y - 1.0, 1)
se_tsc = angle.dAngSep(*se_tm0, *se_tm1)

## SW sensor, left edge RA/DE and scale:
sw_lm0 = q_imwcs['SW'].all_pix2world(L_x + 0.0, L_y, 1)
sw_lm1 = q_imwcs['SW'].all_pix2world(L_x + 1.0, L_y, 1)
sw_lsc = angle.dAngSep(*sw_lm0, *sw_lm1)

## SW sensor, top edge RA/DE and scale:
sw_tm0 = q_imwcs['SW'].all_pix2world(T_x, T_y - 0.0, 1)
sw_tm1 = q_imwcs['SW'].all_pix2world(T_x, T_y - 1.0, 1)
sw_tsc = angle.dAngSep(*sw_tm0, *sw_tm1)

##--------------------------------------------------------------------------##

## NE sensor, right edge slope:
tra, tde = ne_rm0
dra = (tra - tra[0]) * np.cos(np.radians(tde))
ne_slope = np.degrees(dra.max() / np.max(tde - tde[0]))

## SE sensor, right edge slope:
tra, tde = se_rm0
dra = (tra - tra[0]) * np.cos(np.radians(tde))
se_slope = np.degrees(dra.max() / np.max(tde - tde[0]))


##--------------------------------------------------------------------------##

## CRVAL differences:
ne_cv1, ne_cv2 = q_imwcs['NE'].wcs.crval
nw_cv1, nw_cv2 = q_imwcs['NW'].wcs.crval
se_cv1, se_cv2 = q_imwcs['SE'].wcs.crval
sw_cv1, sw_cv2 = q_imwcs['SW'].wcs.crval

avg_cv1 = np.average([ne_cv1, nw_cv1, se_cv1, sw_cv1])
avg_cv2 = np.average([ne_cv2, nw_cv2, se_cv2, sw_cv2])
#avg_cv1 = 294.58280003425
#avg_cv2 =  35.120328483275

# Average cd matrix:
avg_cdm = np.average([x.wcs.cd for x in q_imwcs.values()], axis=0)

##--------------------------------------------------------------------------##

## NE-NW sensor separation (left-right):
cosdec_ne_nw = np.cos(np.radians(0.5*(ne_rm0[1] + nw_lm0[1])))
rddiff_ne_nw = np.array(ne_rm0) - np.array(nw_lm0)
ra_sep_ne_nw = rddiff_ne_nw[0] * cosdec_ne_nw
de_sep_ne_nw = rddiff_ne_nw[1]
degsep_ne_nw = angle.dAngSep(*ne_rm0, *nw_lm0)
pixsep_ne_nw_NE = degsep_ne_nw / ne_rsc
pixsep_ne_nw_NW = degsep_ne_nw / nw_lsc

## SE-SW sensor separation (left-right):
cosdec_se_sw = np.cos(np.radians(0.5*(se_rm0[1] + sw_lm0[1])))
rddiff_se_sw = np.array(se_rm0) - np.array(sw_lm0)
ra_sep_se_sw = rddiff_se_sw[0] * cosdec_se_sw
de_sep_se_sw = rddiff_se_sw[1]
degsep_se_sw = angle.dAngSep(*se_rm0, *sw_lm0)
pixsep_se_sw_SE = degsep_se_sw / se_rsc
pixsep_se_sw_SW = degsep_se_sw / sw_lsc

## NE-SE sensor separation (top-bottom):
cosdec_ne_se = np.cos(np.radians(0.5*(ne_bm0[1] + se_tm0[1])))
rddiff_ne_se = np.array(ne_bm0) - np.array(se_tm0)
ra_sep_ne_se = rddiff_ne_se[0] * cosdec_ne_se
de_sep_ne_se = rddiff_ne_se[1]
degsep_ne_se = angle.dAngSep(*ne_bm0, *se_tm0)
pixsep_ne_se_NE = degsep_ne_se / ne_bsc
pixsep_ne_se_SE = degsep_ne_se / se_tsc

## NW-SW sensor separation (top-bottom):
cosdec_nw_sw = np.cos(np.radians(0.5*(nw_bm0[1] + sw_tm0[1])))
rddiff_nw_sw = np.array(nw_bm0) - np.array(sw_tm0)
ra_sep_nw_sw = rddiff_nw_sw[0] * cosdec_nw_sw
de_sep_nw_sw = rddiff_nw_sw[1]
degsep_nw_sw = angle.dAngSep(*nw_bm0, *sw_tm0)
pixsep_nw_sw_NW = degsep_nw_sw / nw_bsc
pixsep_nw_sw_SW = degsep_nw_sw / sw_tsc

##--------------------------------------------------------------------------##

## Fitting attempt. Simultaneous fit for:
## * global CRVAL1,CRVAL2, CDM
## * per-image CRPIX1, CRPIX2
sensor_order = ['NE', 'NW', 'SE', 'SW']
#par_guess = np.array([avg_cv1, avg_cv2, *avg_cdm.ravel(),
par_guess = np.array([*avg_cdm.ravel(), avg_cv1, avg_cv2,
       0.,    0.,
       ])
#    2122.,  -74.,                       # NE
#     -74.,  -74.,                       # NW
#    2122., 2122.,                       # SE
#     -74., 2122.,                       # SW
#     ])

#fittable_data = [q_cdata[x] for x in sensor_order]

cutoff_arcmin = 3.
cutoff_arcmin = 6.
fittable_data = []
for ss in sensor_order:
    sdata = q_cdata[ss]
    sep = angle.dAngSep(sdata['gra'], sdata['gde'], avg_cv1, avg_cv2)
    which = 60.*sep <= cutoff_arcmin
    keepers = [sdata['gxx'][which], sdata['gyy'][which],
               sdata['gra'][which], sdata['gde'][which]]
    fittable_data.append(keepers)

def four_chip_eval(pars, data):
    cdm_crv = pars[:6]
    #crpix = pars[6:].reshape(-1, 2)
    RX, RY = crpix = pars[6:]
    crpix = np.array([2122. + RX,  -74.0 + RY,
                       -74. + RX,  -74.0 + RY,
                      2122. + RX, 2122.0 + RY,
                       -74. + RX, 2122.0 + RY]).reshape(-1, 2)
    residuals = []
    #for (cpx1, cpx2),stuff in zip(crpix, data):
        #sys.stderr.write("cpx1: %f\n" % cpx1) 
        #sys.stderr.write("cpx2: %f\n" % cpx2) 
        #xrel = stuff['gxx'] - cpx1
        #yrel = stuff['gyy'] - cpx2
    for (cpx1, cpx2),(gxx, gyy, gra, gde) in zip(crpix, data):
        #import pdb; pdb.set_trace()
        xrel = gxx - cpx1
        yrel = gyy - cpx2
        calc_ra, calc_de = helpers.eval_cdmcrv(cdm_crv, xrel, yrel)
        calc_ra %= 360.0
        #import pdb; pdb.set_trace()
        #delta_ra = (stuff['gra'] - calc_ra) * np.cos(np.radians(stuff['gde']))
        #delta_de = stuff['gde'] - calc_de
        delta_ra = (gra - calc_ra) * np.cos(np.radians(gde))
        delta_de = gde - calc_de
        residuals.append(delta_ra)
        residuals.append(delta_de)
    return np.concatenate(residuals)
    #return calc_ra, calc_de

def four_chip_eval_detail(pars, data):
    cdm_crv = pars[:6]
    #crpix = pars[6:].reshape(-1, 2)
    RX, RY = crpix = pars[6:]
    crpix = np.array([2122. + RX,  -74.0 + RY,
                       -74. + RX,  -74.0 + RY,
                      2122. + RX, 2122.0 + RY,
                       -74. + RX, 2122.0 + RY]).reshape(-1, 2)
    xp_positions = []
    yp_positions = []
    ra_residuals = []
    de_residuals = []
    for (cpx1, cpx2),(gxx, gyy, gra, gde) in zip(crpix, data):
        xrel = gxx - cpx1
        yrel = gyy - cpx2
        calc_ra, calc_de = helpers.eval_cdmcrv(cdm_crv, xrel, yrel)
        calc_ra %= 360.0
        delta_ra = (gra - calc_ra) * np.cos(np.radians(gde))
        delta_de = gde - calc_de
        ra_residuals.append(delta_ra)
        de_residuals.append(delta_de)
        xp_positions.append(gxx)
        yp_positions.append(gyy)
    return xp_positions, yp_positions, ra_residuals, de_residuals
    #return calc_ra, calc_de

## This routine produces some kind of "non radialness" measure:
def four_chip_quiver_eval(pars, data):
    cdm_crv = pars[:6]
    #crpix = pars[6:].reshape(-1, 2)
    RX, RY = crpix = pars[6:]
    crpix = np.array([2122. + RX,  -74.0 + RY,
                       -74. + RX,  -74.0 + RY,
                      2122. + RX, 2122.0 + RY,
                       -74. + RX, 2122.0 + RY]).reshape(-1, 2)
    residuals = []
    for (cpx1, cpx2),(gxx, gyy, gra, gde) in zip(crpix, data):
        xrel = gxx - cpx1
        yrel = gyy - cpx2
        rrel = np.hypot(xrel, yrel)
        ru_x = xrel / rrel
        ru_y = yrel / rrel
        #calc_ra, calc_de = helpers.eval_cdmcrv(cdm_crv, xrel, yrel)
        #calc_ra %= 360.0
        calc_xx, calc_yy = helpers.inverse_tan_cdmcrv(cdm_crv, gra, gde)
        xnudge = calc_xx - xrel
        ynudge = calc_yy - yrel
        plen   = xnudge * ru_x + ynudge * ru_y
        xn_res = xnudge - plen * ru_x
        yn_res = ynudge - plen * ru_y
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        residuals.append(xn_res)
        residuals.append(yn_res)
    return np.concatenate(residuals)
    #return calc_ra, calc_de

## This routine produces some kind of "non radialness" measure:
def four_chip_radial_resids(pars, data):
    cdm_crv = pars[:6]
    crpix = pars[6:].reshape(-1, 2)
    rdist = []
    resid = []
    color = []
    count = 0
    for (cpx1, cpx2),(gxx, gyy, gra, gde) in zip(crpix, data):
        count += 1
        xrel = gxx - cpx1
        yrel = gyy - cpx2
        rrel = np.hypot(xrel, yrel)
        calc_xx, calc_yy = helpers.inverse_tan_cdmcrv(cdm_crv, gra, gde)
        xnudge = calc_xx - xrel
        ynudge = calc_yy - yrel
        #plen   = xnudge * ru_x + ynudge * ru_y
        #xn_res = xnudge - plen * ru_x
        #yn_res = ynudge - plen * ru_y
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        rdist.append(rrel)
        resid.append(np.hypot(xnudge, ynudge))
        color.append(np.ones_like(rrel) + count)
    return np.concatenate(rdist), np.concatenate(resid), np.concatenate(color)
    #return calc_ra, calc_de

minimize_this = partial(four_chip_eval, data=fittable_data)
#minimize_this = partial(four_chip_quiver_eval, data=fittable_data)

sys.stderr.write("Multi-sensor fitting ... ")
tik = time.time()
big_result = opti.least_squares(minimize_this, par_guess)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))


## Detailed residuals:
res_xp, res_yp, ra_res, de_res = \
        four_chip_eval_detail(big_result['x'], fittable_data)


##--------------------------------------------------------------------------##



##--------------------------------------------------------------------------##
## Gridding parameters:
ny, nx = 2048, 2048
#gridsize = 32
#gridsize = 16
gridsize =  8
n_xcells = gridsize
n_ycells = gridsize

cellpix_y = float(ny) / float(n_ycells)
cellpix_x = float(nx) / float(n_xcells)

## Gridded analysis 0 -- dig pixels scale out of WCS:
q_wcscl = {}
x_ctrs = cellpix_x * (0.5 + np.arange(n_xcells))
y_ctrs = cellpix_y * (0.5 + np.arange(n_ycells))
wsx, wsy = np.meshgrid(x_ctrs, y_ctrs)  # WCS sampling X,Y
for qq,ww in q_imwcs.items():
    nom_ra, nom_de = ww.all_pix2world(wsx+0, wsy+0, 1)
    off_ra, off_de = ww.all_pix2world(wsx+1, wsy+1, 1)
    deltas_deg = angle.dAngSep(nom_ra, nom_de, off_ra, off_de)
    q_wcscl[qq] = 3600.0 / np.sqrt(2) * deltas_deg
    pass

## Gridded analysis 1 -- point-to-point pixel scale estimates:
q_rhits = {}
q_rangs = {}
max_sep = 50
for qq,cdata in q_cdata.items():
    sys.stderr.write("Point-to-point pixel scales (%s) ... " % qq)
    rhits = []
    rangs = []
    xpix, ypix = cdata['gxx'], cdata['gyy']
    gra, gde = cdata['gra'], cdata['gde']
    for tx,ty,tra,tde in zip(xpix, ypix, gra, gde):
        rsep = np.hypot(xpix - tx, ypix - ty)
        which = (0.0 < rsep) & (rsep < max_sep)
        rdist = rsep[which]
        adist = angle.dAngSep(tra, tde, gra[which], gde[which])
        pxscl = 3600.0 * adist / rdist
        rhits += [(tx, ty, *rp) for rp in zip(rdist, pxscl) if rp[1]>0]
        xdiff = xpix[which] - tx
        ydiff = ypix[which] - ty
        cosde = np.cos(np.radians(gde[which]))
        radif =  (gra[which] - tra) * cosde
        dedif =  gde[which] - tde
        flipt = ydiff < 0
        xdiff[flipt] *= -1.0
        ydiff[flipt] *= -1.0
        radif[flipt] *= -1.0
        dedif[flipt] *= -1.0
        pix_ang = np.arctan(xdiff / ydiff)
        coo_ang = np.arctan(-radif / dedif)
        rangs += [(tx, ty, *pa) for pa in zip(pix_ang, coo_ang)]
        #import pdb; pdb.set_trace()
    q_rhits[qq] = np.array(rhits)
    q_rangs[qq] = np.array(rangs)
    sys.stderr.write("done.\n")

## Gridded analysis 2 -- cell-wise medians:
q_pxscl = {}
for qq,rsdata in q_rhits.items():
    rx, ry, rpix, rpscl = rsdata.T
    icx = np.int_((rx - 0.5) / cellpix_x)
    icy = np.int_((ry - 0.5) / cellpix_y)
    #posangs = np.zeros((n_ycells, n_xcells))  # storage array
    pscales = np.zeros((n_ycells, n_xcells))  # storage array
    pscales *= np.nan
    for yc in range(n_ycells):
        #sys.stderr.write("yc: %d\n" % yc)
        which_ycell = (icy == yc)
        y_subset = rsdata[which_ycell]
        y_icx    = icx[which_ycell]
        for xc in range(n_xcells):
            #sys.stderr.write("yc=%2d, xc=%2d\n" % (yc, xc))
            which_xcell = (y_icx == xc)
            cell_subset = y_subset[which_xcell]
            cell_rx, cell_ry, cell_rpix, cell_pscl = cell_subset.T
            pscales[yc, xc] = np.median(cell_pscl)
    q_pxscl[qq] = pscales

q_pxang = {}
for qq,rsdata in q_rangs.items():
    rx, ry, pang, cang = rsdata.T
    icx = np.int_((rx - 0.5) / cellpix_x)
    icy = np.int_((ry - 0.5) / cellpix_y)
    posangs = np.zeros((n_ycells, n_xcells))  # storage array
    posangs *= np.nan
    for yc in range(n_ycells):
        #sys.stderr.write("yc: %d\n" % yc)
        which_ycell = (icy == yc)
        y_subset = rsdata[which_ycell]
        y_icx    = icx[which_ycell]
        for xc in range(n_xcells):
            #sys.stderr.write("yc=%2d, xc=%2d\n" % (yc, xc))
            which_xcell = (y_icx == xc)
            cell_subset = y_subset[which_xcell]
            cell_rx, cell_ry, cell_pang, cell_cang = cell_subset.T
            posangs[yc, xc] = np.median(np.degrees(cell_pang - cell_cang))
            #import pdb; pdb.set_trace()
    q_pxang[qq] = posangs

## Average rotation of all cells:
every_PA = np.concatenate([x for x in q_pxang.values()]).flatten()
median_PA = np.median(every_PA[~np.isnan(every_PA)])
sys.stderr.write("Median position angle: %.4f degrees\n" % median_PA)

## Remove median to get relative rotations:
for pavec in q_pxang.values():
    pavec -= median_PA

##--------------------------------------------------------------------------##
## Plot config:

# gridspec examples:
# https://matplotlib.org/users/gridspec.html

#gs1 = gridspec.GridSpec(4, 4)
#gs1.update(wspace=0.025, hspace=0.05)  # set axis spacing

#ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3) # top-left + center + right
#ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2) # mid-left + mid-center
#ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) # mid-right + bot-right
#ax4 = plt.subplot2grid((3, 3), (2, 0))            # bot-left
#ax5 = plt.subplot2grid((3, 3), (2, 1))            # bot-center


##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
#fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
wig, wxs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=2, clear=True)
pig, pxs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=3, clear=True)
qig, qxs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=4, clear=True, aspect='equal')
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.grid(True)
#ax1.axis('off')
imskw = {'vmin':0.303, 'vmax':0.306}
axs[0,0].imshow(q_pxscl['NE'], **imskw)
axs[0,1].imshow(q_pxscl['NW'], **imskw)
axs[1,0].imshow(q_pxscl['SE'], **imskw)
axs[1,1].imshow(q_pxscl['SW'], **imskw)
[x.invert_yaxis() for x in axs.ravel()]
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plot_name = 'empirical_gaia_pxscl.png'
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')

wxs[0,0].imshow(q_wcscl['NE'], **imskw)
wxs[0,1].imshow(q_wcscl['NW'], **imskw)
wxs[1,0].imshow(q_wcscl['SE'], **imskw)
wxs[1,1].imshow(q_wcscl['SW'], **imskw)
[x.invert_yaxis() for x in wxs.ravel()]
wig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
wig.savefig('wcs_derived_pxscl.png')

angskw = {'vmin':-0.08, 'vmax':0.08}
angskw = {}
pxs[0,0].imshow(q_pxang['NE'], **angskw)
pxs[0,1].imshow(q_pxang['NW'], **angskw)
pxs[1,0].imshow(q_pxang['SE'], **angskw)
pxs[1,1].imshow(q_pxang['SW'], **angskw)
[x.invert_yaxis() for x in pxs.ravel()]
pig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
pig.savefig('empirical_rotation.png')

qxkw = {'units':'xy'}
qxs[0,0].quiver(res_xp[0], res_yp[0],  ra_res[0], de_res[0], **qxkw)
qxs[0,1].quiver(res_xp[1], res_yp[1],  ra_res[1], de_res[1], **qxkw)
qxs[1,0].quiver(res_xp[2], res_yp[2],  ra_res[2], de_res[2], **qxkw)
qxs[1,1].quiver(res_xp[3], res_yp[3],  ra_res[3], de_res[3], **qxkw)
for ax in qxs.ravel():
    ax.set_xlim(0, 2049)
    ax.set_ylim(0, 2049)
qig.tight_layout()
plot_name = 'joint_crpix_quiver.png'
plt.draw()
qig.savefig(plot_name, bbox_inches='tight')

## Polar scatter:
#skw = {'lw':0, 's':15}
#ax1.scatter(azm_rad, zdist_deg, **skw)

## For polar axes:
#ax1.set_rmin( 0.0)                  # if using altitude in degrees
#ax1.set_rmax(90.0)                  # if using altitude in degrees
#ax1.set_theta_direction(-1)         # clockwise
#ax1.set_theta_direction(+1)         # counterclockwise
#ax1.set_theta_zero_location("N")    # North-up
#ax1.set_rlabel_position(-30.0)      # move labels 30 degrees

## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)

#ax1.plot(kde_pnts, kde_vals)

#ax1.pcolormesh(xx, yy, ivals)

#blurb = "some text"
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes)
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes,
#      va='top', ha='left', bbox=dict(facecolor='white', pad=10.0))
#      fontdict={'family':'monospace'}) # fixed-width
#      fontdict={'fontsize':24}) # larger typeface

#colors = cm.rainbow(np.linspace(0, 1, len(plot_list)))
#for camid, c in zip(plot_list, colors):
#    cam_data = subsets[camid]
#    xvalue = cam_data['CCDATEMP']
#    yvalue = cam_data['PIX_MED']
#    yvalue = cam_data['IMEAN']
#    ax1.scatter(xvalue, yvalue, color=c, lw=0, label=camid)

#mtickpos = [2,5,7]
#ndecades = 1.0   # for symlog, set width of linear portion in units of dex
#nonposx='mask' | nonposx='clip' | nonposy='mask' | nonposy='clip'
#ax1.set_xscale('log', basex=10, nonposx='mask', subsx=mtickpos)
#ax1.set_xscale('log', nonposx='clip', subsx=[3])
#ax1.set_yscale('symlog', basey=10, linthreshy=0.1, linscaley=ndecades)
#ax1.xaxis.set_major_formatter(formatter) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
#ax1.set_xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#for label in ax1.get_xticklabels():
#    label.set_rotation(30)
#    label.set_fontsize(14) 

#ax1.xaxis.label.set_fontsize(18)
#ax1.yaxis.label.set_fontsize(18)

#ax1.set_xlim(nice_limits(xvec, pctiles=[1,99], pad=1.2))
#ax1.set_ylim(nice_limits(yvec, pctiles=[1,99], pad=1.2))

#ax1.legend(loc='best', prop={'size':24})

#spts = ax1.scatter(x, y, lw=0, s=5)
##cbar = fig.colorbar(spts, orientation='vertical')   # old way
#cbnorm = mplcolors.Normalize(*spts.get_clim())
#scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#scm.set_array([])
#cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

#plot_name = 'empirical_gaia_pxscl.png'
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')
#wig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+

######################################################################
# CHANGELOG (05_cross_matching.py):
#---------------------------------------------------------------------
#
#  2025-02-03:
#     -- Increased __version__ to 0.0.1.
#     -- First created 05_cross_matching.py.
#

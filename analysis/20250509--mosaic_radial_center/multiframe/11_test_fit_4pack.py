#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Attempt to fit the radial distortion model to a 4-sensor image using
# fcat data from the current pipeline. An input list exists that maps
# fcat file base names to full paths to all four catalogs. The basename
# is chosen by the user on the command line.
#
# Rob Siverd
# Created:       2026-01-13
# Last modified: 2026-01-13
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
import argparse
#import shutil
#import resource
#import signal
#import glob
import gc
import os
import sys
import time
import copy
import pprint
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
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
#import window_filter as wf
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Angular math routines:
import angle
reload(angle)

## Gaia catalog matching:
import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ecl = extended_catalog.ExtendedCatalog()
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

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
    Attempt to fit the radial distortion model to all four sensors.
    
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
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-L', '--fcat_list', default=None, required=True,
            help='input listing of fcats by basename', type=str)
    iogroup.add_argument('-I', '--image', default=None, required=True,
            help='which image to process (search token)', type=str)
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
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Sanity checks:
if not os.path.isfile(context.fcat_list):
    sys.stderr.write("File not found: %s\n" % context.fcat_list)
    sys.exit(1)

#fcat_paths = pd.

##--------------------------------------------------------------------------##
## Load Gaia data:
## Gaia stuf:
#gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d3.csv'
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d4.csv'
sys.stderr.write("Loading Gaia ... ")
gm.load_sources_csv(gaia_csv_path)
sys.stderr.write("done.\n")

## Load the match mask if it exists:
gmask_file = './gmatch_mask_%s.csv' % context.image
gmask_data = None
ngaia_want = 6963
if os.path.isfile(gmask_file):
    gmask_data = pd.read_csv(gmask_file, low_memory=False)
    ngaia_have = len(gmask_data)
    if ngaia_have != ngaia_want:
        sys.stderr.write("WARNING: unexpected len(gmask_data) (%d != %d) ...\n"
                         % (ngaia_have, ngaia_want))


##--------------------------------------------------------------------------##
## Load the fcat list:
#data_file = context.fcat_list
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = np.atleast_1d(np.genfromtxt(data_file, dtype=None, **gftkw))
#all_data = np.genfromtxt(fix_hashes(data_file), dtype=None, **gftkw)
#all_data = aia.read(data_file)

#all_data = append_fields(all_data, ('ra', 'de'), 
#         np.vstack((ra, de)), usemask=False)
#all_data = append_fields(all_data, cname, cdata, usemask=False)

pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
#all_data = pd.read_csv(data_file)
cat_table = pd.read_csv(context.fcat_list, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)
#nskip, cnames = analyze_header(data_file)
#all_data = pd.read_csv(data_file, names=cnames, skiprows=nskip, **pdkwargs)
#all_data = pd.DataFrame.from_records(npy_data)
#all_data = pd.DataFrame(all_data.byteswap().newbyteorder()) # for FITS tables

## Item lookup:
matches = cat_table.fbase.str.contains(context.image)
if np.sum(matches) != 1:
    sys.stderr.write("Matched none or many ...\n")
    sys.exit(0)

pathcols = [x for x in cat_table.keys() if 'path' in x]
chosen = cat_table[matches]
paths = [chosen[x].iloc[0] for x in pathcols]
ne_fpath, nw_fpath, se_fpath, sw_fpath = paths

## Dictify input paths:
quads = ['NE', 'NW', 'SE', 'SW']
cpath = dict(zip(quads, paths))
sensor_order = quads

## Ensure presence of paths:
if not all([os.path.isfile(x) for x in paths]):
    sys.stderr.write("One or more of the indicated fcat files is missing ...\n")
    sys.exit(1)

## Load extended catalogs:
sys.stderr.write("Loading catalogs ... ")
cdata = {}
chdrs = {}
for qq,fpath in cpath.items():
    ecl.load_from_fits(fpath)
    cdata[qq] = ecl.get_catalog()
    chdrs[qq] = ecl.get_header()
sys.stderr.write("done.\n")

## Get obs time from header:
obs_time = helpers.wircam_timestamp_from_header(chdrs['NE'])
gm.set_epoch(obs_time)

## Promote catalogs to DataFrame:
stars = {qq:pd.DataFrame.from_records(tt) for qq,tt in cdata.items()}

## Embed approximate SNR/uncertainty. Assume:
## * sky + rdnoise^2 =~ 1000.0     (RDNOISE ~ 30)
## * sigma =~ FWHM / SNR
## NOTE: to calibrate, I used known-decent parameters and evaluated the
## following: median(sqrt((resid / asterr)**2))  ==>  2.64
## That factor of 2.64 is the factor by which the errors are underestimated
## This value is consistent with the ~0.05 pixel RMS seen previously.
_rescale = 2.64
_wirgain = 3.8                  # electrons per ADU
#_fluxcol = 'FLUX_ISO'
_fluxcol = 'flux'
for ss in stars.values():
    ## FOR FWHM SEE https://github.com/sep-developers/sep/issues/34
    calc_fwhm = 2.0 * np.sqrt(np.log(2.) * (ss['a']**2 + ss['b']**2))
    src_ele = _wirgain * ss[_fluxcol]
    src_snr = src_ele / np.sqrt(src_ele + 1000.0)
    #ast_err = ss['FWHM_IMAGE'] / src_snr            # a
    ast_err = calc_fwhm / src_snr            # a
    ss[   'fwhm'] = calc_fwhm
    ss['dumbsnr'] = src_snr
    ss['dumberr'] = ast_err
    ss['realerr'] = ast_err * _rescale



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Calculate RA/DE, cross-match to Gaia, update star catalogs:
sensor_mtol = {'NE':0.3, 'NW':0.3, 'SE':0.3, 'SW':0.5}
sensor_mtol = {'NE':2.0, 'NW':2.0, 'SE':2.0, 'SW':2.0}
sensor_mtol = {'NE':4.0, 'NW':4.0, 'SE':4.0, 'SW':4.0}
#sys.stderr.write("RA/DE calculation and Gaia matching ... ")
sys.stderr.write("Gaia matching ... ")
#_xcol, _ycol = 'XWIN_IMAGE', 'YWIN_IMAGE'
_xcol, _ycol = 'x', 'y'
mtol_arcsec = 0.3
gmag_limit  = 19.0
gm.set_Gmag_limit(gmag_limit)
sys.stderr.write("\n")
for qq,ss in stars.items():
    #csln = imwcs.get(qq)
    xpos, ypos = ss[_xcol], ss[_ycol]
    #sra, sde = imwcs[qq].all_pix2world(xpos, ypos, 1, ra_dec_order=True)
    sra, sde = ss['calc_ra'], ss['calc_de']
    #ss['anet_ra'], ss['anet_de'] = sra, sde
    #matches = gm.twoway_gaia_matches(sra, sde, mtol_arcsec)
    matches = gm.twoway_gaia_matches(sra, sde, sensor_mtol.get(qq))
    idx, gra, gde, gid = matches
    gcosdec = np.cos(np.radians(gde))
    #mismatch =
    delta_ra_arcsec = 3600.0 * (gra - sra[idx]) * gcosdec
    delta_de_arcsec = 3600.0 * (gde - sde[idx])
    med_delta_ra, sig_delta_ra = rs.calc_ls_med_MAD(delta_ra_arcsec)
    med_delta_de, sig_delta_de = rs.calc_ls_med_MAD(delta_de_arcsec)
    sys.stderr.write("%s | RA~ %.3f +/- %.3f | DE~ %.3f +/- %.3f\n"
            % (qq, med_delta_ra, sig_delta_ra, med_delta_de, sig_delta_de))

    # also embed Gaia info:
    big_gid = np.zeros(len(xpos), dtype=gid.dtype)
    big_gra = np.zeros_like(sra) * np.nan
    big_gde = np.zeros_like(sde) * np.nan
    big_gid[idx] = gid
    big_gra[idx] = gra
    big_gde[idx] = gde
    ss['gid'] = big_gid
    ss['gra'] = big_gra
    ss['gde'] = big_gde
    #break
    pass
sys.stderr.write("done.\n")

## Split off subsets of source catalogs with just the Gaia matches:
all_gstars, use_gstars = {}, {}
for qq,ss in stars.items():
    all_gstars[qq] = ss[ss.gid > 0].copy()
    use_gstars[qq] = ss[ss.gid > 0].copy()

## Purge masked entries from use_gstars (in place overwrite):
bad_gids = None
if isinstance(gmask_data, pd.DataFrame):
    bad_gids = gmask_data.gid[gmask_data.masked]
    for qq in use_gstars.keys():
        _tmpdf = use_gstars.get(qq).copy()
        #_tmpdf['keeper'] = np.ones(len(_tmpdf), dtype='bool')
    #if isinstance(gmask_data, pd.DataFrame):
    #    keeper = ~_tmpdf.gid.isin(gmask_data.masked)
    #else:
    #    keeper = np.ones(len(_tmpdf), dtype='bool')
        _tmpdf['keeper'] = ~_tmpdf.gid.isin(bad_gids)
        use_gstars[qq] = _tmpdf[_tmpdf['keeper']]


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Note the CRPIXn values from FITS headers as starting point:
imcrval1 = chdrs['NE'].get('CRVAL1')
imcrval2 = chdrs['NE'].get('CRVAL2')


brute_param_guess = np.array([
         -0.00008508,   -0.00000007,   -0.00000005,    0.00008507,
       2071.39243883,  -77.72577076,   -0.0000851 ,    0.00000012,
          0.00000012,    0.00008509, -112.73831575,  -68.825143  ,
         -0.00008508,    0.00000007,    0.00000006,    0.00008508,
       2075.34775651, 2114.75533714,   -0.00008506,    0.00000014,
          0.00000014,    0.00008504, -115.36232304, 2127.45977327,
            imcrval1,      imcrval2])
        #294.59563805,   35.11858534])

##-----------------------------------------------------------------------##
##-----------------------------------------------------------------------##

## Easy-to-read parameter printout with
## * CDxx in [mas]
## * CRPIXn in [pix]
## * CRVALn in [deg]
def parprint(params, stream=sys.stderr):
    sfpar = sift_params(params)
    sfpar['cdmat'] = {qq:vv*3.6e6 for qq,vv in sfpar['cdmat'].items()}
    pprint.pprint(sfpar)


sys.stderr.write("brute_param_guess (new): %s\n" % str(brute_param_guess))

## Pretty good guess for the distortion:
guess_distmod = np.array([ 1.757279e-01,  1.175609e-03,  1.047979e-06,
                           5.059445e-11,  4.450758e-13, -1.039322e-16])

## Polynomial model. Hopefully this is a strictly positive value.
#def poly_eval(r, model):
#    #return model[0] + model[1]*r + model[2]*r*r
#    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r
#    return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r + model[4]*r*r*r*r
#    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r \
#    #        + model[4]*r*r*r*r + model[5]*r*r*r*r*r

def poly_eval2(r, c0, c1, c2):
    return c0 + r * (c1 + r * c2)

def poly_eval3(r, c0, c1, c2, c3):
    return c0 + r * (c1 + r * (c2 + r * c3))

def poly_eval4(r, c0, c1, c2, c3, c4):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r*c4)))

def poly_eval5(r, c0, c1, c2, c3, c4, c5):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r*c5))))

def poly_eval(r, model):
    return poly_eval5(r, *model)

## Radial distortion model X- and Y- corrections. With a strictly positive
## distortion magnitude, you need to *SUBTRACT* these from RA/DE-derived
## positions in order to compare with measured X,Y positions.
def calc_rdist_corrections(xrel, yrel, model):
    rdist = np.hypot(xrel, yrel)     # distance from CRPIX
    rcorr = poly_eval(rdist, model)  # total correction magnitude
    theta = np.arctan2(yrel, xrel)
    xcorr = rcorr * np.cos(theta)
    ycorr = rcorr * np.sin(theta)
    return xcorr, ycorr

## ----------------------------------------------------------------------- ##
## Parameter-parser for solving and diagnostics:
def sift_params(params):
    parsleft = params.copy()

    # Peel CDxx, CRPIXx from the front:
    cdmcrpix = parsleft[:24].reshape(-1, 6)
    parsleft = parsleft[24:]
    #cdmat_4s = cdmcrpix[:, :4]
    test_cdm_calc = {qq:vv for qq,vv in zip(sensor_order, cdmcrpix[:, :4])}
    test_sensor_crpix = {qq:vv for qq,vv in zip(sensor_order, cdmcrpix[:, 4:])}

    # Peel CRVAL1, CRVAL2 from the front next:
    cv1, cv2 = parsleft[:2]
    #crvals12 = parsleft[:2]
    #parsleft = parsleft[2:]

    # Remaining parameters are the distortion model:
    rdist_pars = parsleft[2:]

    return {'cdmat':test_cdm_calc, 'crpix':test_sensor_crpix,
            'crval':[cv1, cv2], 'rpars':rdist_pars}

## Inverse operation:
def unsift_params(sifted):
    parvec = []
    # CD matrix and CRPIX (6 pars per sensor) go first:
    for qq in sensor_order:
        parvec.extend(sifted['cdmat'][qq])
        parvec.extend(sifted['crpix'][qq])
    parvec.extend(sifted['crval'])          # CRVALs go next
    parvec.extend(sifted['rpars'])          # distortion model last
    return np.array(parvec)

## ----------------------------------------------------------------------- ##
## Misrotation calculator:
def calc_roterr_deg(ddata):
    xctr, yctr = 1024.5, 1024.5
    have_ang = np.arctan2(ddata['ymeas'] - yctr,
                          ddata['xmeas'] - xctr)
    want_ang = np.arctan2(ddata['ymeas'] - yctr + ddata['yerror'],
                          ddata['xmeas'] - xctr + ddata['xerror'])
    return np.median(np.degrees(want_ang - have_ang))

## Check for residual rotation in the diagnostic data:
def show_misrotations(diags):
    result = {qq:calc_roterr_deg(dd) for qq,dd in diags.items()}
    for qq in sensor_order:
        sys.stderr.write("Residual %s rotation: %9.4f deg\n" % (qq, result[qq]))
    return result

## Check for residual offset in the diagnostic data:
#def show_bulk_offset(diags):

## ----------------------------------------------------------------------- ##
## Calculate sensor-to-sensor gutters implied by various CRPIX values:
def describe_gutters(sifted):
    sensor_crpix = sifted['crpix']
    ne_crp1, ne_crp2 = sensor_crpix['NE']
    nw_crp1, nw_crp2 = sensor_crpix['NW']
    se_crp1, se_crp2 = sensor_crpix['SE']
    sw_crp1, sw_crp2 = sensor_crpix['SW']
    ne_nw = ne_crp1 - nw_crp1 - 2048.
    se_sw = se_crp1 - sw_crp1 - 2048.
    ne_se = se_crp2 - ne_crp2 - 2048.
    nw_sw = sw_crp2 - nw_crp2 - 2048.
    sys.stderr.write("NE <--> NW (upper): %.2f\n" % ne_nw)
    sys.stderr.write("SE <--> SW (lower): %.2f\n" % se_sw)
    sys.stderr.write("NE <--> SE ( left): %.2f\n" % ne_se)
    sys.stderr.write("NW <--> SW (right): %.2f\n" % nw_sw)
    return


def describe_rotations(sifted):
    _sord = sensor_order
    _smap = sensor_qqmap
    sensor_cdmat = sifted['cdmat']
    _data = np.array([helpers.analyze(sensor_cdmat[x]) for x in _sord])
    #sensor_padeg = _data[:, 0]
    sensor_padeg = {qq:vv for qq,vv in zip(_sord, _data[:, :1])}
    sensor_scale = {qq:vv for qq,vv in zip(_sord, _data[:, 1:]*3600)}
    #sensor_padeg, sensor_xscale, sensor_yscale = {}, {}, {}
    for pair in sensor_padeg.items():
        sys.stderr.write("Sensor %s PA: %.3f\n" % pair)
    for qq,psc in sensor_scale.items():
        sys.stderr.write("Sensor %s pixscales: %s\n" % (qq, str(psc)))
    # Relative PAs:
    for ii,jj in itt.combinations(range(len(_sord)), 2):
        qi, qj = _smap[ii], _smap[jj]
        pi, pj = sensor_padeg[qi], sensor_padeg[qj]
        sys.stderr.write("Between %s and %s: %.4f\n" % (qi, qj, pi-pj))
    return

def describe_answer(sifted):
    describe_gutters(sifted)
    return

## ----------------------------------------------------------------------- ##
## As noted above, this convention SUBTRACTS calculated x- and y-corrections
## from RA/DE-derived xrel/yrel coordinates before comparing to measured X,Y.
## THIS VERSION ALSO FITS RADIAL DISTORTION PARAMETERS
## This version expects per-sensor CD11, CD12, CD21, CD22, CRPIX1, CRPIX2 at
## the front of the parameters array.
def squared_residuals_foc2ccd_rdist(params, diags=False,
                    dataset=use_gstars, unsquared=False, snrweight=False):
    # parse parameters
    sifted = sift_params(params)
    brute_cdmat = sifted['cdmat']
    brute_crpix = sifted['crpix']
    using_crval = sifted['crval']
    #rdist_pars  = sifted['rpars'] 

    #nextra = 5 - len(rdist_pars)
    #nextra = 6 - len(rdist_pars)
    #test_distmod = rdist_pars.tolist() + [0.0]*nextra
    test_distmod = guess_distmod

    # note average star count for normalization
    avg_nstars = np.average([len(x) for x in dataset.values()])
    typical_rdist = 1448.0     # 0.5 * np.hypot(2048, 2048)
    qxres, qyres = {}, {}
    xres, yres = [], []
    diag_data = {}
    for qq,gst in dataset.items():
        nstar_scale_factor = np.sqrt(avg_nstars / float(len(gst)))
        tcpx1, tcpx2 = brute_crpix.get(qq)
        #gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        gxx, gyy = gst['x'], gst['y']
        #cdmcrv = np.array(brute_crval.get(qq).tolist() + [test_crval1, test_crval2])
        cdmcrv = np.array(brute_cdmat.get(qq).tolist() + using_crval)
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    dataset[qq]['gra'], dataset[qq]['gde'])
        #import pdb; pdb.set_trace()
        #breakpoint()
        test_rrel = np.hypot(test_xrel, test_yrel)
        xnudge, ynudge = calc_rdist_corrections(test_xrel, test_yrel, test_distmod)
        test_xccd = test_xrel + xnudge + tcpx1
        test_yccd = test_yrel + ynudge + tcpx2
        x_error = test_xccd - gxx.values
        y_error = test_yccd - gyy.values
        scaled_xerr = x_error * nstar_scale_factor
        scaled_yerr = y_error * nstar_scale_factor
        if snrweight:
            scaled_xerr /= gst['realerr']
            scaled_yerr /= gst['realerr']
        #scaled_xerr *= test_rrel / typical_rdist    # more weight far away
        #scaled_yerr *= test_rrel / typical_rdist    # more weight far away
        #scaled_xerr *= np.sqrt(test_rrel / typical_rdist)    # more weight far away
        #scaled_yerr *= np.sqrt(test_rrel / typical_rdist)    # more weight far away
        #qxres[qq] = scaled_xerr
        #qyres[qq] = scaled_yerr
        xres.extend(scaled_xerr)
        yres.extend(scaled_yerr)
        #xres.extend(xres)
        if diags:
            diag_data[qq] = {   "gid":gst['gid'],
                              "xmeas":gxx,
                              "ymeas":gyy,
                              "xcalc":test_xccd,
                              "ycalc":test_yccd,
                              'rdist':test_rrel,
                             'xnudge':xnudge,
                             'ynudge':ynudge,
                             'xerror':x_error,
                             'yerror':y_error,
                             'rerror':np.hypot(x_error, y_error),
                             'scaled_xerror':scaled_xerr,
                             'scaled_yerror':scaled_yerr,
                             'scaled_rerror':np.hypot(scaled_xerr, scaled_yerr),
                               'flux':gst['flux'],
                               'fwhm':gst['fwhm'],
                              'flags':gst['flag'],
                            'dumbsnr':gst['dumbsnr'],
                            'realerr':gst['realerr'],
                            }

        pass
    #return qxres, qyres
    #return xres, yres
    if diags:
        return diag_data
    if unsquared:
        return np.concatenate((xres, yres))
    return np.concatenate((xres, yres))**2

def fmin_squared_residuals_foc2ccd_rdist(params, **kwargs):
    return np.sum(squared_residuals_foc2ccd_rdist(params, **kwargs))

use_params = brute_param_guess.copy()

## Optimize those parameters:
sys.stderr.write("Optimizing parameters ...\n")
#slvkw = {'loss':'soft_l1'}
typical_scale = np.array([0.01, 1.0, 1.0, 0.01, 0.01, 1e-5])
#slvkw = {'loss':'linear'}
#slvkw = {'loss':'linear', 'x_scale':typical_scale}
slvkw = {'method':'lm', 'xtol':1e-14, 'ftol':1e-14}
reskw = {'unsquared':True}
#reskw = {'unsquared':True, 'snrweight':True}
answer = opti.least_squares(squared_residuals_foc2ccd_rdist, use_params, kwargs=reskw, **slvkw)
sys.stderr.write("Ended up with: %s\n" % str(answer))
sys.stderr.write("Ended up with: %s\n" % str(answer['x']))


sys.stderr.write("\n\n\nTry again with fmin ....\n")
fmkw = {'full_output':True, 'xtol':1e-14, 'ftol':1e-14}
#shrink_this = partial(fmin_squared_residuals_foc2ccd_rdist, unsquared=False, snrweight=True)
shrink_this = partial(fmin_squared_residuals_foc2ccd_rdist, unsquared=False, snrweight=False)
fanswer = opti.fmin(shrink_this, use_params, **fmkw)
print(fanswer[0])

sys.stderr.write("\n")
sys.stderr.write("levmar results in `answer['x']`\n")
sys.stderr.write("fmin results in `fanswer[0]`\n")
sys.stderr.write("\n")

_CHOICE = 'fmin'
#_CHOICE = 'lstq'

if _CHOICE == 'fmin':
    fitted_pars = fanswer[0]
if _CHOICE == 'lstq':
    fitted_pars = answer['x']

sifted_pars = sift_params(fitted_pars)



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## ----------------------------------------------------------------------- ##
## Diagnostics time. There are TWO sets of diagnostic data, each corresponding
## to a different input star dataset.
## 1) The usual set includes residuals etc based on only the 'use_gstars'
## subset that were in use this round. Outliers identified in previous passes
## are removed. Use this for plotting and to save for external analysis.
## 2) Another set of residuals is evaluated using the 'all_gstars' dataset.
## This contains all of the stars with their errors etc according to the
## latest fitting parameters. We need this in order to update the blacklist.

#diag_lstq = squared_residuals_foc2ccd_rdist(answer['x'], diags=True)
#diag_fmin = squared_residuals_foc2ccd_rdist(fanswer[0], diags=True)
#diag_orig = squared_residuals_foc2ccd_rdist(fitted_pars, diags=True)
#diag_data = diag_fmin
#diag_data = diag_lstq
orig_diag_data = squared_residuals_foc2ccd_rdist(fitted_pars, diags=True)
## Note residual rotation:
diag_data = orig_diag_data
rot_error = show_misrotations(diag_data)

## Modify parameters:
#_adjust_pa_crpix = False
#_adjust_pa_crpix = True

derot_sifted = copy.deepcopy(sifted_pars)
for qq,resid in rot_error.items():
    rmat = helpers.rotation_matrix(-1.0 * np.radians(resid))
    _new_cdm = np.dot(rmat, derot_sifted['cdmat'][qq].reshape(2, 2))
    derot_sifted['cdmat'][qq] = _new_cdm.flatten()
    pass
derot_params = unsift_params(derot_sifted)

## The following has the rotation fixed but wonky CRPIX:
derot_diags = squared_residuals_foc2ccd_rdist(derot_params, diags=True)
diag_data = derot_diags

## Modify CRPIX:
fixed_sifted = copy.deepcopy(derot_sifted)
for qq,ddata in derot_diags.items():
    fixed_sifted['crpix'][qq][0] -= np.median(ddata['xerror'])
    fixed_sifted['crpix'][qq][1] -= np.median(ddata['yerror'])
fixed_params = unsift_params(fixed_sifted)

## De-rotated and shifted results:
fixed_diags = squared_residuals_foc2ccd_rdist(fixed_params, diags=True)
diag_data = fixed_diags

## For now, adopt the fixed (derot+shift) parameters to evaluate
## the all_gstars dataset. This second set of diags is the basis for
## updating the blacklist.
full_diag_data = squared_residuals_foc2ccd_rdist(fixed_params,
                                                 diags=True, dataset=all_gstars)

#diag_data = diag_lstq
#diag_data = orig_diag_data
#diag_data = derot_diags

## ----------------------------------------------------------------------- ##
## -----------------         MCMC Parameter Search          -------------- ##
## ----------------------------------------------------------------------- ##

## MCMC sampler:
import emcee
import corner
from multiprocessing import Pool

## Settings:
_do_MCMC = True
_do_MCMC = False
_xy_rmsd = 0.057
_xy_rmsd = 0.08 
_xy_rms2 = _xy_rmsd**2

def lnprior(params):
    return 0

### DUMB version with fixed RMS:
#def lnlike(params):
#    residuals = squared_residuals_foc2ccd_rdist(params, snrweight=False)
#    return -0.5*np.sum(residuals / _xy_rms2)

## Slightly-less-dumb version with SNR-scaled errors ...
def lnlike(params):
    residuals = squared_residuals_foc2ccd_rdist(params, snrweight=True)
    return -0.5*np.sum(residuals)

def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params)

if _do_MCMC:
    tik = time.time()
    initial = fixed_params.copy()
    ndim = len(initial)
    #nwalkers = 32
    nwalkers = 64
    nwalkers = 128
    #nthreads = 10
    nthreads = 8
    #nthreads = 4
    sprexp = 6
    sprexp = 5
    spread = 10.**(-1. * sprexp)
    p0 = [np.array(initial) + spread*initial*np.random.randn(ndim) \
            for i in range(nwalkers)]
    niter, thinned = 4000, 15
    niter, thinned = 4000, 5
    niter, thinned = 2000, 5
    #niter, thinned = 8000, 5
    #niter, thinned = 12000, 5
    #niter, thinned = 2000, 5 
    #niter, thinned = 200, 5 
    #niter, thinned = 20000, 5 
    with Pool(nthreads) as pool:
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #, args=arglist)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

        sys.stderr.write("Running burn-in ... ")
        p0, _, _ = sampler.run_mcmc(p0, 200) #, progress=True)
        sys.stderr.write("done.\n")
        sampler.reset()

        sys.stderr.write("Running full MCMC ... \n")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        #sys.stderr.write("done.\n")

    flat_samples = sampler.get_chain(discard=100, thin=thinned, flat=True)
    tok = time.time()
    sys.stderr.write("Running MCMC took %.2f seconds.\n" % (tok-tik))

    # best parameters from mcmc:
    mcmcpars = np.array([np.median(x) for x in sampler.flatchain.T])

    # some plots:
    corner_png = 'brutal_26par_w%03d_s%d_n%05d_t%02d.%s.png' \
            % (nwalkers, sprexp, niter, thinned, _CHOICE)
    slabtxt = ['CD11', 'CD12', 'CD21', 'CD22', 'CRPIX1', 'CRPIX2']
    #plabels = ['p%d'%(x+1) for x in range(ndim)]
    plabels = list(itt.chain.from_iterable([[y+'_'+x for x in slabtxt] \
                                for y in quads])) + ['CRVAL1', 'CRVAL2']

    #cornerfig = plt.figure(22, figsize=(12,9))
    #cornerfig = plt.figure(22, figsize=(24,18))
    cornerfig = plt.figure(22, figsize=(36, 27))
    cornerfig.clf()
    #corner.corner(flat_samples, labels=plabels, 
    #              truths=mcmcpars, fig=cornerfig)
    ckw = {'smooth':None, 'labelpad':0.5, 'truths':None, 'max_n_ticks':3,
           'show_titles':True}
    corner.corner(flat_samples, labels=plabels, fig=cornerfig, **ckw)
    cornerfig.savefig(corner_png, bbox_inches='tight')
    plt.draw()
    print(mcmcpars)

## ----------------------------------------------------------------------- ##
## Create a large, flat DataFrame for further analysis. This panel contains
## data from all stars (from full_diag_data) and is suitable for updating
## the Gaia blacklist. A subset is selected later for plotting.

## Full set 'bigdf' with data for blacklist:
dflist = []
for qq,ddata in full_diag_data.items():
    _tdf = pd.DataFrame.from_dict(ddata)
    _tdf['qq'] = [qq for x in range(len(_tdf))]
    dflist.append(_tdf)
bigdf = pd.concat(dflist)
del _tdf, dflist
#bigdf.to_csv(diags_csv, index=False)
#sys.stderr.write("To keep: mv -f best_fit_diags.csv projection_func/.\n")

## Fraction of total residuals in outliers:
total_rerror = np.sum(bigdf['rerror']**2)
sys.stderr.write("Total rerror: %10.4f\n" % total_rerror)
rr_hi_thresh = 0.6
rr_hi_masked = bigdf.rerror > rr_hi_thresh
rr_hi_rerror = np.sum(bigdf[rr_hi_masked]['rerror']**2)
rr_hi_fraction = rr_hi_rerror / total_rerror
sys.stderr.write("Outly rerror: %10.4f (%.1f%% > %.2f)\n"
                 % (rr_hi_rerror, 100.*rr_hi_fraction, rr_hi_thresh))
bigdf['rr_hi_mask'] = rr_hi_masked
inner_thresh = 0.4
#inner_bounds = (500, 2500)
inner_bounds = (50, 2900)
inner_masked = bigdf.rdist.between(*inner_bounds) & (bigdf.rerror > inner_thresh)
bigdf['inner_mask'] = inner_masked

# With ~stable solution, flag outliers in X/Y:
#xx_lo_thresh, xx_hi_thresh = -0.3, 0.3
xx_lo_thresh, xx_hi_thresh = -0.25, 0.25
xx_lo_masked = bigdf.xerror < xx_lo_thresh
xx_hi_masked = bigdf.xerror > xx_hi_thresh
xx_masked    = xx_lo_masked | xx_hi_masked
#yy_lo_thresh, yy_hi_thresh = -0.3, 0.3
yy_lo_thresh, yy_hi_thresh = -0.25, 0.25
yy_lo_masked = bigdf.yerror < yy_lo_thresh
yy_hi_masked = bigdf.yerror > yy_hi_thresh
yy_masked    = yy_lo_masked | yy_hi_masked
xy_masked    = xx_masked | yy_masked

combo_masked = inner_masked | rr_hi_masked | xy_masked
bigdf['masked'] = combo_masked
combo_rerror = np.sum(bigdf[combo_masked]['rerror']**2)
combo_fraction = combo_rerror / total_rerror
sys.stderr.write("Combo rerror: %10.4f (%.1f%%)\n" % (combo_rerror, 100.*combo_fraction))
## Save latest bigdf for projection use:
diags_csv = 'best_fit_diags.csv'
bigdf.to_csv(diags_csv, index=False)
sys.stderr.write("To keep: mv -f best_fit_diags.csv projection_func/.\n")

#sys.stderr.write("HALT after writing out CSV file ...\n")
#sys.exit(0)

## Augment the list of masked objects. IMPORTANT: GIDs in the current
## blacklist are NOT in bigdf (they are already pruned). Objects masked
## in the current iteration of bigdf must be ADDED to the blacklist.
current_mskcols = bigdf[['gid', 'rdist', 'rerror', 'masked']]

## Sanity check (started with 6862 objects):
if len(current_mskcols) != 6862:
    sys.stderr.write("WARNING ... dataset size changed ...\n")

## INTERACTIVE: also save/update a blacklist of discrepant matches:
# current_mskcols.to_csv(gmask_file, index=False)
# updated_blacklist.to_csv(gmask_file, index=False)

## INTERACTIVE: also save/update a blacklist of discrepant matches:
# mskdf = bigdf[['gid', 'rdist', 'rerror', 'masked']]
# mskdf.to_csv(gmask_file, index=False)
# outly_verybad = mskdf.rerror > rr_hi_thresh

## Downselect bigdf to match 'use_gstars' for plotting:
keeper_gids = pd.concat([x['gid'] for x in use_gstars.values()])
pltdf = bigdf[bigdf['gid'].isin(keeper_gids)]

## All the non-masked points for an RMS update:
clndf = bigdf[~bigdf['masked']]
xstd_pix = np.std(clndf.xerror)
ystd_pix = np.std(clndf.yerror)
xstd_mas = 306. * xstd_pix
ystd_mas = 306. * ystd_pix
sys.stderr.write("X-error stddev (pix, mas): %.4f, %6.2f\n" % (xstd_pix, xstd_mas))
sys.stderr.write("Y-error stddev (pix, mas): %.4f, %6.2f\n" % (ystd_pix, ystd_mas))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


fig_dims = (8, 8)
fig, axs1 = plt.subplots(2, 2, figsize=fig_dims, num=1, clear=True)
[ax.set_aspect('equal', adjustable='box') for ax in axs1.flatten()]
ax1map  = {'NE':axs1[0, 0], 'NW':axs1[0, 1], 'SE':axs1[1, 0], 'SW':axs1[1,1]}

fig2, axs2 = plt.subplots(2, 2, figsize=fig_dims, num=2, clear=True)
ax2map = {'NE':axs2[0, 0], 'NW':axs2[0, 1], 'SE':axs2[1, 0], 'SW':axs2[1,1]}

fig3, axs3 = plt.subplots(2, 1, figsize=fig_dims, num=3, clear=True, squeeze=False)

fig4, axs4 = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=4, clear=True)
qaxs4 = {'NE':axs4[0,0], 'NW':axs4[0,1], 'SE':axs4[1,0], 'SW':axs4[1,1]}

fig5, axs5 = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=5, clear=True)
qaxs5 = {'NE':axs5[0,0], 'NW':axs5[0,1], 'SE':axs5[1,0], 'SW':axs5[1,1]}

fig6, axs6 = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=6, clear=True)
qaxs6 = {'NE':axs6[0,0], 'NW':axs6[0,1], 'SE':axs6[1,0], 'SW':axs6[1,1]}

# Combined radial data for plotting/analysis:
every_rsep = []
every_rerr = []
for qq in sensor_order:
    _tdd = diag_data[qq]
    every_rsep.extend(_tdd['rdist'])
    xshift = _tdd['xerror'] - _tdd['xnudge']
    yshift = _tdd['yerror'] - _tdd['ynudge']
    rshift = np.hypot(xshift, yshift)
    every_rerr.extend(rshift)
#every_rdist = np.concatenate([diag_data[qq]['rdist'] for qq in sensor_order])
every_rsep = np.array(every_rsep)
every_rerr = np.array(every_rerr)

#fig4.clf()
#ax4 = fig4.add_subplot(111)
skw = {'lw':0, 's':10}
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
qpopt, qpcov = {}, {}
xtemp = np.linspace(250, 2750)
mskkw = {'c':'m', 's':50}
ascale = 5
ascale = 25
for qq,ddata in diag_data.items():
    #snsdf = bigdf[bigdf['qq'] == qq]    # this sensor
    #mskdf = snsdf[snsdf['masked']]      # this sensor + masked
    baddf = pltdf[(pltdf['qq'] == qq) & pltdf['masked']]

    # Illustrate the distortion correction:
    ax1map.get(qq).quiver(ddata['xmeas'], ddata['ymeas'],
                          ddata['xnudge'], ddata['ynudge'], color='k', **qxkw)

    # X,Y error vs pixel position:
    ax2map.get(qq).quiver(ddata['xmeas'], ddata['ymeas'],
                          ascale*ddata['xerror'], ascale*ddata['yerror'],
                          color='r', **qxkw)

    # R error vs R distance:
    axs3[0, 0].scatter(ddata['rdist'], ddata['rerror'], label=qq, **skw)

    # R error:
    #_rbad = subbdf[subbdf['rr_hi_mask']]
    rax = qaxs4[qq]
    rax.set_title(qq)
    rax.scatter(baddf['rdist'], baddf['rerror'], **mskkw)
    rax.scatter(ddata['rdist'], ddata['rerror'], label='raw', **skw)
    rax.scatter(ddata['rdist'], ddata['scaled_rerror'], label='scl', **skw)
    rax.set_ylabel('R error [pix]')
    # X error:
    xax = qaxs5[qq]
    xax.scatter(baddf['rdist'], baddf['xerror'], **mskkw)
    xax.scatter(ddata['rdist'], ddata['xerror'], label='raw', **skw)
    xax.scatter(ddata['rdist'], ddata['scaled_xerror'], label='scl', **skw)
    xax.set_title(qq)
    xax.set_ylabel('X error [pix]')
    # Y error:
    yax = qaxs6[qq]
    yax.set_title(qq)
    yax.scatter(baddf['rdist'], baddf['yerror'], **mskkw)
    yax.scatter(ddata['rdist'], ddata['yerror'], label='raw', **skw)
    yax.scatter(ddata['rdist'], ddata['scaled_yerror'], label='scl', **skw)
    yax.set_ylabel('Y error [pix]')
    pass

for ax in axs3.ravel():
    ax.grid(True)

## Outlier thresholds:
llabel = '%.2f%% of R^2 above %.1f' % (rr_hi_fraction*100., rr_hi_thresh)
axs3[0, 0].axhline(0.6, c='k', ls='--', label=llabel)
axs3[0, 0].legend(loc='upper left')
axs3[0, 0].plot(inner_bounds, (inner_thresh, inner_thresh), ls='--', c='k')


#axs3.grid(True)
#axs3[0, 0].set_ylim(ymax=1.0)
#axs3.legend(loc='upper left')
axs3[1, 0].scatter(every_rsep, every_rerr, **skw)
#fig3.savefig('prev_20251028_fig3.png', bbox_inches='tight')

for dax in axs4.ravel():
    dax.legend(loc='upper left')
    dax.grid(True)
for rax in qaxs4.values():
    rax.legend(loc='upper left')
    rax.set_ylim(-0.05, 1.0)
    rax.grid(True)
for xax in qaxs5.values():
    xax.legend(loc='upper left')
    xax.set_ylim(-1.0, 1.0)
    xax.grid(True)
for yax in qaxs6.values():
    yax.legend(loc='upper left')
    yax.set_ylim(-1.0, 1.0)
    yax.grid(True)
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()






sys.exit(0)






######################################################################
# CHANGELOG (11_test_fit_4pack.py):
#---------------------------------------------------------------------
#
#  2026-01-13:
#     -- Increased __version__ to 0.0.1.
#     -- First created 11_test_fit_4pack.py.
#

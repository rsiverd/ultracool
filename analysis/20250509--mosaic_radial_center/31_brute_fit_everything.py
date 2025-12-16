#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Test out my joint CRPIX model.
#
# Rob Siverd
# Created:       2025-10-16
# Last modified: 2025-11-11
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Optional matplotlib control:
#from matplotlib import use, rc, rcParams
#from matplotlib import use
#from matplotlib import rc
#from matplotlib import rcParams
#use('GTKAgg')  # use GTK with Anti-Grain Geometry engine
#use('agg')     # use Anti-Grain Geometry engine (file only)
#use('ps')      # use PostScript engine for graphics (file only)
#use('cairo')   # use Cairo (pretty, file only)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('font',**{'sans-serif':'Arial','family':'sans-serif'})
#rc('text', usetex=True) # enables text rendering with LaTeX (slow!)
#rcParams['axes.formatter.useoffset'] = False   # v. 1.4 and later
#rcParams['agg.path.chunksize'] = 10000
#rcParams['font.size'] = 10

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
import copy
import pickle
import pprint
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
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
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

## Helpers for this investigation:
import helpers
reload(helpers)

## Sensor geometry helper:
import sensor_geom
reload(sensor_geom)
sg = sensor_geom


##--------------------------------------------------------------------------##
## Projections with cartopy:
#try:
#    import cartopy.crs as ccrs
#    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#    from cartopy.feature.nightshade import Nightshade
#    #from cartopy import config as cartoconfig
#except ImportError:
#    sys.stderr.write("Error: cartopy module not found!\n")
#    sys.exit(1)

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

## Gaia stuf:
#gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d3.csv'
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc/gaia_calib1_NE.0d4.csv'
sys.stderr.write("Loading Gaia ... ")
gm.load_sources_csv(gaia_csv_path)
sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
## Images and data files:
quads = ['NE', 'NW', 'SE', 'SW']
ipath = {q:'slvd_11BQ02/solved_J_1325826p.%s.fits'%q     for q in quads}
cpath = {q:'data_11BQ02/wircam_J_1325826p.%s.fits.cat'%q for q in quads}
#gmask = 'blacklisted_gaia_matches.csv'

## Load the match mask if it exists:
gmask_file = 'data_11BQ02/gmatch_mask_J_1325826p.csv'
gmask_data = None
ngaia_want = 6862
if os.path.isfile(gmask_file):
    gmask_data = pd.read_csv(gmask_file, low_memory=False)
    ngaia_have = len(gmask_data)
    if ngaia_have != ngaia_want:
        sys.stderr.write("WARNING: unexpected len(gmask_data) (%d != %d) ...\n"
                         % (ngaia_have, ngaia_want))

#ne_ipath = 'slvd_11BQ02/solved_J_1325826p.NE.fits'
#nw_ipath = 'slvd_11BQ02/solved_J_1325826p.NW.fits'
#se_ipath = 'slvd_11BQ02/solved_J_1325826p.SE.fits'
#sw_ipath = 'slvd_11BQ02/solved_J_1325826p.SW.fits'

## Catalog paths:
#ne_cpath = 'data_11BQ02/wircam_J_1325826p.NE.fits.cat'
#nw_cpath = 'data_11BQ02/wircam_J_1325826p.NW.fits.cat'
#se_cpath = 'data_11BQ02/wircam_J_1325826p.SE.fits.cat'
#sw_cpath = 'data_11BQ02/wircam_J_1325826p.SW.fits.cat'

## Load images:
idata, ihdrs, imwcs = {}, {}, {}
for qq in quads:
    idata[qq], ihdrs[qq] = pf.getdata(ipath[qq], header=True)
    imwcs[qq] = awcs.WCS(ihdrs[qq])
#ne_idata, ne_ihdrs = pf.getdata(ne_ipath, header=True)
#nw_idata, nw_ihdrs = pf.getdata(nw_ipath, header=True)
#se_idata, se_ihdrs = pf.getdata(se_ipath, header=True)
#sw_idata, sw_ihdrs = pf.getdata(sw_ipath, header=True)

## Initialize Gaia matcher with appropriate time:
obs_time = helpers.wircam_timestamp_from_header(ihdrs['NE'])
gm.set_epoch(obs_time)

## Load catalogs:
cdata = {qq:pf.getdata(cc) for qq,cc in cpath.items()}

## Promote to DataFrame:
stars = {qq:pd.DataFrame.from_records(tt) for qq,tt in cdata.items()}

## Column-by-column fix for mixed-endian data tables (ick):
#asdf = stars['NE']
for ss in stars.values():
    for kk in ss.keys():
        dtbytes = ss[kk].dtype.byteorder
        if (dtbytes != '='):
            ss[kk] = ss[kk].values.byteswap().newbyteorder()
        #sys.stderr.write("%20s %s " % (kk, dtbytes))
        #if ( dtbytes == '=' ):
        #    sys.stderr.write("native!\n")
        #else:
        #    sys.stderr.write("NON-native!\n")
        #    ss[kk] = ss[kk].values.byteswap().newbyteorder()
        pass
#for ss in stars.values():
#    for kk in ss.keys():
#        dtbytes = ss[kk].dtype.byteorder
#        sys.stderr.write("%20s %s " % (kk, dtbytes))
#        if ( dtbytes == '=' ):
#            sys.stderr.write("native!\n")
#        else:
#            sys.stderr.write("NON-native!\n")
#            ss[kk] = ss[kk].values.byteswap().newbyteorder()
#        pass

## Embed approximate SNR/uncertainty. Assume:
## * sky + rdnoise^2 =~ 1000.0     (RDNOISE ~ 30)
## * sigma =~ FWHM / SNR
## NOTE: to calibrate, I used known-decent parameters and evaluated the
## following: median(sqrt((resid / asterr)**2))  ==>  2.64
## That factor of 2.64 is the factor by which the errors are underestimated
## This value is consistent with the ~0.05 pixel RMS seen previously.
_rescale = 2.64
_wirgain = 3.8                  # electrons per ADU
_fluxcol = 'FLUX_ISO'
for ss in stars.values():
    src_ele = _wirgain * ss[_fluxcol]
    src_snr = src_ele / np.sqrt(src_ele + 1000.0)
    ast_err = ss['FWHM_IMAGE'] / src_snr            # a
    ss['dumbsnr'] = src_snr
    ss['dumberr'] = ast_err
    ss['realerr'] = ast_err * _rescale



## Sensor centers, according to WCS:
xctr, yctr = 1024.5, 1024.5
imctrs = {qq:imwcs[qq].all_pix2world(xctr, yctr, 1, ra_dec_order=True) \
                                                    for qq in quads}
ra_ctrs, de_ctrs = np.array([x for x in imctrs.values()]).T
savg_ra, savg_de = angle.spheremean_deg(ra_ctrs, de_ctrs)

## Grab CD matrix elements from each image:
cdkeys = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
cdm_vals = {qq:np.array([ihdrs[qq][kk] for kk in cdkeys]) for qq in quads}
for qq,cdm in cdm_vals.items():
    print(helpers.analyze(cdm))

## Grab CRVALs from each image:
every_crval1 = np.array([ihdrs[qq]['CRVAL1'] for qq in quads])
every_crval2 = np.array([ihdrs[qq]['CRVAL2'] for qq in quads])
savg_cv1, savg_cv2 = angle.spheremean_deg(every_crval1, every_crval2)

## Grab CRPIX from each image:
every_crpix1 = np.array([ihdrs[qq]['CRPIX1'] for qq in quads])
every_crpix2 = np.array([ihdrs[qq]['CRPIX2'] for qq in quads])

## Calculate RA/DE, cross-match to Gaia, update star catalogs:
sensor_mtol = {'NE':0.3, 'NW':0.3, 'SE':0.3, 'SW':0.5}
sys.stderr.write("RA/DE calculation and Gaia matching ... ")
_xcol, _ycol = 'XWIN_IMAGE', 'YWIN_IMAGE'
mtol_arcsec = 0.3
gmag_limit  = 19.0
gm.set_Gmag_limit(gmag_limit)
sys.stderr.write("\n")
for qq,ss in stars.items():
    #csln = imwcs.get(qq)
    xpos, ypos = ss[_xcol], ss[_ycol]
    sra, sde = imwcs[qq].all_pix2world(xpos, ypos, 1, ra_dec_order=True)
    ss['anet_ra'], ss['anet_de'] = sra, sde
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

#[asdf[x].dtype.byteorder for x in asdf.keys()]

## Split off subsets of source catalogs with just the Gaia matches:
all_gstars, use_gstars = {}, {}
for qq,ss in stars.items():
    all_gstars[qq] = ss[ss.gid > 0].copy()
    use_gstars[qq] = ss[ss.gid > 0].copy()

#sys.exit(0)

## Region file maker:
def regify_gstars(filename, data, r1=3, r2=10):
    with open(filename, 'w') as fff:
        xcoo, ycoo = data['XWIN_IMAGE'], data['YWIN_IMAGE']
        for coo in zip(xcoo, ycoo):
            text = 'image; annulus(%9.3f, %9.3f, %.1f, %.1f)' % (*coo, r1, r2)
            fff.write(text + '\n')
            pass
        pass
    return

## Save these catalogs to file for separate analysis:
save_gdir = 'gmatches_31'
if not os.path.isdir(save_gdir):
    os.mkdir(save_gdir)
for qq,gg in all_gstars.items():
    sys.stderr.write("qq: %s\n" % qq)
    cbase = os.path.basename(cpath.get(qq))
    save_gcsv = '%s/%s.gaia.csv' % (save_gdir, cbase)
    save_greg = '%s/%s.gaia.reg' % (save_gdir, cbase)
    sys.stderr.write("save_gcsv: %s\n" % save_gcsv)
    gg.to_csv(save_gcsv, index=False)
    sys.stderr.write("save_greg: %s\n" % save_greg)
    regify_gstars(save_greg, gg)

#sys.exit(0)

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

## Eval test:
yaypars = np.array([-0.11849369, 1981.53392088, 156.12566409,
                        294.60495726, 35.13831321])
#yaypars = np.array([-0.12091838, -0.00000258, 2069.90785791, -308.87100033,
#                        294.59603615, 35.09902007])
yaypars = np.array([  -0.11493478, 2133.52441875, -140.01453587, 
                        294.58941299, 35.11319793])

yaypars = np.array([  -0.11493478, 2133.52441875, -140.01453587, 
                        294.58941299, 35.11319793])

# First attempt from solver:
yaypars = np.array([  -0.11493478, 2159.19182843, -238.56586468, 
                        294.58679834, 35.10499829])

# second attempt from solver:
yaypars = np.array([  -0.11493478, 2185.36416169, -268.85706722, 
                        294.58410422, 35.10244381])

# third attempt from solver:
yaypars = np.array([  -0.11579962, 2210.56641722, -286.40207281, 
                        294.58150973, 35.10096784])

# third attempt from solver:
yaypars = np.array([  -0.11702289, 2114.69715339, -181.08663971, 
                        294.59137184, 35.10984073])

# attempts with fmin:
yaypars = np.array([-0.11556858, 2042.39898725,  -72.60960425,
                        294.59880079, 35.11900887])
yaypars = np.array([-0.11677674, 2078.78185865,  -69.30868514,
                        294.59504417, 35.11929469])
yaypars = np.array([-0.11722164, 2073.36368714,  -67.21853003,
                        294.59560743, 35.11945769])


# ----------------------------------------------------------------------- 
# ----------------------------------------------------------------------- 

# Emit starting values based on existing 4-sensor model:
sensor_order = ['NE', 'NW', 'SE', 'SW']
sensor_qqmap = {kk:vv for kk,vv in enumerate(sensor_order)}
ne_pa_deg, ne_crpix1, ne_crpix2, cv1, cv2 = yaypars
model_4s_crpix = sg.get_4sensor_crpix(ne_crpix1, ne_crpix2)
model_4s_cdmat = helpers.make_four_cdmats(yaypars[0])
brute_param_guess = []
# Start with sensor-specific parameters:
for qq in sensor_order:
    sys.stderr.write("qq: %s\n" % qq)
    brute_param_guess.extend(model_4s_cdmat[qq])
    brute_param_guess.extend(model_4s_crpix[qq])
# Append CRVAL guess:
brute_param_guess.extend((cv1, cv2))
# Append distortion model:
brute_param_guess.extend([0.0, 0.0001, 0.00000268, 0.0])
# Promote to numpy:
brute_param_guess = np.array(brute_param_guess)
sys.stderr.write("brute_param_guess: %s\n" % str(brute_param_guess))
old_brute_param_guess = np.copy(brute_param_guess)
#sys.exit(0)

# Useful indices into the parameter array:
crpix1_idx = 4 + 6*np.arange(4)
crpix2_idx = 5 + 6*np.arange(4)


## Better initial guess (derotated):
#brute_param_guess = np.array([
#         -0.00008513,   -0.00000007,   -0.00000005,    0.00008509,
#       2071.81677382,  -69.53244894,   -0.00008515,    0.00000012,
#          0.00000012,    0.00008515, -111.89264852,  -60.06047991,
#         -0.00008512,    0.00000006,    0.00000006,    0.00008513,
#       2075.90424021, 2122.91922917,   -0.00008514,    0.00000014,
#          0.00000014,    0.00008508, -114.15493415, 2135.83383521,
#        294.59556737,   35.11929418,    0.00006127,    0.00010052,
#          0.00000273,    0.        ])

# Better initial guess (derotated):
brute_param_guess = np.array([
         -0.00008513,   -0.00000007,   -0.00000005,    0.00008509,
       2071.81677382,  -69.53244894,   -0.00008515,    0.00000012,
          0.00000012,    0.00008515, -111.89264852,  -60.06047991,
         -0.00008512,    0.00000006,    0.00000006,    0.00008513,
       2075.90424021, 2122.91922917,   -0.00008514,    0.00000014,
          0.00000014,    0.00008508, -114.15493415, 2135.83383521,
        294.59556737,   35.11929418,    0.00000000,    0.00000000,
          0.00000273,    0.        ])

# Better initial guess (further, lstq):
brute_param_guess = np.array([
     	 -0.00008509,   -0.00000009,   -0.00000005,    0.00008508,
       2069.20556007,  -75.37176152,   -0.00008511,    0.00000013,
          0.00000013,    0.00008511, -104.91672351,  -69.19102693,
         -0.00008508,    0.00000007,    0.00000005,    0.00008509,
       2073.01976981, 2113.84434071,   -0.00008507,    0.00000013,
          0.00000013,    0.00008505, -117.88136   , 2126.72097676,
        294.59587155,   35.11852314,    1.21092663,   -0.00188892,
          0.00000337,    0.        ])
##         0.00000337,    0.        , 0.])
##brute_param_guess[24] += 0.001

# EVEN BETTER initial guess (known radial profile):
brute_param_guess = np.array([
     	 -0.00008509,   -0.00000009,   -0.00000005,    0.00008508,
       2069.20556007,  -75.37176152,   -0.00008511,    0.00000013,
          0.00000013,    0.00008511, -104.91672351,  -69.19102693,
         -0.00008508,    0.00000007,    0.00000005,    0.00008509,
       2073.01976981, 2113.84434071,   -0.00008507,    0.00000013,
          0.00000013,    0.00008505, -117.88136   , 2126.72097676,
        294.59587155,   35.11852314, 
          0.009138186689690436,
          0.0024918987630597146, 
          -1.5464598039350375e-06,
          2.1778020126153285e-09,
          -3.3187074714126664e-13])
##         0.00000337,    0.        , 0.])
##brute_param_guess[24] += 0.001
brute_param_guess = brute_param_guess[:-5]   # trim the polynomial

brute_param_guess = np.array([
         -0.00008508,   -0.00000009,   -0.00000005,    0.00008507,
       2071.42878248,  -77.58024797,   -0.00008510,    0.00000013,
          0.00000013,    0.00008509, -112.88925890,  -68.55046592,
         -0.00008508,    0.00000007,    0.00000005,    0.00008508,
       2075.25551562, 2115.01759186,   -0.00008506,    0.00000015,
          0.00000013,    0.00008504, -115.47841845, 2127.56859459,
          294.59565158, 35.11860268])

brute_param_guess = np.array([
         -0.00008508,   -0.00000007,   -0.00000005,    0.00008507,
       2071.38069932,  -77.55544294,   -0.00008510,    0.00000012,
          0.00000012,    0.00008509, -112.73189620,  -68.65409208,
         -0.00008508,    0.00000007,    0.00000006,    0.00008508,
       2075.35273261, 2114.93167374,   -0.00008506,    0.00000014,
          0.00000014,    0.00008505, -115.36526041, 2127.63577728,
          294.59563838, 35.11860024])

# from MCMC:
brute_param_guess = np.array([
         -0.00008508,   -0.00000007,   -0.00000005,    0.00008507,
       2071.39243883,  -77.72577076,   -0.0000851 ,    0.00000012,
          0.00000012,    0.00008509, -112.73831575,  -68.825143  ,
         -0.00008508,    0.00000007,    0.00000006,    0.00008508,
       2075.34775651, 2114.75533714,   -0.00008506,    0.00000014,
          0.00000014,    0.00008504, -115.36232304, 2127.45977327,
        294.59563805,   35.11858534])


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


##-----------------------------------------------------------------------##
##-----------------------------------------------------------------------##

def calc_gutter_pixels(params):
    _npix = 2048
    ne_crpix1, nw_crpix1, se_crpix1, sw_crpix1 = params[crpix1_idx]
    ne_crpix2, nw_crpix2, se_crpix2, sw_crpix2 = params[crpix2_idx]
    n_width = ne_crpix1 - 2048.0 - nw_crpix1
    s_width = se_crpix1 - 2048.0 - sw_crpix1
    return n_width, s_width
    

horiz_gutter_pix = brute_param_guess
# Optional fixed offsets (TESTING):
offset_crpix1 = 0.0
offset_crpix2 = 0.0
#crpix1_idx = 4 + 6*np.arange(4)
#crpix2_idx = 5 + 6*np.arange(4)
brute_param_guess[crpix1_idx] += offset_crpix1   # nudge CRPIX1
brute_param_guess[crpix2_idx] += offset_crpix2   # nudge CRPIX2
#for ii in crpix1_idx:
#    brute_param_guess[ii] += offset_crpix1    # nudge CRPIX1
#for ii in crpix2_idx:
#    brute_param_guess[ii] += offset_crpix2    # nudge CRPIX2

sys.stderr.write("brute_param_guess (new): %s\n" % str(brute_param_guess))

# ----------------------------------------------------------------------- 
# ----------------------------------------------------------------------- 
fig_dims = (10, 10)
fig_dims = (8, 8)
fig, axs1 = plt.subplots(2, 2, figsize=fig_dims, num=1, clear=True)
[ax.set_aspect('equal', adjustable='box') for ax in axs1.flatten()]
ax1map  = {'NE':axs1[0, 0], 'NW':axs1[0, 1], 'SE':axs1[1, 0], 'SW':axs1[1,1]}

fig2, axs2 = plt.subplots(2, 2, figsize=fig_dims, num=2, clear=True)
ax2map = {'NE':axs2[0, 0], 'NW':axs2[0, 1], 'SE':axs2[1, 0], 'SW':axs2[1,1]}
test_crval1, test_crval2 = savg_cv1, savg_cv2
#test_crval1 -= 22.37 / 3600.0
#test_crval2 -= 22.37 / 3600.0
test_crpix1, test_crpix2 = every_crpix1[0], every_crpix2[0]
#test_crpix1, test_crpix2 = 2048.0, 1.0
#test_crval1, test_crval2 = 294.597953, 35.125201
#test_crpix1, test_crpix2 = 145.0, -145.0
#test_crpix1 = 2048 + 145.0
#test_crpix2 = 1.0  - 145.0
test_crpix1, test_crpix2 = 2122.691, -81.679
#test_crpix1 -= 1.0
#test_crpix2 -= 1.0
#test_crval1, test_crval2 = 294.590165, 35.118152
#test_crval1, test_crval2 = 294.5902476229806, 35.11840452525253
#test_crval1, test_crval2 = 294.59013858,  35.11828773
#test_crval1, test_crval2 = 294.59024157,  35.11837179
test_crval1, test_crval2 = 294.59047819,  35.11822976

test_crpix1, test_crpix2 = yaypars[1], yaypars[2]
#test_crval1, test_crval2 = 294.55798386,   35.12740249
test_crval1, test_crval2 = yaypars[3], yaypars[4]

#test_crpix1 += 2.0
#test_crpix2 -= 2.0
#test_crval1 += 0.0001
#test_crval2 -= 0.0001
#test_crval1 -= 0.0002
#test_crval2 += 0.0001
sensor_crpix = sg.get_4sensor_crpix(test_crpix1, test_crpix2)
#tt = list(sensor_crpix['SW'])
##tt[0] += 5.0
##tt[1] -= 2.0
#sensor_crpix['SW'] = tuple(tt)
pctcheck = [5, 95]
#cdm_vals['SW'][3] = 0.0000855
#cdm_calc = helpers.make_four_cdmats(-0.15)
#cdm_calc = helpers.make_four_cdmats(-0.05)
#cdm_calc = helpers.make_four_cdmats(-0.032)
#cdm_calc = helpers.make_four_cdmats(-0.0613)
cdm_calc = helpers.make_four_cdmats(yaypars[0])
#cdm_calc = helpers.make_four_cdmats(+0.032)
use_cdm_vals = cdm_calc
#use_cdm_vals = cdm_vals
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
qrrel, qrerr = {}, {}
rrel, rerr = [], []
approx_r2_coeff = 0.00000254
approx_r2_coeff = yaypars[1]
approx_r2_coeff = 0.00000254

#fig.tight_layout()
#fig2.tight_layout()
#sys.exit(0)

### MUST halt here if qrrel was not populated ...
#if len(qrrel) < 4:
#    sys.stderr.write("\nAborting: radial distance/distortion arrays empty!\n")
#    sys.exit(1)

# parabola to fit:
#def parabola(x, a, b, c):
#    return a + b*x + c*x*x
def parabola(x, a, c):
    return a + c*x*x

#fig3 = plt.figure(3)
#fig3.clf()
#ax3 = fig3.add_subplot(111)
#skw = {'lw':0, 's':10}
#qpopt, qpcov = {}, {}
#xtemp = np.linspace(250, 2750)
#for qq in quads:
#    xdata = qrrel.get(qq)
#    ydata = qrerr.get(qq)
#    qpopt[qq], qpcov[qq] = opti.curve_fit(parabola, xdata, ydata)
#    ypred = parabola(xdata, *qpopt[qq])
#    outly = ydata - ypred
#    ax3.scatter(xdata, ydata, label=qq, **skw)
#    ytemp = parabola(xtemp, *qpopt[qq])
#    ax3.plot(xtemp, ytemp, c='k')
##ax3.scatter(rrel, rerr, **skw)
#avg_popt = np.average(list(qpopt.values()), axis=0)
#avg_popt[0] = 0.0
#ytemp = parabola(xtemp, *avg_popt)
#ax3.plot(xtemp, ytemp, c='k', lw=5)
#ax3.legend(loc='upper left')

# Trial distortion correction:
#use_popt = avg_popt * np.array([0.0, 1.0, 1.0])

## FROM EXECUTION, this should be approximately correct:
guess_distmod = np.array([0.0, 0.0, 0.00000255])
#guess_distmod = np.array([0.0, 0.0, 0.00000267])
guess_distmod = np.array([0.0, 0.00138713, 0.00000205])
guess_distmod = np.array([0.0, 0.00138713, 0.00000205, 0.0])
guess_distmod = np.array([0.0, 0.00138713, 0.000013, 0.0])
guess_distmod = np.array([0.0, 0.00005971, 0.00000258, 0.0])
#guess_distmod = np.array([0.0, 0.00077147, 0.00000238, 0.0])
guess_distmod = np.array([-0.00001875, 0.00012424, 0.00000268, 0.0])
guess_distmod = np.array([0.009138186689690436, 0.0024918987630597146,
                         -1.5464598039350375e-06, 2.1778020126153285e-09,
                         -3.3187074714126664e-13])

guess_distmod = np.array([-7.481374e-02,  2.720092e-03, -1.770719e-06,
                           2.267144e-09, -3.437447e-13])

guess_distmod = np.array([-8.904428e-02,  2.715352e-03, -1.758365e-06,
                           2.264144e-09, -3.435743e-13])

guess_distmod = np.array([ 0.0000000000,  2.383655e-03, -1.398144e-06,
                           2.111627e-09, -3.211944e-13])

guess_distmod = np.array([ 1.757279e-01,  1.175609e-03,  1.047979e-06,
                           5.059445e-11,  4.450758e-13, -1.039322e-16])

#scatter(np.hypot(test_xrel, test_yrel), np.sqrt(np.sum(radial_evec**2, axis=0)))

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
        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
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

sys.stderr.write("Test evaluate badness (rdist version) ...\n")
##use_params = np.copy(yaypars)
##use_params = np.array(yaypars.tolist() + [0.0, 0.00000255, 0.0, 0.0])
##use_params = np.array(yaypars.tolist() + [0.0, 0.00000255]) #, 0.0, 0.0])
##use_params = np.array(yaypars.tolist() + [0.00055223, 0.00000255]) #, 0.0, 0.0])
##use_params = np.array(yaypars.tolist() + [0.000, 0.00000255]) #, 0.0, 0.0])
#use_params = np.array(yaypars.tolist() + [0.0, 0.000, 0.00000255]) #, 0.0, 0.0])
#use_params = np.array(yaypars.tolist() + [0.0, 0.000, 0.00000255, 0.0]) #, 0.0, 0.0])
##use_params[0] += 0.01
##tt_xres, tt_yres = eval_badness_foc2ccd(use_params)
##residuals = eval_badness_foc2ccd(use_params)
#
### Start in the middle of the gutters:
#use_params[1] = 2048.0 + 70.
#use_params[2] =    1.0 - 70.
use_params = brute_param_guess.copy()

## Optimize those parameters:
sys.stderr.write("Optimizing parameters ...\n")
#slvkw = {'loss':'soft_l1'}
typical_scale = np.array([0.01, 1.0, 1.0, 0.01, 0.01, 1e-5])
#slvkw = {'loss':'linear'}
#slvkw = {'loss':'linear', 'x_scale':typical_scale}
slvkw = {'method':'lm', 'xtol':1e-14, 'ftol':1e-14}
reskw = {'unsquared':True}
reskw = {'unsquared':True, 'snrweight':True}
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
#sys.exit(0)

#legible_lstq_pars = sift_params(answer['x'])
#legible_fmin_pars = sift_params(fanswer[0])

#_CHOICE = 'fmin'
_CHOICE = 'lstq'

if _CHOICE == 'fmin':
    fitted_pars = fanswer[0]
if _CHOICE == 'lstq':
    fitted_pars = answer['x']

sifted_pars = sift_params(fitted_pars)

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
#_do_MCMC = False
_xy_rmsd = 0.057
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
    #niter, thinned = 2000, 5
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

## ----------------------------------------------------------------------- ##


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

#fig4 = plt.figure(4)
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

#    xdata = qrrel.get(qq)
#    ydata = qrerr.get(qq)
#    qpopt[qq], qpcov[qq] = opti.curve_fit(parabola, xdata, ydata)
#    ypred = parabola(xdata, *qpopt[qq])
#    outly = ydata - ypred
#    ax3.scatter(xdata, ydata, label=qq, **skw)
#    ytemp = parabola(xtemp, *qpopt[qq])
#    ax3.plot(xtemp, ytemp, c='k')
#ax3.scatter(rrel, rerr, **skw)
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()






sys.exit(0)


######################################################################
# CHANGELOG (11_joint_crpix_test.py):
#---------------------------------------------------------------------
#
#  2025-05-09:
#     -- Increased __version__ to 0.0.1.
#     -- First created 11_joint_crpix_test.py.
#

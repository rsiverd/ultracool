#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Test out my joint CRPIX model.
#
# Rob Siverd
# Created:       2025-05-09
# Last modified: 2025-05-09
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
## Image paths:
quads = ['NE', 'NW', 'SE', 'SW']
ipath = {q:'slvd_11BQ02/solved_J_1325826p.%s.fits'%q     for q in quads}
cpath = {q:'data_11BQ02/wircam_J_1325826p.%s.fits.cat'%q for q in quads}

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
gstars = {}
for qq,ss in stars.items():
    gstars[qq] = ss[ss.gid > 0]

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
save_gdir = 'gmatches'
if not os.path.isdir(save_gdir):
    os.mkdir(save_gdir)
for qq,gg in gstars.items():
    sys.stderr.write("qq: %s\n" % qq)
    cbase = os.path.basename(cpath.get(qq))
    save_gcsv = '%s/%s.gaia.csv' % (save_gdir, cbase)
    save_greg = '%s/%s.gaia.reg' % (save_gdir, cbase)
    sys.stderr.write("save_gcsv: %s\n" % save_gcsv)
    gg.to_csv(save_gcsv, index=False)
    sys.stderr.write("save_greg: %s\n" % save_greg)
    regify_gstars(save_greg, gg)

#sys.exit(0)

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
for qq,gst in gstars.items():
    tcpx1, tcpx2 = sensor_crpix.get(qq)
    gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
    cdmcrv = np.array(use_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
    test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                gstars[qq]['gra'], gstars[qq]['gde'])
    test_xccd = test_xrel + tcpx1
    test_yccd = test_yrel + tcpx2
    x_error = test_xccd - gxx.values
    y_error = test_yccd - gyy.values
    #ax1map.get(qq).quiver(gxx, gyy, x_error, y_error, **qxkw)
    ## error percentiles:
    #xerrpct = np.percentile(x_error, pctcheck)
    #yerrpct = np.percentile(y_error, pctcheck)
    #sys.stderr.write("%s X 5th/95th: %7.3f, %7.3f\n" % (qq, *xerrpct))
    #sys.stderr.write("%s Y 5th/95th: %7.3f, %7.3f\n" % (qq, *yerrpct))

    # unit vector towards CRPIX:
    test_xy_rel = np.array((test_xrel, test_yrel))
    _vec_length = np.sqrt(np.sum(test_xy_rel**2, axis=0))
    _src_v_unit = test_xy_rel / _vec_length
    _src_v_errs = np.array((x_error, y_error))
    # radial distance (towards CRPIX):
    _src_R_dist = np.hypot(test_xrel, test_yrel)
    # radial distortion correction:
    _src_R_corr = approx_r2_coeff * _src_R_dist**2
    _src_v_corr = _src_R_corr * _src_v_unit
    xcorrection, ycorrection = _src_v_corr
    # total errors after radial adjustment:
    _tot_v_errs = _src_v_errs + _src_v_corr
    corr_x_error, corr_y_error = _tot_v_errs
    # show me!
    ax1map.get(qq).quiver(gxx, gyy,  xcorrection,  ycorrection, color='k', **qxkw)
    ax2map.get(qq).quiver(gxx, gyy, corr_x_error, corr_y_error, color='r', **qxkw)

    # magnitude of error radial component:
    radial_emag = x_error*_src_v_unit[0] + y_error*_src_v_unit[1]
    # error radial component as vector:
    radial_evec = _src_v_unit * radial_emag
    nonrad_evec = _src_v_errs - radial_evec

    nonrad_emag = np.sqrt(np.sum(nonrad_evec**2, axis=0))  # size of nonrad err
    implied_rot = np.degrees(nonrad_emag / _src_R_dist)

    # SPEW UNIT VECTORS:
    radial_unit_avg = np.median(_src_v_unit, axis=1)
    radial_dist_avg = np.median(test_xy_rel, axis=1)
    radial_errs_avg = np.median(_src_v_errs, axis=1)
    radial_corr_avg = np.median(_src_v_corr, axis=1)
    radial_evec_avg = np.median(radial_evec, axis=1)
    nonrad_evec_avg = np.median(nonrad_evec, axis=1)
    
    sys.stderr.write("----------------------\n")
    sys.stderr.write("%s quad\n" % qq)
    sys.stderr.write("avg radial unit: %s\n" % str(radial_unit_avg))
    sys.stderr.write("avg radial dist: %s\n" % str(radial_dist_avg))
    sys.stderr.write("avg radial errs: %s\n" % str(radial_errs_avg))
    sys.stderr.write("avg radial corr: %s\n" % str(radial_corr_avg))
    sys.stderr.write("avg radial evec: %s\n" % str(radial_evec_avg))
    sys.stderr.write("avg nonrad evec: %s\n" % str(nonrad_evec_avg))
    #pass
    #break

    #errdot = x_error*_src_v_unit[0] + y_error*_src_v_unit[1]
    # magnitude of error radial component:
    radial_emag = x_error*_src_v_unit[0] + y_error*_src_v_unit[1]
    # error radial component as vector:
    radial_evec = _src_v_unit * radial_emag
    nonrad_evec = _src_v_errs - radial_evec
    # stash:
    qrrel[qq] = _src_R_dist #np.hypot(test_xrel, test_yrel)
    qrerr[qq] = np.sqrt(np.sum(radial_evec**2, axis=0))
    rrel.extend(_src_R_dist) #np.hypot(test_xrel, test_yrel).tolist())
    rerr.extend(np.sqrt(np.sum(radial_evec**2, axis=0)).tolist())

    #break

fig.tight_layout()
fig2.tight_layout()
#sys.exit(0)

## MUST halt here if qrrel was not populated ...
if len(qrrel) < 4:
    sys.stderr.write("\nAborting: radial distance/distortion arrays empty!\n")
    sys.exit(1)

# parabola to fit:
#def parabola(x, a, b, c):
#    return a + b*x + c*x*x
def parabola(x, a, c):
    return a + c*x*x

fig3 = plt.figure(3)
fig3.clf()
ax3 = fig3.add_subplot(111)
skw = {'lw':0, 's':10}
qpopt, qpcov = {}, {}
xtemp = np.linspace(250, 2750)
for qq in quads:
    xdata = qrrel.get(qq)
    ydata = qrerr.get(qq)
    qpopt[qq], qpcov[qq] = opti.curve_fit(parabola, xdata, ydata)
    ypred = parabola(xdata, *qpopt[qq])
    outly = ydata - ypred
    ax3.scatter(xdata, ydata, label=qq, **skw)
    ytemp = parabola(xtemp, *qpopt[qq])
    ax3.plot(xtemp, ytemp, c='k')
#ax3.scatter(rrel, rerr, **skw)
avg_popt = np.average(list(qpopt.values()), axis=0)
avg_popt[0] = 0.0
ytemp = parabola(xtemp, *avg_popt)
ax3.plot(xtemp, ytemp, c='k', lw=5)
ax3.legend(loc='upper left')

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

#scatter(np.hypot(test_xrel, test_yrel), np.sqrt(np.sum(radial_evec**2, axis=0)))

## Polynomial model. Hopefully this is a strictly positive value.
def poly_eval(r, model):
    #return model[0] + model[1]*r + model[2]*r*r
    #return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r
    return model[0] + model[1]*r + model[2]*r*r + model[3]*r*r*r + model[4]*r*r*r*r

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

### ----------------------------------------------------------------------- ##
### As noted above, this convention SUBTRACTS calculated x- and y-corrections
### from RA/DE-derived xrel/yrel coordinates before comparing to measured X,Y.
### Contents of 'params' array:
### --> NE_pos_ang_deg, NE_CRPIX1, NE_CRPIX2, CRVAL1, CRVAL2
### * CRVAL1
#def eval_badness_foc2ccd(params):
#    # parse parameters
#    ne_pa_deg, ne_crpix1, ne_crpix2, test_crval1, test_crval2 = params
#    # initialize CRPIX and CDM for all 4 sensors 
#    sensor_crpix = sg.get_4sensor_crpix(ne_crpix1, ne_crpix2)
#    test_cdm_calc = helpers.make_four_cdmats(ne_pa_deg)
#    test_cdm_vals = test_cdm_calc
#    # note average star count for normalization
#    avg_nstars = np.average([len(x) for x in gstars.values()])
#    qxres, qyres = {}, {}
#    xres, yres = [], []
#    for qq,gst in gstars.items():
#        nstar_scale_factor = np.sqrt(avg_nstars / float(len(gst)))
#        tcpx1, tcpx2 = sensor_crpix.get(qq)
#        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
#        cdmcrv = np.array(test_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
#        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
#                                    gstars[qq]['gra'], gstars[qq]['gde'])
#        #import pdb; pdb.set_trace()
#        #breakpoint()
#        xnudge, ynudge = calc_rdist_corrections(test_xrel, test_yrel, guess_distmod)
#        test_xccd = test_xrel + xnudge + tcpx1
#        test_yccd = test_yrel + ynudge + tcpx2
#        x_error = test_xccd - gxx.values
#        y_error = test_yccd - gyy.values
#        scaled_xerr = x_error * nstar_scale_factor
#        scaled_yerr = y_error * nstar_scale_factor
#        #qxres[qq] = scaled_xerr
#        #qyres[qq] = scaled_yerr
#        xres.extend(scaled_xerr)
#        yres.extend(scaled_yerr)
#        #xres.extend(xres)
#        pass
#    #return qxres, qyres
#    return xres, yres
#
### Square and concatenate for minimization:
#def squared_residuals_foc2ccd(params):
#    return np.concatenate(eval_badness_foc2ccd(params))**2
#
#sys.stderr.write("Test evaluate badness ...\n")
#use_params = np.copy(yaypars)
#tt_xres, tt_yres = eval_badness_foc2ccd(use_params)
##residuals = eval_badness_foc2ccd(use_params)
#
### Optimize those parameters:
#sys.stderr.write("Optimizing parameters ...\n")
#answer = opti.least_squares(squared_residuals_foc2ccd, use_params)
#sys.stderr.write("Ended up with: %s\n" % str(answer))
#sys.stderr.write("Ended up with: %s\n" % str(answer['x']))

## ----------------------------------------------------------------------- ##

## ----------------------------------------------------------------------- ##
## As noted above, this convention SUBTRACTS calculated x- and y-corrections
## from RA/DE-derived xrel/yrel coordinates before comparing to measured X,Y.
## THIS VERSION ALSO FITS RADIAL DISTORTION PARAMETERS
## Contents of 'params' array:
## --> NE_pos_ang_deg, NE_CRPIX1, NE_CRPIX2, CRVAL1, CRVAL2, rcoef2
## * CRVAL1
def squared_residuals_foc2ccd_rdist(params, diags=False):
    # parse parameters
    #ne_pa_deg, ne_crpix1, ne_crpix2, test_crval1, test_crval2,  = params
    ne_pa_deg, ne_crpix1, ne_crpix2 = params[0:3]
    test_crval1, test_crval2 = params[3:5]
    rdist_pars = params[5:]
    #rcoef1, rcoef2, rcoef3, rcoef4 = params[5:]
    #rcoef3 = 0.0
    #rcoef4 = 0.0
    #coeffs = params[5:]
    #test_distmod = np.array([0.0] + coeffs.tolist())
    nextra = 5 - len(rdist_pars)
    #test_distmod = np.array([0.0, rcoef1, rcoef2, rcoef3, rcoef4]) 
    test_distmod = rdist_pars.tolist() + [0.0]*nextra
    #test_distmod[1] = 0.0
    # initialize CRPIX and CDM for all 4 sensors 
    test_sensor_crpix = sg.get_4sensor_crpix(ne_crpix1, ne_crpix2)
    test_cdm_calc = helpers.make_four_cdmats(ne_pa_deg)
    test_cdm_vals = test_cdm_calc
    # note average star count for normalization
    avg_nstars = np.average([len(x) for x in gstars.values()])
    typical_rdist = 1448.0     # 0.5 * np.hypot(2048, 2048)
    qxres, qyres = {}, {}
    xres, yres = [], []
    diag_data = {}
    for qq,gst in gstars.items():
        nstar_scale_factor = np.sqrt(avg_nstars / float(len(gst)))
        tcpx1, tcpx2 = test_sensor_crpix.get(qq)
        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        cdmcrv = np.array(test_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    gstars[qq]['gra'], gstars[qq]['gde'])
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

        #scaled_xerr *= test_rrel / typical_rdist    # more weight far away
        #scaled_yerr *= test_rrel / typical_rdist    # more weight far away
        scaled_xerr *= np.sqrt(test_rrel / typical_rdist)    # more weight far away
        scaled_yerr *= np.sqrt(test_rrel / typical_rdist)    # more weight far away
        #qxres[qq] = scaled_xerr
        #qyres[qq] = scaled_yerr
        xres.extend(scaled_xerr)
        yres.extend(scaled_yerr)
        #xres.extend(xres)
        if diags:
            diag_data[qq] = { "xmeas":gxx,
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
#            diag_data[qq] = {'rdist':test_rrel,
#                             'xerror':x_error,
#                             'yerror':y_error,
#                             'rerror':np.hypot(x_error, y_error),
#                             'scaled_xerror':scaled_xerr,
#                             'scaled_yerror':scaled_yerr,
#                             'scaled_rerror':np.hypot(scaled_xerr, scaled_yerr),
#                            }

        pass
    #return qxres, qyres
    #return xres, yres
    if diags:
        return diag_data
    return np.concatenate((xres, yres))**2

def fmin_squared_residuals_foc2ccd_rdist(params):
    return np.sum(squared_residuals_foc2ccd_rdist(params))

sys.stderr.write("Test evaluate badness (rdist version) ...\n")
#use_params = np.copy(yaypars)
#use_params = np.array(yaypars.tolist() + [0.0, 0.00000255, 0.0, 0.0])
#use_params = np.array(yaypars.tolist() + [0.0, 0.00000255]) #, 0.0, 0.0])
#use_params = np.array(yaypars.tolist() + [0.00055223, 0.00000255]) #, 0.0, 0.0])
#use_params = np.array(yaypars.tolist() + [0.000, 0.00000255]) #, 0.0, 0.0])
use_params = np.array(yaypars.tolist() + [0.0, 0.000, 0.00000255]) #, 0.0, 0.0])
use_params = np.array(yaypars.tolist() + [0.0, 0.000, 0.00000255, 0.0]) #, 0.0, 0.0])
#use_params[0] += 0.01
#tt_xres, tt_yres = eval_badness_foc2ccd(use_params)
#residuals = eval_badness_foc2ccd(use_params)

## Start in the middle of the gutters:
use_params[1] = 2048.0 + 70.
use_params[2] =    1.0 - 70.

## Optimize those parameters:
sys.stderr.write("Optimizing parameters ...\n")
#slvkw = {'loss':'soft_l1'}
typical_scale = np.array([0.01, 1.0, 1.0, 0.01, 0.01, 1e-5])
#slvkw = {'loss':'linear'}
#slvkw = {'loss':'linear', 'x_scale':typical_scale}
slvkw = {}
answer = opti.least_squares(squared_residuals_foc2ccd_rdist, use_params, **slvkw)
sys.stderr.write("Ended up with: %s\n" % str(answer))
sys.stderr.write("Ended up with: %s\n" % str(answer['x']))

sys.stderr.write("\n\n\nTry again with fmin ....\n")
fmkw = {'full_output':True, 'xtol':1e-5}
fanswer = opti.fmin(fmin_squared_residuals_foc2ccd_rdist, use_params, **fmkw)
print(fanswer[0])

## ----------------------------------------------------------------------- ##
## Diagnostics time ...

diag_data = squared_residuals_foc2ccd_rdist(fanswer[0], diags=True)

swdiags = diag_data['SW']
#ststars = 
rerrs = np.hypot(swdiags['xerror'], swdiags['yerror'])
worst10idx = np.argsort(rerrs)[-10:]
gstars['SW'].iloc[worst10idx]
#xworst = 




fig4, axs4 = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=4, clear=True)
qaxs4 = {'NE':axs4[0,0], 'NW':axs4[0,1], 'SE':axs4[1,0], 'SW':axs4[1,1]}

fig5, axs5 = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=5, clear=True)
qaxs5 = {'NE':axs5[0,0], 'NW':axs5[0,1], 'SE':axs5[1,0], 'SW':axs5[1,1]}

fig6, axs6 = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=6, clear=True)
qaxs6 = {'NE':axs6[0,0], 'NW':axs6[0,1], 'SE':axs6[1,0], 'SW':axs6[1,1]}

#fig4 = plt.figure(4)
#fig4.clf()
#ax4 = fig4.add_subplot(111)
skw = {'lw':0, 's':10}
qpopt, qpcov = {}, {}
xtemp = np.linspace(250, 2750)
for qq,ddata in diag_data.items():
    # R error:
    rax = qaxs4[qq]
    rax.set_title(qq)
    rax.scatter(ddata['rdist'], ddata['rerror'], label='raw', **skw)
    rax.scatter(ddata['rdist'], ddata['scaled_rerror'], label='scl', **skw)
    rax.set_ylabel('R error [pix]')
    # X error:
    xax = qaxs5[qq]
    xax.scatter(ddata['rdist'], ddata['xerror'], label='raw', **skw)
    xax.scatter(ddata['rdist'], ddata['scaled_xerror'], label='scl', **skw)
    xax.set_title(qq)
    xax.set_ylabel('X error [pix]')
    # Y error:
    yax = qaxs6[qq]
    yax.set_title(qq)
    yax.scatter(ddata['rdist'], ddata['yerror'], label='raw', **skw)
    yax.scatter(ddata['rdist'], ddata['scaled_yerror'], label='scl', **skw)
    yax.set_ylabel('Y error [pix]')
    pass


for rax in qaxs4.values():
    rax.legend(loc='upper left')
for xax in qaxs5.values():
    xax.legend(loc='upper left')
for yax in qaxs6.values():
    yax.legend(loc='upper left')


#    xdata = qrrel.get(qq)
#    ydata = qrerr.get(qq)
#    qpopt[qq], qpcov[qq] = opti.curve_fit(parabola, xdata, ydata)
#    ypred = parabola(xdata, *qpopt[qq])
#    outly = ydata - ypred
#    ax3.scatter(xdata, ydata, label=qq, **skw)
#    ytemp = parabola(xtemp, *qpopt[qq])
#    ax3.plot(xtemp, ytemp, c='k')
#ax3.scatter(rrel, rerr, **skw)
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()









sys.exit(0)

## ----------------------------------------------------------------------- ##
## As noted above, this convention SUBTRACTS calculated x- and y-corrections
## from RA/DE-derived xrel/yrel coordinates before comparing to measured X,Y.
## V3 will look directly at the radialness of errors ...
## Contents of 'params' array:
## --> NE_pos_ang_deg, NE_CRPIX1, NE_CRPIX2, CRVAL1, CRVAL2
## * CRVAL1
def squared_residuals_foc2ccd_rbinerr(params):
    # parse parameters
    ne_pa_deg, ne_crpix1, ne_crpix2, test_crval1, test_crval2 = params
    # initialize CRPIX and CDM for all 4 sensors 
    sensor_crpix = sg.get_4sensor_crpix(ne_crpix1, ne_crpix2)
    test_cdm_calc = helpers.make_four_cdmats(ne_pa_deg)
    test_cdm_vals = test_cdm_calc
    # note average star count for normalization
    avg_nstars = np.average([len(x) for x in gstars.values()])
    qxres, qyres = {}, {}
    xres, yres = [], []
    every_rrel = []
    every_rerr = []
    for qq,gst in gstars.items():
        nstar_scale_factor = np.sqrt(avg_nstars / float(len(gst)))
        tcpx1, tcpx2 = sensor_crpix.get(qq)
        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        cdmcrv = np.array(test_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    gstars[qq]['gra'], gstars[qq]['gde'])
        #import pdb; pdb.set_trace()
        #breakpoint()
        xnudge, ynudge = calc_rdist_corrections(test_xrel, test_yrel, guess_distmod)
        test_xccd = test_xrel + xnudge + tcpx1
        test_yccd = test_yrel + ynudge + tcpx2
        x_error = test_xccd - gxx.values
        y_error = test_yccd - gyy.values
        scaled_xerr = x_error * nstar_scale_factor
        scaled_yerr = y_error * nstar_scale_factor
        #qxres[qq] = scaled_xerr
        #qyres[qq] = scaled_yerr
        xres.extend(scaled_xerr)
        yres.extend(scaled_yerr)
        test_rrel = np.hypot(test_xrel, test_yrel)
        test_rerr = np.hypot(scaled_xerr, scaled_yerr)
        every_rrel.extend(test_rrel)
        every_rerr.extend(test_rerr)
        #xres.extend(xres)
        pass
    #return qxres, qyres

    return np.concatenate((xres, yres))**2

sys.stderr.write("Test evaluate badness ...\n")
use_params = np.copy(yaypars)
tt_xres, tt_yres = eval_badness_foc2ccd(use_params)
#residuals = eval_badness_foc2ccd(use_params)

## Optimize those parameters:
sys.stderr.write("Optimizing parameters ...\n")
answer = opti.least_squares(squared_residuals_foc2ccd, use_params)
sys.stderr.write("Ended up with: %s\n" % str(answer))
sys.stderr.write("Ended up with: %s\n" % str(answer['x']))


## ----------------------------------------------------------------------- ##

sys.exit(0)

    #if qq == 'SE':
    #    break
    #xyccd[qq] = 

## Make a chart:
# sharex='col' | sharex='row'


### Set up WCS objects:
#ne_imwcs = awcs.WCS(ne_ihdrs)
#nw_imwcs = awcs.WCS(nw_ihdrs)
#se_imwcs = awcs.WCS(se_ihdrs)
#sw_imwcs = awcs.WCS(sw_ihdrs)

### Load star catalogs:
#ne_stars = pf.getdata(ne_cpath)
#nw_stars = pf.getdata(nw_cpath)
#se_stars = pf.getdata(se_cpath)
#sw_stars = pf.getdata(sw_cpath)

##--------------------------------------------------------------------------##
## Self-consistent CRPIX1/2 generator:
#def get_nwsesw_crpix(ne_crpix1, ne_crpix2):
#    dx = ne_crpix1 - 2048.0
#    dy = ne_crpix2 -    1.0
#    nw_crpix1 = -134.580 + 0.9999522 * dx + 0.0042017 * dy
#    nw_crpix2 =    9.697 + 0.0063453 * dx + 0.9998133 * dy
#    se_crpix1 = 2052.228 + 1.0022410 * dx + 0.0027602 * dy
#    se_crpix2 = 2192.539 + 0.0055770 * dx + 0.9998227 * dy
#    sw_crpix1 = -137.002 + 1.0026923 * dx - 0.0001494 * dy
#    sw_crpix2 = 2195.456 + 0.0024810 * dx + 1.0004179 * dy
#    return ((nw_crpix1, nw_crpix2), 
#            (se_crpix1, se_crpix2),
#            (sw_crpix1, sw_crpix2))




##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (9, 8)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
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
#ax1.xaxis.set_major_formatter(fptformat) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
#ax1.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#ax1.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'], rotation=45)
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

#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')


######################################################################
# CHANGELOG (11_joint_crpix_test.py):
#---------------------------------------------------------------------
#
#  2025-05-09:
#     -- Increased __version__ to 0.0.1.
#     -- First created 11_joint_crpix_test.py.
#

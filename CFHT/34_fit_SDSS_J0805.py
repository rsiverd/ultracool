#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Analyze SDSS J0805 data, fit simultaneous astrometric and orbital solution.
#
# Rob Siverd
# Created:       2024-04-29
# Last modified: 2024-04-29
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
import resource
import signal
#import glob
import gc
import os
import sys
import time
import pickle
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

## MCMC sampler and plotter:
import emcee
import corner

## Astrometry fitting module:
import astrom_test_2
reload(astrom_test_2)
at2 = astrom_test_2
#af  = at2.AstFit()  # used for target

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
try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

## For Earth position:
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import GCRS, ICRS

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

##--------------------------------------------------------------------------##
## New-style string formatting (more at https://pyformat.info/):

#oldway = '%s %s' % ('one', 'two')
#newway = '{} {}'.format('one', 'two')

#oldway = '%d %d' % (1, 2)
#newway = '{} {}'.format(1, 2)

# With padding:
#oldway = '%10s' % ('test',)        # right-justified
#newway = '{:>10}'.format('test')   # right-justified
#oldway = '%-10s' % ('test',)       #  left-justified
#newway = '{:10}'.format('test')    #  left-justified

# Ordinally:
#newway = '{1} {0}'.format('one', 'two')     # prints "two one"

# Dictionarily:
#newway = '{lastname}, {firstname}'.format(firstname='Rob', lastname='Siverd')

# Centered (new-only):
#newctr = '{:^10}'.format('test')      # prints "   test   "

# Numbers:
#oldway = '%06.2f' % (3.141592653589793,)
#newway = '{:06.2f}'.format(3.141592653589793)

##--------------------------------------------------------------------------##
## On-the-fly file modifications:
#def fix_hashes(filename):
#    with open(filename, 'r') as ff:
#        for line in ff:
#            if line.startswith('#'):
#                if ('=' in line):
#                    continue                # param, ignore
#                else:
#                    yield line.lstrip('#')  # header, keep
#            else:
#                yield line

#def analyze_header(filename):
#    skip_rows = 0
#    col_names = []
#    with open(filename, 'r') as ff:
#        for line in ff:
#            if line.startswith('#'):
#                skip_rows += 1
#                if ('=' in line):
#                    continue
#                else:
#                    hline = line.rstrip()
#                    col_names = hline.lstrip('#').split()
#                    continue
#            else:
#                #sys.stderr.write("Found data ... stopping.\n")
#                break
#    return skip_rows, col_names

##--------------------------------------------------------------------------##
## Load pickled science target dataset:
data_file = 'clean_J0805.pickle'
sdss0805 = load_pickled_object(data_file)

## Load data into fitter:
afn = at2.AstFit()  # used for target
afn.setup(sdss0805, ra_key='dtr_dra', de_key='dtr_dde')

## Fit astrometry:
sig_thresh = 3
maxiters = 30
bestpars = afn.fit_bestpars(sigcut=sig_thresh)
iterpars = bestpars.copy()
for i in range(maxiters):
    sys.stderr.write("Iteration %d ...\n" % i)
    iterpars = afn.iter_update_bestpars(iterpars)
    if afn.is_converged():
        sys.stderr.write("Converged!  Iteration: %d\n" % i)
        break

fitpars = afn.get_latest_params()
#bestfit_ra_deg = np.degrees
ra_deg, de_deg, pmra_mas, pmde_mas, prlx_mas = afn.nice_units(fitpars)
sys.stderr.write("\n"
               + "------------------------------\n"
               + "RA       (deg):  %12.6f\n" % ra_deg
               + "DE       (deg):  %12.6f\n" % de_deg
               + "pmRA* (mas/yr):  %12.6f\n" % pmra_mas
               + "pmDE  (mas/yr):  %12.6f\n" % pmde_mas
               + "prlx     (mas):  %12.6f\n" % prlx_mas
               + "------------------------------\n")


##--------------------------------------------------------------------------##
## Quick ASCII I/O:
#data_file = 'data.txt'
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

data_file = 'clean_J0805.csv'
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
#all_data = pd.read_csv(data_file)
j0805 = pd.read_csv(data_file, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)
#nskip, cnames = analyze_header(data_file)
#all_data = pd.read_csv(data_file, names=cnames, skiprows=nskip, **pdkwargs)
#all_data = pd.DataFrame.from_records(npy_data)
#all_data = pd.DataFrame(all_data.byteswap().newbyteorder()) # for FITS tables


## Existing solution:
params = {
     'name'             :  'SDSS0805',
   # 'ra_deg'           : 121.3813807,
   # 'de_deg'           :  48.2094111,
   #'pmra_cosdec_asyr'   :  -0.4583,        # pmRA * cos(dec)
   #'pmde_asyr'          :   0.0498,
   #'pmra_cosdec_asyr'   :  -0.443769,      # pmRA * cos(dec)
   #'pmra_cosdec_asyr'   :  -0.461000,      # pmRA * cos(dec)
   #'pmde_asyr'          :   0.041000,
   #'parallax_as'        :   0.0431,
   #'epoch_jdutc'        :   2454429.1,
   # RJS fit:
    'ra_deg'            : 121.38118137,
    'de_deg'            :  48.20929239,
   'pmra_cosdec_asyr'   :  -0.460877,      # pmRA * cos(dec)
   'pmde_asyr'          :   0.0426226,
   'parallax_as'        :   0.03959883,
   'epoch_jdutc'        :   2454429.1,
   'epoch_jdtdb'        :   2454429.099542902,
}

## Use the fit above:
#fitted_pars = afn.get_latest_par
params['epoch_jdtdb'] = afn.ref_tdb
params[     'ra_deg'] = 121.38118086
params[     'de_deg'] =  48.20929227



cos_dec   = np.cos(np.radians(j0805['de_deg']))
dt_years  = (j0805['jdtdb'] - params['epoch_jdtdb']) / 365.25
delta_ra_arcsec = params['pmra_cosdec_asyr'] / cos_dec * dt_years
delta_ra_deg    = delta_ra_arcsec / 3600.0
delta_de_arcsec = params['pmde_asyr'] * dt_years
delta_de_deg    = delta_de_arcsec / 3600.0
prev_soln_ra = params['ra_deg'] + delta_ra_deg
prev_soln_de = params['de_deg'] + delta_de_deg


resid_ra = j0805['ra_deg'] - prev_soln_ra
resid_de = j0805['de_deg'] - prev_soln_de

jpts = 300
nyrs = 12.0
filled_dt_years = np.arange(jpts) / float(jpts) * nyrs
filled_jd_range = filled_dt_years * 365.25 + params['epoch_jdtdb']
date_range = astt.Time(filled_jd_range, scale='utc', format='jd')



## Fix offsets:
#resid_ra -= np.median(resid_ra)
#resid_de -= np.median(resid_de)

## Convert to mas:
#resid_ra *= 3.6e6
#resid_de *= 3.6e6

## Use results from fitter:
resid_ra, resid_de = afn.get_radec_minus_prmot_mas(cos_dec_mult=True)
#ra_pmra_model_deg, de_pmra_model_deg = afn.get_bestfit_prmot_deg()
#resid_ra
resid_ra_all, resid_de_all = afn.get_radec_minus_model_mas(cos_dec_mult=True)
jdtdb_all = afn.dataset['jdtdb']

## Generate groups in JDTDB for averaging. A new group starts when the
## difference exceeds daytol (about half a day).
ep_tol_days = 0.5
grp_num = 0
ep_list = [0]
prev_jd = jdtdb_all[0]
for jj in jdtdb_all[1:]:
    if (jj - prev_jd) > ep_tol_days:
        # next group
        grp_num += 1
    #if (jj - prev_jd) < ep_tol_days:
        # same group
    ep_list.append(grp_num)
    prev_jd = jj
    pass
ep_list = np.array(ep_list)

## Per-epoch medians:
epoch_avg_ra_resid = []
epoch_med_ra_resid = []
epoch_avg_de_resid = []
epoch_med_de_resid = []
epoch_avg_jdtdb    = []
epoch_med_jdtdb    = []
for ep in np.unique(ep_list):
    which = (ep_list == ep)
    these_ra = resid_ra_all[which]
    these_de = resid_de_all[which]
    these_jd = jdtdb_all[which]
    epoch_avg_jdtdb.append(np.mean(these_jd))
    epoch_med_jdtdb.append(np.median(these_jd))
    epoch_avg_ra_resid.append(np.mean(these_ra))
    epoch_med_ra_resid.append(np.median(these_ra))
    epoch_avg_de_resid.append(np.mean(these_de))
    epoch_med_de_resid.append(np.median(these_de))

epoch_avg_jdtdb    = np.array(epoch_avg_jdtdb)
epoch_med_jdtdb    = np.array(epoch_med_jdtdb)
epoch_avg_de_resid = np.array(epoch_avg_de_resid)
epoch_med_de_resid = np.array(epoch_med_de_resid)
epoch_avg_ra_resid = np.array(epoch_avg_ra_resid)
epoch_med_ra_resid = np.array(epoch_med_ra_resid)
epoch_avg_jd_yrs   = (epoch_avg_jdtdb - epoch_avg_jdtdb[0]) / 365.25
epoch_med_jd_yrs   = (epoch_med_jdtdb - epoch_med_jdtdb[0]) / 365.25


#sys.exit(0)

## Gimme a 'light curve' I can analyze:
lcfile = 'pos_curve_j0805.txt'
with open(lcfile, 'w') as lf:
    for stuff in zip(j0805['jdtdb'], resid_ra_all, resid_de_all):
        lf.write("%.7f %12.7f %12.7f\n" % stuff)


##--------------------------------------------------------------------------##
## Parameter specs for MCMC:
ref_tdb = afn.ref_tdb
twopi = 2.0 * np.pi

## params = [ra0, de0, mur, mud, plx] + [per, esino, ecoso, inc, asc, lam] + [sma, ada]

#niter = 4
def make_orbit(orbpars, niter=4):
    """Generate dimensionless orbit at the times corresponding
    to data points used in the afn fitter. This guarantees the same
    reference epoch for the model as for the fitter.

    This routine requires an array of model parameters as input:
    * orbital period in years
    * sqrt(e)*sin(omega)
    * sqrt(e)*cos(omega)
    * orbit inclination (radians)
    * (asc) longitude of the ascending node in RADIANS
    * (lam) phase offset between periastron and ref epoch 
    * (aphot) size of photocenter orbit semi-major axis in MILLIARCSEC

    Return values:
    * delta_ra -- photocenter displacement in RA (RADIANS), delta_RA*cos(dec)
    * delta_de -- photocenter displacement in DE (RADIANS)
    """
    per, esino, ecoso, inc, asc, lam, aphot_mas = orbpars
    aphot_deg = aphot_mas / 3.6e6    # now in degrees
    aphot_rad = np.radians(aphot_deg)
    ecc = esino**2 + ecoso**2
    arg = np.arctan2(esino, ecoso)  # omega
    eccsqr = ecc**2
    # we can use afn._dt_yrs in place of (ep - refep)
    M_anom = twopi * afn._dt_yrs / per + lam - arg
    E_anom = M_anom + ecc * np.sin(M_anom) + 0.5 * eccsqr * np.sin(2.0*M_anom)
    # iteratively solve for eccentric anomaly:
    for i in range(niter):
        E_anom += (M_anom - E_anom + ecc * np.sin(E_anom)) / (1.0 - ecc * np.cos(E_anom))

    # calculate Thiele-Innes orbital elements:
    sinarg = np.sin(arg)
    cosarg = np.cos(arg)
    sinasc = np.sin(asc)
    cosasc = np.cos(asc)
    cosinc = np.cos(inc)
    A =  cosarg*cosasc - sinarg*sinasc*cosinc
    B =  cosarg*sinasc + sinarg*cosasc*cosinc
    F = -sinarg*cosasc - cosarg*sinasc*cosinc
    G = -sinarg*sinasc + cosarg*cosasc*cosinc
    X1 = np.cos(E_anom) - ecc
    Y1 = np.sin(E_anom) * np.sqrt(1.0 - eccsqr)
    # calculate Cartesian orbit positions (dimensionless)
    dr1 = B*X1 + G*Y1
    #dr2 = B*X2 + G*Y2
    dd1 = A*X1 + F*Y1
    #dd2 = A*X2 + F*Y2
    # compute orbit separation (dimensionless) and position angle
    #se1 = sma*np.sqrt(dr1**2 + dd1**2)
    #pa1 = np.arctan2(dr1,dd1)
    # compute integrated-light Cartesian coordinates
    #ra2 = ra0 + (dtyr*mur + pir*plx + dr2*aph)*inv_cosd
    #de2 = de0 +  dtyr*mud + pid*plx + dd2*aph
    return aphot_rad*dr1, aphot_rad*dd1

# per, esino, ecoso, inc, asc, lam = orbpars

## Test parameters:
sahl_per_day    = 740.419
sahl_per_yrs    = sahl_per_day / 365.25
sahl_inc_deg    = 111.795
sahl_ecc        =   0.422
sahl_omega_deg  = -55.857    # degrees
sahl_lonasc_deg = -13.705    # degrees
sahl_esino      = np.sqrt(sahl_ecc) * np.sin(np.radians(sahl_omega_deg))
sahl_ecoso      = np.sqrt(sahl_ecc) * np.cos(np.radians(sahl_omega_deg))
#dummy_lambda    = 0.3
dummy_lambda    = -0.45
dummy_aphot     = 3.6e6
dummy_aphot     = 15.0

test_orb_pars = np.array([sahl_per_yrs, sahl_esino, sahl_ecoso,
    np.radians(sahl_inc_deg), np.radians(sahl_lonasc_deg), 
    dummy_lambda, dummy_aphot])

test_dra, test_dde = make_orbit(test_orb_pars)

full_initial_guess = np.concatenate((afn.full_result[0], test_orb_pars))

## Illustrate parameter parsing:
def parse_astorb_params(parameters):
    astpars = parameters[:5]
    orbpars = parameters[5:]
    return astpars, orbpars


def lnprior(params):
    ra0, de0, pmra, pmde, prlx, per, esino, ecoso, inc, asc, lam, aphot = params
    ecc = esino**2 + ecoso**2
    if per <= 0.0:
        return -np.inf
    if ecc < 0.0 or ecc >= 1.0:
        return -np.inf
    if esino <= -1.0 or esino >= 1.0:
        return -np.inf
    if ecoso <= -1.0 or ecoso >= 1.0:
        return -np.inf
    if inc <= 0.0 or inc >= np.pi:
        return -np.inf
    if asc < -np.pi or asc >= 3.0*np.pi:
        return -np.inf
    if lam < -np.pi or lam >= 3.0*np.pi:
        return -np.inf
    if ra0 < 0.0 or ra0 >= 360.0:
        return -np.inf
    if de0 < -90.0 or de0 > 90.0:
        return -np.inf
    if prlx <= 0.0:
        return -np.inf
    return np.log( np.sin(inc) * 1.0/per * (1.0/prlx)**4 )

## Evaluator for MCMC:
def lnlike(params):
    # The first 
    astpars, orbpars = parse_astorb_params(params)
    # model RA/DE in radians using astrom_test_2:
    model_rra, model_rde = afn.eval_model(astpars)
    # model orbit delta RA/DE in radians:
    orbit_rra, orbit_rde = make_orbit(orbpars)
    orbit_rra /= np.cos(model_rde)

    total_rra = model_rra + orbit_rra
    total_rde = model_rde + orbit_rde

    diffs = np.concatenate((fitme_ra_rad - total_rra[data_which], 
                            fitme_de_rad - total_rde[data_which]))
    
    return -0.5 * np.sum(diffs**2 * inv_sigma2 - np.log(inv_sigma2))

def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params)

## Make dataset:
data_ra_deg = afn.dataset['dtr_dra']
data_de_deg = afn.dataset['dtr_dde']
data_which  = afn.dataset['inliers']
fitme_ra_rad = np.radians(data_ra_deg)[data_which]
fitme_de_rad = np.radians(data_de_deg)[data_which]
fitme_rade_vals = np.concatenate((fitme_ra_rad, fitme_de_rad))

dummy_err_mas = 20.0
dummy_err_rad = np.radians(dummy_err_mas / 3.6e6)
fitme_ra_err_rad = np.ones_like(fitme_ra_rad) * dummy_err_rad
fitme_de_err_rad = np.ones_like(fitme_de_rad) * dummy_err_rad

fitme_rade_errs = np.concatenate((fitme_ra_err_rad, fitme_de_err_rad))

inv_sigma2 = 1.0 / fitme_rade_errs**2


_PERFORM_MCMC = False
#_PERFORM_MCMC = True

#niter, thinned =  2000,  5
#niter, thinned =  4000, 15
niter, thinned = 40000, 15
save_chain = True
chain_file = 'chain.csv'
corner_png = 'corner_%06d_%03d.png' % (niter, thinned)

if _PERFORM_MCMC:
    tik = time.time()
    initial = full_initial_guess.copy()
    ndim = len(initial)
    nwalkers = 32
    p0 = [np.array(initial) + 1e-6*initial*np.random.randn(ndim) \
            for i in range(nwalkers)]
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=arglist)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #, args=arglist)

    sys.stderr.write("Running burn-in ... ")
    p0, _, _ = sampler.run_mcmc(p0, 200)
    sys.stderr.write("done.\n")
    sampler.reset()

    sys.stderr.write("Running full MCMC ... ")
    pos, prob, state = sampler.run_mcmc(p0, niter)
    sys.stderr.write("done.\n")

    ra_chain, de_chain, pmra_chain, pmde_chain, prlx_chain, \
            per_chain, esino_chain, ecoso_chain, inc_chain, \
            asc_chain, lam_chain, aphot_chain = sampler.flatchain.T

    plabels = ['ra', 'de', 'pmra', 'pmde', 'prlx',
            'per', 'esino', 'ecoso', 'inc', 'asc', 'lam', 'aphot']
    flat_samples = sampler.get_chain(discard=100, thin=thinned, flat=True)

    # save the chain to external CSV if requested:
    if save_chain and chain_file:
        sys.stderr.write("Saving chain to '%s' ... " % chain_file)
        with open(chain_file, 'w') as cf:
            # make header:
            cf.write(','.join(plabels) + '\n')
            # add data:
            for line in sampler.flatchain:
                cf.write(','.join([str(x) for x in line]) + '\n')
                pass
            pass
        pass
        sys.stderr.write("done.\n")

    tok = time.time()
    sys.stderr.write("Running MCMC took %.2f seconds.\n" % (tok-tik))

    cornerfig = plt.figure(31, figsize=(9,7))
    cornerfig.clf()
    corner.corner(flat_samples, labels=plabels, fig=cornerfig)
    cornerfig.savefig(corner_png, bbox_inches='tight')
    #corner.corner(flat_samples, labels=plabels, fig=cornerfig,
    #                    truths=iterpars)


    pass
# per, esino, ecoso, inc, asc, lam = orbpars


##--------------------------------------------------------------------------##
## Misc:
#def log_10_product(x, pos):
#   """The two args are the value and tick position.
#   Label ticks with the product of the exponentiation."""
#   return '%.2f' % (x)  # floating-point
#
#formatter = plt.FuncFormatter(log_10_product) # wrap function for use

## Convenient, percentile-based plot limits:
#def nice_limits(vec, pctiles=[1,99], pad=1.2):
#    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
#    middle = np.average(ends)
#    return (middle + pad * (ends - middle))

## Convenient plot limits for datetime/astropy.Time content:
#def nice_time_limits(tvec, buffer=0.05):
#    lower = tvec.min()
#    upper = tvec.max()
#    ndays = upper - lower
#    return ((lower - 0.05*ndays).datetime, (upper + 0.05*ndays).datetime)

## Convenient limits for datetime objects:
#def dt_limits(vec, pad=0.1):
#    tstart, tstop = vec.min(), vec.max()
#    trange = (tstop - tstart).total_seconds()
#    tpad = dt.timedelta(seconds=pad*trange)
#    return (tstart - tpad, tstop + tpad)

##--------------------------------------------------------------------------##
## Solve prep:
#ny, nx = img_vals.shape
#x_list = (0.5 + np.arange(nx)) / nx - 0.5            # relative (centered)
#y_list = (0.5 + np.arange(ny)) / ny - 0.5            # relative (centered)
#xx, yy = np.meshgrid(x_list, y_list)                 # relative (centered)
#xx, yy = np.meshgrid(nx*x_list, ny*y_list)           # absolute (centered)
#xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))   # absolute
#yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij') # absolute
#yy, xx = np.nonzero(np.ones_like(img_vals))          # absolute
#yy, xx = np.mgrid[0:ny,   0:nx].astype('uint16')     # absolute (array)
#yy, xx = np.mgrid[1:ny+1, 1:nx+1].astype('uint16')   # absolute (pixel)

## 1-D vectors:
#x_pix, y_pix, ivals = xx.flatten(), yy.flatten(), img_vals.flatten()
#w_vec = np.ones_like(ivals)            # start with uniform weights
#design_matrix = np.column_stack((np.ones(x_pix.size), x_pix, y_pix))

## Image fitting (statsmodels etc.):
#data = sm.datasets.stackloss.load()
#ols_res = sm.OLS(ivals, design_matrix).fit()
#rlm_res = sm.RLM(ivals, design_matrix).fit()
#rlm_model = sm.RLM(ivals, design_matrix, M=sm.robust.norms.HuberT())
#rlm_res = rlm_model.fit()
#data = pd.DataFrame({'xpix':x_pix, 'ypix':y_pix})
#rlm_model = sm.RLM.from_formula("ivals ~ xpix + ypix", data)

##--------------------------------------------------------------------------##
## Theil-Sen line-fitting (linear):
#model = ts.linefit(xvals, yvals)
#icept, slope = ts.linefit(xvals, yvals)

## Theil-Sen line-fitting (loglog):
#xvals, yvals = np.log10(original_xvals), np.log10(original_yvals)
#xvals, yvals = np.log10(df['x'].values), np.log10(df['y'].values)
#llmodel = ts.linefit(np.log10(xvals), np.log10(yvals))
#icept, slope = ts.linefit(xvals, yvals)
#fit_exponent = slope
#fit_multiplier = 10**icept
#bestfit_x = np.arange(5000)
#bestfit_y = fit_multiplier * bestfit_x**fit_exponent

## Log-log evaluator:
#def loglog_eval(xvals, model):
#    icept, slope = model
#    return 10**icept * xvals**slope
#def loglog_eval(xvals, icept, slope):
#    return 10**icept * xvals**slope


##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
fig.clf()
forb = plt.figure(2, figsize=fig_dims)
forb.clf()
fax1 = forb.add_subplot(221, aspect='equal')
fax2 = forb.add_subplot(222, aspect='equal')
fax3 = forb.add_subplot(223, aspect='equal')
skw = {'lw':0, 's':15}
fax1.scatter(resid_ra_all, resid_de_all, c=dt_years, **skw)
fax1.set_xlim(-200, 200)
fax1.set_ylim(-200, 200)
fax2.scatter(test_dra, test_dde, **skw)
fax2.grid(True)
#fax3.scatter(epoch_avg_ra_resid, epoch_avg_de_resid, c=epoch_avg_jd_yrs, **skw)
fax3.scatter(epoch_med_ra_resid, epoch_med_de_resid, c=epoch_med_jd_yrs, **skw)
#orblims = (-1.5, 1.5)
orblims = np.array([-1.5, 1.5])
orblims *= 3.6e6* 180
#fax2.set_xlim(*orblims); fax2.set_ylim(*orblims)
forb.tight_layout()

#plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222, sharex=ax1)
ax3 = fig.add_subplot(223, sharex=ax1)
ax4 = fig.add_subplot(224, sharex=ax1)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
axlist = [ax1, ax2, ax3, ax4]
for ax in axlist:
    ax.grid(True)
#ax1.axis('off')
ax1.set_title('RA - proper_motion')
ax2.set_title('DE - proper_motion')


skw = {'lw':0, 's':15}
#ax1.scatter(j0805.jdtdb, resid_ra, **skw)
#ax2.scatter(j0805.jdtdb, resid_de, **skw)
ax1.scatter(dt_years, resid_ra, **skw)
ax2.scatter(dt_years, resid_de, **skw)
ax3.scatter(dt_years, resid_ra_all, **skw)
ax4.scatter(dt_years, resid_de_all, **skw)
ax3.set_xlabel("Years since %.1f" % afn.ref_tdb)
ax4.set_xlabel("Years since %.1f" % afn.ref_tdb)

for ax in axlist:
    ax.set_ylim(-150, 150)
#ax1.set_ylim(-200, 200)
#ax2.set_ylim(-200, 200)

## Polar scatter:
#skw = {'lw':0, 's':15}
#ax1.scatter(azm_rad, zdist_deg, **skw)

## For polar axes:
#ax1.set_rmin( 0.0)                  # if using altitude in degrees
#ax1.set_rmax(90.0)                  # if using altitude in degrees
#ax1.set_theta_direction(-1)         # counterclockwise
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

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')



######################################################################
# CHANGELOG (34_fit_SDSS_J0805.py):
#---------------------------------------------------------------------
#
#  2024-04-29:
#     -- Increased __version__ to 0.0.1.
#     -- First created 34_fit_SDSS_J0805.py.
#

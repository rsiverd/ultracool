#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Measure relative detector rotations across the calib1 data set using
# sexterp alignment info.
#
# Rob Siverd
# Created:       2025-04-15
# Last modified: 2025-04-15
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
np.set_printoptions(suppress=True, linewidth=160)
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

unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
if (resource.getrlimit(resource.RLIMIT_DATA) == unlimited):
    resource.setrlimit(resource.RLIMIT_DATA,  (3e9, 6e9))
if (resource.getrlimit(resource.RLIMIT_AS) == unlimited):
    resource.setrlimit(resource.RLIMIT_AS, (3e9, 6e9))

## Memory management:
#def get_memory():
#    with open('/proc/meminfo', 'r') as mem:
#        free_memory = 0
#        for i in mem:
#            sline = i.split()
#            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
#                free_memory += int(sline[1])
#    return free_memory
#
#def memory_limit():
#    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

### Measure memory used so far:
#def check_mem_usage_MB():
#    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#    return max_kb_used / 1000.0

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
## Quick ASCII I/O:
data_file = 'shifts.csv'
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
all_data = pd.read_csv(data_file, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)
#nskip, cnames = analyze_header(data_file)
#all_data = pd.read_csv(data_file, names=cnames, skiprows=nskip, **pdkwargs)
#all_data = pd.DataFrame.from_records(npy_data)
#all_data = pd.DataFrame(all_data.byteswap().newbyteorder()) # for FITS tables

## Remove whitespace from FILTER:
all_data['FILTER'] = all_data.FILTER.str.strip()

## Extract polynomial order and quadrant:
qmap = {'NE':1, 'NW':2, 'SE':3, 'SW':4}
all_data[ 'base'] = [os.path.basename(x) for x in all_data.FILENAME]
all_data[ 'inum'] = [int(x.split('_')[-1].split('p')[0]) for x in all_data.base]
#all_data[ 'poly'] = [int(x.split('_')[1][1]) for x in all_data.FILENAME]
all_data[ 'poly'] = [int(x.split('/')[0][-1]) for x in all_data.FILENAME]
all_data[ 'quad'] = [x.split('/')[1].split('_')[1] for x in all_data.FILENAME]
all_data[ 'qnum'] = [qmap[x] for x in all_data.quad]
all_data['runid'] = [x.split('/')[1].split('_')[0] for x in all_data.FILENAME]

##--------------------------------------------------------------------------##
## Dec compensation (see DEC_COMPENSATION.txt):
mctr_ra = 294.591259
mctr_de =  35.118443
nudge_arcmin = 5.25
nudge_degree = nudge_arcmin / 60.0
upper_ctr_de = mctr_de + nudge_degree
lower_ctr_de = mctr_de - nudge_degree
tbore_cosdec = np.cos(np.radians(mctr_de))         # 0.8179645859842891
upper_cosdec = np.cos(np.radians(upper_ctr_de))    # 0.8170851035450546
lower_cosdec = np.cos(np.radians(lower_ctr_de))    # 0.8188421607447035
upper_cosrat = upper_cosdec / tbore_cosdec         # 0.9989247915443965
lower_cosrat = lower_cosdec / tbore_cosdec         # 1.0010728762289365

tmp_x_offset = all_data.OFFSET_X.copy()
tmp_x_offset[all_data.quad == 'NE'] /= upper_cosrat
tmp_x_offset[all_data.quad == 'NW'] /= upper_cosrat
tmp_x_offset[all_data.quad == 'SE'] /= lower_cosrat
tmp_x_offset[all_data.quad == 'SW'] /= lower_cosrat
all_data['OFFSET_X'] = tmp_x_offset
#all_data[all_data.quad == 'NE']['OFFSET_X'] *=


#sys.exit(0)

#n17p1 = len(all_data[(all_data.runid == '17BQ03') & (all_data.poly == 1)].inum.unique())
#n17p2 = len(all_data[(all_data.runid == '17BQ03') & (all_data.poly == 2)].inum.unique())
#sys.stderr.write("n17p1: %d\n" % n17p1)
#sys.stderr.write("n17p2: %d\n" % n17p2)

## Strip out the QSOGRADE > 3 cases before proceeding:
#badgrade = 4
badgrade = 3
raw_data = all_data
all_data = all_data[all_data.QSOGRADE < badgrade]

## Wait ... does QSOGRADE not match?
nchunks = raw_data.groupby('inum')
for nn,subset in nchunks:
    grades = np.unique(subset.QSOGRADE)
    if len(grades) != 1:
        sys.stderr.write("TROUBLE!\n") 
        sys.stderr.write("grades: %s\n" % str(grades))
        #sys.stderr.write("%s\n" % str(subset))
        sys.exit(1)
        #import pdb; pdb.set_trace()
    pass


## Count images per runid, drop sparse RUNIDs:
reqimgs = 10
rchunks = raw_data.groupby('runid')
thin_runids = []
for rr,subset in rchunks:
    refs = subset.INT_REF.unique()
    nonref = ~subset.base.str[7:].isin(refs)
    decent = subset.QSOGRADE < 3
    subkeep = subset[nonref & decent]
    #nimg = len(subkeep[subkeep.poly
    nimg = len(subkeep.inum.unique())
    #sys.stderr.write("QRUNID: %s -- %d\n" % (rr, nimg))
    if nimg < reqimgs:
        sys.stderr.write("Dropping QRUNID %s (%d images) ... \n" % (rr, nimg))
        thin_runids.append(rr)
    pass

thin_runids.append('15BQ09')
drop = all_data.runid.isin(thin_runids)
all_data = all_data[~drop]

#sys.exit(0)

##--------------------------------------------------------------------------##
## Make sure the data are aligned before proceeding ...
_poly1 = all_data.poly == 1
_poly2 = all_data.poly == 2
#_quad1 = all_data.qnum == 1
p1data = all_data[all_data.poly == 1]
picky1 = {x:p1data['qnum']==x for x in range(1,5)}
p1qsel = {x:p1data[w] for x,w in picky1.items()}
p2data = all_data[all_data.poly == 2]
picky2 = {x:p2data['qnum']==x for x in range(1,5)}
p2qsel = {x:p2data[w] for x,w in picky2.items()}

## Compare q1 vs q2:
for other in (2, 3, 4):
    for dat in (p1qsel, p2qsel):
        if not np.all(p1qsel[1]['inum'].values == p1qsel[other]['inum'].values):
            sys.stderr.write("We have a problem ...\n")
            sys.stderr.write("Mismatch with other=%d\n" % other)
            sys.exit(1)
del p1data, picky1, p1qsel, p2data, picky2, p2qsel
sys.stderr.write("gc.collect() ... %d\n" % gc.collect())
sys.stderr.write("gc.collect() ... %d\n" % gc.collect())

## Drop everything with QSOGRADE > 3:
#junk = all_data.QSOGRADE > 3

## Look for grossly discrepant alignments ...
_showbad = False
#_showbad = True
distcols = ['OFFSET_X', 'OFFSET_Y', 'OFFSET_R']
distdevs = []
ichunks = all_data.groupby('inum')
devthresh = 2.0
bad_inums = []
for nn,subset in ichunks:
    devs = np.array([subset[x].std() for x in distcols])
    distdevs.append(devs)
    #devtxt = '%.4f %.4f %.4f' % (*devs,)
    #sys.stderr.write("nn: %d (%d) -- %s\n" % (nn, len(subset), devtxt))
    if np.any(devs > devthresh):
        bad_inums.append(subset.inum.iloc[0])
        sys.stderr.write("nn: %d (%d) -- %.4f %.4f %.4f\n" % (nn, len(subset), *devs))
        if _showbad:
            sys.stderr.write("%s\n" % str(subset))
            sys.stderr.write("\n")
            ds9cmd1 = "fztfs " + ' '.join(subset.FILENAME)
            sys.stderr.write("%s\n" % ds9cmd1)
            itpdirs = [os.path.dirname(x) for x in subset.FILENAME]
            for qq in qmap.keys():
                wildcards = [x+'/int*fits' for x in itpdirs if x.endswith(qq)]
                ds9cmd2 = "fztfs " + ' '.join(wildcards)
                sys.stderr.write("%s\n" % ds9cmd2)
                wildcards = ['unpacked/'+x.split('/')[1]+'/wirc*fits' for x in itpdirs if x.endswith(qq)]
                ds9cmd3 = "fztfs " + ' '.join(wildcards)
                sys.stderr.write("%s\n" % ds9cmd3)
            #ds9cmdu = "fztfs " + 
            sys.stderr.write("\n")
    #break
#med_devs = np.median(distdevs, axis=0)
#med_devs, iqr_devs = rs.calc_ls_med_IQR(distdevs, axis=0)
# adopt cutoff =~ 2

## Bark in case of mismatched poly=1 and poly=2:


## Drop reference frames (should be 0-offset) before proceeding:
refs = all_data.INT_REF.unique()
keep = ~all_data.base.str[7:].isin(refs)
all_data = all_data[keep]

## Drop specific inums that appear to be terrible:
keep = ~all_data.inum.isin(bad_inums)
all_data = all_data[keep]

#sys.exit(0)

##--------------------------------------------------------------------------##
## Filter choice ('J', 'H2', 'both'):
_WANTFILT = 'both'
_WANTFILT = 'J'
#_WANTFILT = 'H2'

## Select the non-zero shifts:
if _WANTFILT == 'both':
    keepers = (all_data.OFFSET_R > -0.01)
else:
    keepers = (all_data.OFFSET_R > -0.01) & (all_data.FILTER == _WANTFILT)

sdata = all_data[keepers]
#sys.exit(0)

## Make list of all RUNIDs:
every_runid = np.unique(all_data.runid)

# As of here ...
#n17p1 = len(all_data[(all_data.runid == '17BQ03') & (all_data.poly == 1)].inum.unique())
#n17p2 = len(all_data[(all_data.runid == '17BQ03') & (all_data.poly == 2)].inum.unique())
#sys.stderr.write("n17p1: %d\n" % n17p1)
#sys.stderr.write("n17p2: %d\n" % n17p2)
#sys.exit(0)


##--------------------------------------------------------------------------##
## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array((c, -s, s, c)).reshape(2, 2)

##--------------------------------------------------------------------------##
## Group by images, organize by quadrant:
shifts = {}
pchunks = sdata.groupby('poly')
for pp,psubset in pchunks:
    shifts[pp] = {}
    qchunks  = psubset.groupby('quad')
    #qresults = {}
    for qq,qsubset in qchunks:
        stuff = qsubset.sort_values(by='inum')
        #dx = stuff.OFFSET_X
        #dy = stuff.OFFSET_Y
        shifts[pp][qq] = {'dx':stuff.OFFSET_X.values,
                          'dy':stuff.OFFSET_Y.values,
                          'rr':stuff.runid.values,
                          'ii':stuff.inum.values}
        pass
    #shifts[pp] = qresults

### Shift each RUNID dx,dy values to zero-mean:
#for x in every_runid:
#    which = (shifts[1]['NE']['rr'] == x)        # elements from this runid
#    if np.sum(which) < 1:
#        continue

## Tally data points and X,Y spread in each RUNID:
#rtally = {x:np.sum(shifts[1]['NE']['rr'] == x) for x in every_runid}
rtally = {}
rxypos = {}
mypoly = 1
for x in every_runid:
    which = (shifts[mypoly]['NE']['rr'] == x)
    rtally[x] = np.sum(which)
    rxypos[x] = (shifts[mypoly]['NE']['dx'][which], shifts[mypoly]['NE']['dy'][which])

#sys.exit(0)
##--------------------------------------------------------------------------##
## Fit NE-to-SE rotation:
#fshifts_NE
#shifts[1]['NE']['dx']
#shifts[2]['NE']['dx']
#shifts[1]['NE']['dy']
#shifts[2]['NE']['dy']
_CHOICE = 'both'
#_CHOICE = 1
_CHOICE = 2

if _CHOICE == 'both':
    shifts_NE_dx = np.concatenate((shifts[1]['NE']['dx'], shifts[2]['NE']['dx']))
    shifts_NE_dy = np.concatenate((shifts[1]['NE']['dy'], shifts[2]['NE']['dy']))
    shifts_NE_rr = np.concatenate((shifts[1]['NE']['rr'], shifts[2]['NE']['rr']))
    shifts_NW_dx = np.concatenate((shifts[1]['NW']['dx'], shifts[2]['NW']['dx']))
    shifts_NW_dy = np.concatenate((shifts[1]['NW']['dy'], shifts[2]['NW']['dy']))
    shifts_NW_rr = np.concatenate((shifts[1]['NW']['rr'], shifts[2]['NW']['rr']))
    shifts_SE_dx = np.concatenate((shifts[1]['SE']['dx'], shifts[2]['SE']['dx']))
    shifts_SE_dy = np.concatenate((shifts[1]['SE']['dy'], shifts[2]['SE']['dy']))
    shifts_SE_rr = np.concatenate((shifts[1]['SE']['rr'], shifts[2]['SE']['rr']))
    shifts_SW_dx = np.concatenate((shifts[1]['SW']['dx'], shifts[2]['SW']['dx']))
    shifts_SW_dy = np.concatenate((shifts[1]['SW']['dy'], shifts[2]['SW']['dy']))
    shifts_SW_rr = np.concatenate((shifts[1]['SW']['rr'], shifts[2]['SW']['rr']))
else:
    shifts_NE_dx = shifts[_CHOICE]['NE']['dx']
    shifts_NE_dy = shifts[_CHOICE]['NE']['dy']
    shifts_NE_rr = shifts[_CHOICE]['NE']['rr']
    shifts_NW_dx = shifts[_CHOICE]['NW']['dx']
    shifts_NW_dy = shifts[_CHOICE]['NW']['dy']
    shifts_NW_rr = shifts[_CHOICE]['NW']['rr']
    shifts_SE_dx = shifts[_CHOICE]['SE']['dx']
    shifts_SE_dy = shifts[_CHOICE]['SE']['dy']
    shifts_SE_rr = shifts[_CHOICE]['SE']['rr']
    shifts_SW_dx = shifts[_CHOICE]['SW']['dx']
    shifts_SW_dy = shifts[_CHOICE]['SW']['dy']
    shifts_SW_rr = shifts[_CHOICE]['SW']['rr']

shifts_NE = np.column_stack((shifts_NE_dx, shifts_NE_dy))
shifts_NW = np.column_stack((shifts_NW_dx, shifts_NW_dy))
shifts_SE = np.column_stack((shifts_SE_dx, shifts_SE_dy))
shifts_SW = np.column_stack((shifts_SW_dx, shifts_SW_dy))

def roteval(params, dxy1, dxy2):
    #padeg = params[0]
    #nudge = params[1:]
    #padeg, x0, y0 = params
    #rmat = rotation_matrix(np.radians(padeg))
    rmat = params[:4].reshape(2,2)
    nudge = params[4:]
    #resid = np.matmul(dxy1, rmat) - dxy2 - nudge
    #resid = np.matmul(dxy1, rmat) - dxy2
    resid = np.matmul(rmat, dxy1.T) - dxy2.T - nudge[:, None]
    return resid.ravel()
    #return np.sum(resid**2)

minimize_NE_NW = partial(roteval, dxy1=shifts_NE, dxy2=shifts_NW)
minimize_NE_SE = partial(roteval, dxy1=shifts_NE, dxy2=shifts_SE)
minimize_NE_SW = partial(roteval, dxy1=shifts_NE, dxy2=shifts_SW)
minimize_NW_SE = partial(roteval, dxy1=shifts_NW, dxy2=shifts_SE)
minimize_NW_SW = partial(roteval, dxy1=shifts_NW, dxy2=shifts_SW)
minimize_SE_SW = partial(roteval, dxy1=shifts_SE, dxy2=shifts_SW)
#guess = np.array([0.0, 0.0, 0.0])
#guess = np.array([0.0])
guess = np.zeros(4)
guess = np.zeros(6)

sys.stderr.write("Fitting rotations (whole thing) ... ")
rfit_NE_NW = opti.least_squares(minimize_NE_NW, guess.copy())
rfit_NE_SE = opti.least_squares(minimize_NE_SE, guess.copy())
rfit_NE_SW = opti.least_squares(minimize_NE_SW, guess.copy())
rfit_NW_SE = opti.least_squares(minimize_NW_SE, guess.copy())
rfit_NW_SW = opti.least_squares(minimize_NW_SW, guess.copy())
rfit_SE_SW = opti.least_squares(minimize_SE_SW, guess.copy())
sys.stderr.write("done.\n")

good_guess = {
    'NE_NW'  : np.array([0.99998009,  0.00420331,  0.0063609 , 0.99980749, -0.00622012, 0.00163287]),
    'NE_SE'  : np.array([1.0022247 ,  0.00273342,  0.00557709, 0.99983466,  0.00160602, 0.0075152 ]),
    'NE_SW'  : np.array([1.0027265 , -0.00014886,  0.00249428, 1.00043846, -0.00482773, 0.00914531]),
    'NW_SE'  : np.array([1.0022702 , -0.00144964, -0.00077761, 1.00001337,  0.00778413, 0.00594833]),
    'NW_SW'  : np.array([1.00276108, -0.00435576, -0.0038686 , 1.00063832,  0.00131232, 0.00724353]),
    'SE_SW'  : np.array([1.00047543, -0.00289404, -0.00308505, 1.00061152, -0.00642231, 0.00158428]),
}

def rfit_many(_CHOICE, runid=None):
    if _CHOICE == 'both':
        shifts_NE_dx = np.concatenate((shifts[1]['NE']['dx'], shifts[2]['NE']['dx']))
        shifts_NE_dy = np.concatenate((shifts[1]['NE']['dy'], shifts[2]['NE']['dy']))
        shifts_NE_rr = np.concatenate((shifts[1]['NE']['rr'], shifts[2]['NE']['rr']))
        shifts_NW_dx = np.concatenate((shifts[1]['NW']['dx'], shifts[2]['NW']['dx']))
        shifts_NW_dy = np.concatenate((shifts[1]['NW']['dy'], shifts[2]['NW']['dy']))
        shifts_NW_rr = np.concatenate((shifts[1]['NW']['rr'], shifts[2]['NW']['rr']))
        shifts_SE_dx = np.concatenate((shifts[1]['SE']['dx'], shifts[2]['SE']['dx']))
        shifts_SE_dy = np.concatenate((shifts[1]['SE']['dy'], shifts[2]['SE']['dy']))
        shifts_SE_rr = np.concatenate((shifts[1]['SE']['rr'], shifts[2]['SE']['rr']))
        shifts_SW_dx = np.concatenate((shifts[1]['SW']['dx'], shifts[2]['SW']['dx']))
        shifts_SW_dy = np.concatenate((shifts[1]['SW']['dy'], shifts[2]['SW']['dy']))
        shifts_SW_rr = np.concatenate((shifts[1]['SW']['rr'], shifts[2]['SW']['rr']))
    else:
        shifts_NE_dx = shifts[_CHOICE]['NE']['dx']
        shifts_NE_dy = shifts[_CHOICE]['NE']['dy']
        shifts_NE_rr = shifts[_CHOICE]['NE']['rr']
        shifts_NW_dx = shifts[_CHOICE]['NW']['dx']
        shifts_NW_dy = shifts[_CHOICE]['NW']['dy']
        shifts_NW_rr = shifts[_CHOICE]['NW']['rr']
        shifts_SE_dx = shifts[_CHOICE]['SE']['dx']
        shifts_SE_dy = shifts[_CHOICE]['SE']['dy']
        shifts_SE_rr = shifts[_CHOICE]['SE']['rr']
        shifts_SW_dx = shifts[_CHOICE]['SW']['dx']
        shifts_SW_dy = shifts[_CHOICE]['SW']['dy']
        shifts_SW_rr = shifts[_CHOICE]['SW']['rr']
    if runid:
        shifts_NE = np.column_stack((shifts_NE_dx, shifts_NE_dy))[shifts_NE_rr == runid]
        shifts_NW = np.column_stack((shifts_NW_dx, shifts_NW_dy))[shifts_NW_rr == runid]
        shifts_SE = np.column_stack((shifts_SE_dx, shifts_SE_dy))[shifts_SE_rr == runid]
        shifts_SW = np.column_stack((shifts_SW_dx, shifts_SW_dy))[shifts_SW_rr == runid]
    else:
        shifts_NE = np.column_stack((shifts_NE_dx, shifts_NE_dy))
        shifts_NW = np.column_stack((shifts_NW_dx, shifts_NW_dy))
        shifts_SE = np.column_stack((shifts_SE_dx, shifts_SE_dy))
        shifts_SW = np.column_stack((shifts_SW_dx, shifts_SW_dy))
    minimize_NE_NW = partial(roteval, dxy1=shifts_NE, dxy2=shifts_NW)
    minimize_NE_SE = partial(roteval, dxy1=shifts_NE, dxy2=shifts_SE)
    minimize_NE_SW = partial(roteval, dxy1=shifts_NE, dxy2=shifts_SW)
    minimize_NW_SE = partial(roteval, dxy1=shifts_NW, dxy2=shifts_SE)
    minimize_NW_SW = partial(roteval, dxy1=shifts_NW, dxy2=shifts_SW)
    minimize_SE_SW = partial(roteval, dxy1=shifts_SE, dxy2=shifts_SW)
    guess = np.zeros(6)
    sys.stderr.write("Fitting rotations ... ")
    rfit_NE_NW = opti.least_squares(minimize_NE_NW, good_guess['NE_NW'])
    rfit_NE_SE = opti.least_squares(minimize_NE_SE, good_guess['NE_SE'])
    rfit_NE_SW = opti.least_squares(minimize_NE_SW, good_guess['NE_SW'])
    rfit_NW_SE = opti.least_squares(minimize_NW_SE, good_guess['NW_SE'])
    rfit_NW_SW = opti.least_squares(minimize_NW_SW, good_guess['NW_SW'])
    rfit_SE_SW = opti.least_squares(minimize_SE_SW, good_guess['SE_SW'])
    sys.stderr.write("done.\n")
    return rfit_NE_NW, rfit_NE_SE, rfit_NE_SW, \
            rfit_NW_SE, rfit_NW_SW, rfit_SE_SW

def analyze(data):
    cd_matrix = data[:4].reshape(2,2)
    cd_pscales = np.sqrt(np.sum(cd_matrix**2, axis=1))
    #cd_pscales = np.sqrt(np.sum(cd_matrix**2, axis=0))
    norm_cdmat = cd_matrix / cd_pscales
    #cd_ang_rad = np.arccos(norm_cdmat[0, 0])
    #sys.stderr.write("atan2: %s\n" % str(np.arctan2(norm_cdmat[1,:], norm_cdmat[0,:])))
    cd_ang_rad = np.arctan2(norm_cdmat[1,:], norm_cdmat[0,:])[0]
    cd_ang_deg = np.degrees(cd_ang_rad)
    return cd_ang_deg, cd_pscales[0], cd_pscales[1]

def fltfmt(data):
    #ang_deg = analyze(data)
    yetmore = analyze(data)
    #rot,xsc,ysc = yetmore
    things = ['%12.7f'%x for x in data]
    things += ['%12.7f'%x for x in yetmore]
    return ''.join(things)

columns = ['cd11', 'cd12', 'cd21', 'cd22', 'xnudge', 'ynudge', 'pa_deg', 'pscale0', 'pscale1']
sys.stderr.write("       " + ''.join(['%12s'%x for x in columns]) + "\n")
sys.stderr.write("NE-NW: %s\n" % fltfmt(rfit_NE_NW['x']))
sys.stderr.write("NE-SE: %s\n" % fltfmt(rfit_NE_SE['x']))
sys.stderr.write("NE-SW: %s\n" % fltfmt(rfit_NE_SW['x']))
sys.stderr.write("NW-SE: %s\n" % fltfmt(rfit_NW_SE['x']))
sys.stderr.write("NW-SW: %s\n" % fltfmt(rfit_NW_SW['x']))
sys.stderr.write("SE-SW: %s\n" % fltfmt(rfit_SE_SW['x']))

##--------------------------------------------------------------------------##

## A test-run with RUNID selected ...
test_runid = '21AQ13'
sys.stderr.write("%s\n" % fulldiv)
sys.stderr.write("rfit_many test run using %s ...\n" % test_runid)

rfit_NE_NW, rfit_NE_SE, rfit_NE_SW, rfit_NW_SE, rfit_NW_SW, rfit_SESW = \
        rfit_many(_CHOICE, test_runid)

columns = ['cd11', 'cd12', 'cd21', 'cd22', 'xnudge', 'ynudge', 'pa_deg', 'pscale0', 'pscale1']
sys.stderr.write("       " + ''.join(['%12s'%x for x in columns]) + "\n")
sys.stderr.write("NE-NW: %s\n" % fltfmt(rfit_NE_NW['x']))
sys.stderr.write("NE-SE: %s\n" % fltfmt(rfit_NE_SE['x']))
sys.stderr.write("NE-SW: %s\n" % fltfmt(rfit_NE_SW['x']))
sys.stderr.write("NW-SE: %s\n" % fltfmt(rfit_NW_SE['x']))
sys.stderr.write("NW-SW: %s\n" % fltfmt(rfit_NW_SW['x']))
sys.stderr.write("SE-SW: %s\n" % fltfmt(rfit_SE_SW['x']))

##--------------------------------------------------------------------------##
#sys.exit(0)

sys.stderr.write("\n%s\n%s\n" % (fulldiv, fulldiv))
sys.stderr.write("Separate solves for each RUNID ...\n")

## Run separately for each RUNID:
answers = {}
for runid in every_runid:
    sys.stderr.write("Relative rotations for %s ... " % runid)
    answers[runid] = rfit_many(_CHOICE, runid)

#ansfits = [answers[rr]['x'] for rr in every_runid]

results_NE_NW = np.array([answers[rr][0]['x'] for rr in every_runid])
results_NE_SE = np.array([answers[rr][1]['x'] for rr in every_runid])
results_NE_SW = np.array([answers[rr][2]['x'] for rr in every_runid])
results_NW_SE = np.array([answers[rr][3]['x'] for rr in every_runid])
results_NW_SW = np.array([answers[rr][4]['x'] for rr in every_runid])
results_SE_SW = np.array([answers[rr][5]['x'] for rr in every_runid])

meds_NE_NW, sigs_NE_NW = rs.calc_ls_med_MAD(results_NE_NW, axis=0)
meds_NE_SE, sigs_NE_SE = rs.calc_ls_med_MAD(results_NE_SE, axis=0)
meds_NE_SW, sigs_NE_SW = rs.calc_ls_med_MAD(results_NE_SW, axis=0)
meds_NW_SE, sigs_NW_SE = rs.calc_ls_med_MAD(results_NW_SE, axis=0)
meds_NW_SW, sigs_NW_SW = rs.calc_ls_med_MAD(results_NW_SW, axis=0)
meds_SE_SW, sigs_SE_SW = rs.calc_ls_med_MAD(results_SE_SW, axis=0)


### Stash old version as pickle:
#pklname = "old_results.pickle"
#stash_as_pickle(pklname, (results_NE_NW, results_NE_SE, results_NE_SW,
#                            results_NW_SE, results_NW_SW, results_SE_SW))
#results_NE_NW, results_NE_SE, results_NE_SW, \
#    results_NW_SE, results_NW_SW, results_SE_SW = load_pickled_object(pklname)
#
#derp = load_pickled_object(pklname)



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

sys.stderr.write("PLOTTING DISABLED (slow)\n")
sys.exit(0)

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (18, 9)
#fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
figax_pairs = []
for pairidx in range(6):
    figax_pairs.append(plt.subplots(3, 3, sharex=True, figsize=fig_dims, num=1+pairidx, clear=True))
#nenw_fig, nenw_axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
#nese_fig, nese_axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=2, clear=True)
#nesw_fig, nesw_axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=3, clear=True)
#nwse_fig, nwse_axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=4, clear=True)
#nwsw_fig, nwsw_axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=5, clear=True)
#sesw_fig, sesw_axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=6, clear=True)
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

## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)

sys.stderr.write("\n%s\n%s\n" % (fulldiv, fulldiv))
sys.stderr.write("Making breakout plots for each sensor pair ...\n")

#exclude_runids = ['18AQ05', '18BQ18', '16BQ11'] # '16BQ06'
#exclude_runids = ['15BQ04', '18AQ05', '18BQ18', '16BQ11'] # '16BQ06'
exclude_runids = []
display_runids = [x for x in every_runid if x not in exclude_runids]
display_ridpos = list(range(len(display_runids)))

pairnames = ['NE_NW', 'NE_SE', 'NE_SW', 'NW_SE', 'NW_SW', 'SE_SW']
col_names = ['cd11', 'cd12', 'cd21', 'cd22', 'xnudge', 'ynudge', 'PA', 'pscale0', 'pscale1']

for pairidx,name in enumerate(pairnames):
    sys.stderr.write("name: %s\n" % name)
    data = np.array([answers[rr][pairidx]['x'] for rr in display_runids])
    daux = np.array([analyze(x) for x in data])
    dall = np.column_stack((data, daux))
    fig, axs = figax_pairs[pairidx]
    this_name = pairnames[pairidx]
    fig.suptitle(this_name)
    for colidx in range(9):
        ax = axs.flatten()[colidx]
        ax.grid(True)
        ax.plot(dall[:, colidx])
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.set_ylabel(col_names[colidx])
    for ax in axs[-1]:
        ax.set_xlabel('RUNID')
        ax.set_xticks(display_ridpos)
        ax.set_xticklabels(display_runids, rotation=90, fontsize=10)
    fig.tight_layout()
    plot_name = 'breakout_%s.png' % this_name
    fig.savefig(plot_name, bbox_inches='tight')


##--------------------------------------------------------------------------##
## Separate figure for image tally:
fig_dims = (15, 8)
fig = plt.figure(99, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(121) ; ax1.grid(True)
ax2 = fig.add_subplot(122) ; ax2.grid(True)

xlabkw = {'rotation':90, 'fontsize':10}

every_runpos = list(range(len(every_runid)))
every_rcount = [rtally.get(x) for x in every_runid]
ax1.plot(every_runpos, every_rcount)
ax1.set_xticks(every_runpos)
ax1.set_xticklabels(every_runid, **xlabkw)
ax1.set_ylabel('Good Images')
ax1.set_title('All QRUNIDs')

display_rcount = [rtally.get(x) for x in display_runids]
ax2.plot(display_ridpos, display_rcount)
ax2.set_xticks(display_ridpos)
ax2.set_xticklabels(display_runids, **xlabkw)
ax2.set_title('"Clean" QRUNIDs shown on other plots')

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
plot_name = 'runid_imcount.png'
fig.savefig(plot_name, bbox_inches='tight')

##--------------------------------------------------------------------------##
## Another figure for dx,dy coverage by RUNID:
fig_dims = (14, 9.5)
#fig = plt.figure(98, figsize=fig_dims)
#fig.clf()
spkw = {'aspect':'equal'}
cfig, caxs = plt.subplots(6, 7, sharex=True, sharey=True, figsize=fig_dims, num=98, 
        clear=True, subplot_kw=spkw) #, adjustable='box', aspect='equal')

[ax.set_aspect('equal', adjustable='box') for ax in caxs.flatten()]

for ii,rr in enumerate(every_runid):
    ax = caxs.flatten()[ii]
    #ax.set_aspect('equal', adjustable='box')
    tx, ty = rxypos.get(rr)
    ax.scatter(tx, ty)
    ax.set_title(rr)

cfig.suptitle("$\Delta$X,$\Delta$Y offset (image - ref), QSOGRADE <= 2")

ax.set_xlim(-700, 700)
ax.set_ylim(-500, 500)
cfig.tight_layout()
plt.draw()
plot_name = 'runid_dithers.png'
cfig.savefig(plot_name, bbox_inches='tight')

## Polar scatter:
#skw = {'lw':0, 's':15}
#ax1.scatter(azm_rad, zdist_deg, **skw)

#mtickpos = [2,5,7]
#ndecades = 1.0   # for symlog, set width of linear portion in units of dex
#nonposx='mask' | nonposx='clip' | nonposy='mask' | nonposy='clip'
#ax1.set_xscale('log', basex=10, nonposx='mask', subsx=mtickpos)
#ax1.set_xscale('log', nonposx='clip', subsx=[3])
#ax1.set_yscale('symlog', basey=10, linthreshy=0.1, linscaley=ndecades)
#ax1.xaxis.set_major_formatter(fptformat) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10x0, 30.0, 100.0])
#plt.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#plt.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'], rotation=45)
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
# CHANGELOG (12_relative_rotations.py):
#---------------------------------------------------------------------
#
#  2025-04-15:
#     -- Increased __version__ to 0.0.1.
#     -- First created 12_relative_rotations.py.
#

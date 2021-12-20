#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Fetch Spitzer Heritage Archive data relevant to specified object.
#
# Rob Siverd
# Created:       2019-08-27
# Last modified: 2021-12-19
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.5.0"

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
import signal
import gc
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#from matplotlib import colors
#import matplotlib.colors as mplcolors
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import itertools as itt
from zipfile import ZipFile
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Because obviously:
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

##--------------------------------------------------------------------------##
## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers

## Parallax pipeline coordinate helpers:
try:
    import coord_helpers
    reload(coord_helpers)
except ImportError:
    logger.error("failed to import coord_helpers module!")
    sys.exit(1)
cfr = coord_helpers.CoordFileReader()

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
    from astroquery import sha
    from astropy import coordinates as coord
    from astropy import units as uu
except ImportError:
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

### SHA retrieval module:
#try:
#    import download_sha
#    reload(download_sha)
#    dsha = download_sha.DownloadSHA()
#except ImportError:
#    sys.stderr.write("\nError: hsa_fetch module not found!\n")
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
fequdiv = '=' * 80

##--------------------------------------------------------------------------##
## Catch interruption cleanly:
def signal_handler(signum, frame):
    sys.stderr.write("\nInterrupted!\n\n")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

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
#def ldmap(things):
#    return dict(zip(things, range(len(things))))
#
#def argnear(vec, val):
#    return (np.abs(vec - val)).argmin()




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
    Download data from Spitzer Heritage Archive by sky coordinate.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #parser.set_defaults(data_source=None)
    #telgroup = parser.add_argument_group('Data/Telescope Choice')
    #telgroup = telgroup.add_mutually_exclusive_group()
    #telgroup.add_argument('-S', '--spitzer', required=False,
    #        dest='data_source', action='store_const', const='spitzer',
    #        help='retrieve data from Spitzer Heritage Archive')
    #telgroup.add_argument('--CFHT', required=False,
    #        dest='data_source', action='store_const', const='CFHT',
    #        help='retrieve data from CFHT archive')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-f', '--fetch_list', required=False, default=None,
            help='save list of download(ed) files to FILE', type=str)
    iogroup.add_argument('-o', '--output_dir', required=True, default=None,
            help='output folder for retrieved files', type=str)
    iogroup.add_argument('-t', '--target_list', required=True, default=None,
            help='input list of target coordinates', type=str)
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    srchgroup = parser.add_argument_group('Search Options')
    srchgroup.add_argument('-R', '--radius', required=False, default=0.05,
            dest='search_rad_deg', help='radius of search cone in DEGREES')
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

### Quit if no imaging DB selected:
#if not context.data_source:
#    sys.stderr.write("Must select imaging source!\n")
#    sys.exit(1)

## Create target list:
targets = []

## Ensure target list exists:
if not os.path.isfile(context.target_list):
    sys.stderr.write("Error: file not found: %s\n" % context.target_list)
    sys.exit(1)

targets = cfr.load_coords(context.target_list)
sys.stderr.write("\n%s\n" % fequdiv)
sys.stderr.write("Loaded %d target(s) from file:\n" % len(targets))
sys.stderr.write("--> %s\n" % context.target_list)

## Warn/abort if no targets:
if not targets:
    sys.stderr.write("No targets loaded from file(s)!  Nothing to do.\n")
    sys.exit(0)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Name parsing helpers:
def aor_from_ibase(base_name):
    return base_name.split('_')[2]

## Flavor-specific storage path for a known output basename:
def make_storage_relpath(save_name):
    imname = os.path.basename(save_name)
    #subdir = None
    subdir = 'r' + aor_from_ibase(imname)
    impath = os.path.join(subdir, imname) if subdir else imname
    return impath

def make_flavor_imname(item, suffix):
    return item['ibase'].replace('_bcd.fits', suffix)

#def make_storage_relpath(save_name, suffix):
#    imname = os.path.basename(save_name).replace('_bcd.fits', suffix)
#    imname = 
#    subdir = None
#    impath = os.path.join(subdir, imname) if subdir else imname
#    return impath


## Retrieve item+ancillary and save zip archive:
def get_all_as_zip(item, save_path,
        urlkey='accessWithAnc1Url', tpath='./.spitzer', addpid=True):
    data_url = item[urlkey].strip()
    if addpid:
        tpath += str(os.getpid())
    _tmp_dir = os.path.dirname(tpath) + '/'
    _tmpfile = os.path.basename(tpath)
    _tmppath = tpath + '.zip'
    sha.save_file(data_url, out_dir=_tmp_dir, out_name=_tmpfile)
    if not os.path.isfile(_tmppath):
        sys.stderr.write("result missing!!!!")
        return False
    shutil.move(_tmppath, save_path)
    return True

def unzip_and_move_by_suffix(zfile, suffix, outdir):
    with ZipFile(zfile, 'r') as zz:
        for zi in zz.infolist():
            if zi.filename.endswith(suffix):
                #zi.filename = os.path.basename(zi.filename)
                zi.filename = make_storage_relpath(zi.filename)
                zz.extract(zi, outdir)
    return

## Driver routine to retrieve zip file and unpack desired flavors:
def retrieve_anc_zip(item, suff_list, outdir):
    tzpath = 'temp_%s.zip' % str(os.getpid())
    if not get_all_as_zip(item, tzpath):
        sys.stderr.write("retrieval error!!\n")
        return False
    for flavor in suff_list:
        unzip_and_move_by_suffix(tzpath, flavor, outdir)
    os.unlink(tzpath)
    return True

## Check whether files already retrieved:
def already_have_data(item, suff_list, outdir):
    found = []
    for suffix in suff_list:
        flav_name = make_flavor_imname(item, suffix)
        flav_path = os.path.join(outdir, make_storage_relpath(flav_name))
        found.append(os.path.isfile(flav_path))
    return all(found)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Search for results:
max_imgs = 0
max_objs = 0
tmp_zsave = 'temp.zip'
wanted_instruments = ['I1', 'I2']
wanted_image_types = ['_bcd.fits', '_cbcd.fits', '_cbunc.fits']
full_frame_npixels = 256
#data_storage_specs = {'bcd':'_bcd.fits', 'cbcd':'_cbcd.fits'}
for nn,targ in enumerate(targets, 1):
    sys.stderr.write("%s\n" % fulldiv)

    # retrieve query results
    sys.stderr.write("Querying SHA database ... ")
    try:
        hits = sha.query(coord=targ, size=context.search_rad_deg, dataset=1)
    except IndexError:
        sys.stderr.write("query failed (probably no matches).\n")
        sys.exit(0)
    sys.stderr.write("done.\n")

    # Added value and sanity checking:
    hits['ibase'] = [os.path.basename(x.strip()) for x in hits['externalname']]
    if not all([x.startswith('SPITZER') for x in hits['ibase']]):
        sys.stderr.write("Unexpected file name structure, please address!\n")
        sys.exit(1)
    hits['bcd_url'] = [x.strip() for x in hits['accessUrl']]
    hits['anc_url'] = [x.strip() for x in hits['accessWithAnc1Url']]
    hits[  'instr'] = [x.split('_')[1] for x in hits['ibase']]
    #hits[ 'aorkey'] = [x.split('_')[2] for x in hits['ibase']]
    hits[ 'aorkey'] = [aor_from_ibase(x) for x in hits['ibase']]
    #sys.exit(0)

    # drop unavailable files:
    blocked = (hits['bcd_url'] == 'NONE')
    keepers = hits[~blocked]

    # select full-frame IRAC images:
    is_irac = np.array([x in wanted_instruments for x in keepers['instr']])
    keepers = keepers[is_irac]
    keepers = keepers[keepers['naxis1'] == full_frame_npixels]  # full in X
    keepers = keepers[keepers['naxis2'] == full_frame_npixels]  # full in Y
    nfound  = len(keepers)
    sys.stderr.write("Found %d images to download around:\n%s\n"
            % (nfound, str(targ)))

    # optionally augment list of files to fetch:
    if context.fetch_list:
        sys.stderr.write("Saving list of images to fetch ... ")
        with open(context.fetch_list, 'a') as fl:
            for item in keepers['ibase']:
                fl.write("%s\n" % item)
        sys.stderr.write("done.\n")

    # retrieve everything:
    sys.exit(0)
    n_retrieved = 0
    for ii,item in enumerate(keepers, 1):
        sys.stderr.write("\rFile %d of %d: %s ... "
                % (ii, nfound, item['ibase']))
        if already_have_data(item, wanted_image_types, context.output_dir):
            sys.stderr.write("already retrieved!     ")
            continue
        n_retrieved += 1

        sys.stderr.write("downloading ... ")
        if not retrieve_anc_zip(item, wanted_image_types, context.output_dir):
            sys.stderr.write("problem!!!\n")
            sys.exit(1)
        sys.stderr.write("done.\n") 
        if (max_imgs > 0) and (n_retrieved >= max_imgs):
            sys.stderr.write("Stopping early (max_imgs=%d)\n" % max_imgs)
            sys.exit(0)
            #break

    sys.stderr.write("\n")
    if (max_objs > 0) and (nn >= max_objs):
        sys.stderr.write("Stopping early (max_objs=%d)\n" % max_objs)
        sys.exit(0)
        #break


######################################################################
# CHANGELOG (fetch_hsa_data.py):
#---------------------------------------------------------------------
#
#  2019-08-29:
#     -- Increased __version__ to 0.3.0.
#     -- Cleaned up lots of no-longer-useful code.
#     -- Added support for future separation of flavors into subfolders.
#     -- Check for existing files now covers all wanted image types.
#     -- Now keep both _bcd and _cbcd FITS files.
#
#  2019-08-28:
#     -- Increased __version__ to 0.2.0.
#     -- Now download bcd+ancillary data in zip file.
#
#  2019-08-27:
#     -- Increased __version__ to 0.1.0.
#     -- First created fetch_sha_data.py.
#

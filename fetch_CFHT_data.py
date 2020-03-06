#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Fetch Spitzer Heritage Archive data relevant to specified object.
#
# Rob Siverd
# Created:       2020-03-02
# Last modified: 2020-03-04
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.0"

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
import gc
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import itertools as itt
#from zipfile import ZipFile
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Fancy downloading:
try:
    import downloading
    reload(downloading)
    fdl = downloading.Downloader()
except ImportError:
    sys.stderr.write("\nRequired 'downloading' module not found!\n")
    #raise ImportError

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
    import astropy.table as apt
    from astroquery.cadc import Cadc
    cadc = Cadc()
    from astropy import coordinates as coord
    from astropy import units as uu
except ImportError:
    sys.stderr.write("\nError: astropy/astroquery import failed!\n")
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
    #parser.set_defaults(temp_folder='/tmp')
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
    iogroup.add_argument('--temp_folder', required=False, default='/tmp',
            help='where to store in-process downloads')
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
    #        help='KELT image with WCS')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    srchgroup = parser.add_argument_group('Search Options')
    srchgroup.add_argument('-R', '--radius', required=False, default=0.1,
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

## Ensure target list exists:
if not os.path.isfile(context.target_list):
    sys.stderr.write("Error: file not found: %s\n" % context.target_list)
    sys.exit(1)

## Slightly less dumb parsing (assume deg units if unspecified):
def skycoordify(text):
    tcoo = None
    try:
        tcoo = coord.SkyCoord(text)
    except:
        try:
            tcoo = coord.SkyCoord(text, unit="deg")
        except:
            sys.stderr.write("Failed to parse coordinates: '%s'\n" % text)
    return tcoo

## Load data from list:
with open(context.target_list, 'r') as f:
    contents = [x.strip() for x in f.readlines()]

## Parse target info:
delim = '#'
targets = []
for ii, line in enumerate(contents, 1):
    tname = 'pointing%03d' % ii
    if delim in line:
        tname = line.split(delim)[1].strip()
    nocomment = line.split(delim)[0].strip()
    tcoord = skycoordify(nocomment)
    if tcoord:
        targets.append((tcoord, tname))


#    #fldnames = []
#    for ii,line in enumerate([x.strip() for x in f.readlines()], 1):
#        nocomment = line.split('#')[0].strip()
#        contents.append(nocomment)
#        if not '#' in line:
#            sys.stderr.write("\n")
#
### Make SkyCoords (resort to deg, deg if parse fails):
#targets += [skycoordify(x) for x in contents]
#targets = [x for x in targets if x]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

### Flavor-specific storage path for a known output basename:
#def make_storage_relpath(save_name):
#    imname = os.path.basename(save_name)
#    subdir = None
#    impath = os.path.join(subdir, imname) if subdir else imname
#    return impath
#
#def make_flavor_imname(item, suffix):
#    return item['ibase'].replace('_bcd.fits', suffix)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Ensure presence of temporary folder:
if not os.path.isdir(context.temp_folder):
    sys.stderr.write("\nError: temporary folder '%s' not found!\n"
            % context.temp_folder)
    sys.exit(1)

## Ensure presence of output folder:
if not os.path.isdir(context.output_dir):
    sys.stderr.write("\nError: output folder '%s' not found!\n"
            % context.output_dir)
    sys.exit(1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Search for results:
srch_deg = uu.deg * context.search_rad_deg
max_imgs = 0
max_objs = 0
#tmp_zsave = 'temp.zip'
#wanted_instruments = ['I1', 'I2']
#wanted_instruments = ['WIRCam', 'jeffcam']
#wanted_image_types = ['_bcd.fits', '_cbcd.fits', '_cbunc.fits']
#data_storage_specs = {'bcd':'_bcd.fits', 'cbcd':'_cbcd.fits'}
#searchkw = {'radius':uu.deg * context.search_rad_deg, 'collection':'CFHT'}

## How to select things 
instr_names = ['WIRCam'] #, 'jeffcam']
prod_suffix = ['p', 's']
ntodo = 0
chunksize = 50
tmp_dl_base = 'tmpfetch.%d.fits' % os.getpid()
tmp_dl_path = os.path.join(context.temp_folder, tmp_dl_base)
fdl.prefer_requests()

## Select the things I want:
def pick_favorites(results):
    # work on a duplicate:
    hits = results.copy()

    # select desired instruments:
    hits = apt.vstack([hits[(hits['instrument_name'] == x)] for x in instr_names])

    # select desired productID suffixes:
    which = np.array([any([pp.endswith(x) for x in prod_suffix]) \
            for pp in hits['productID']])
    hits = hits[which]
    # return whatever remains:
    return hits
    
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

for nn,tinfo in enumerate(targets, 1):
    targ, tname = tinfo
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Target %d of %d: %s\n" % (nn, len(targets), tname))

    # ensure output folder exists:
    save_dir = os.path.join(context.output_dir, tname)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # retrieve query results
    sys.stderr.write("Querying CFHT database ... ")
    #hits = cadc.query(coord=targ, size=context.search_rad_deg, dataset=1)
    hits = cadc.query_region(targ, radius=srch_deg, collection='CFHT')
    hits.sort('productID')
    sys.stderr.write("done.\n")

    # select useful subset:
    sys.stderr.write("Selecting useful subset ... ")
    useful = pick_favorites(hits)
    useful['ibase'] = ['%s.fits.fz'%x for x in useful['productID']]
    useful['isave'] = [os.path.join(save_dir, x) for x in useful['ibase']]
    sys.stderr.write("done. Identified %d images.\n" % len(useful))

    # exclude images already retrieved:
    already_have = np.array([os.path.isfile(x) for x in useful['isave']])
    nfound = np.sum(already_have)
    sys.stderr.write("Excluding %d already-retrieved images.\n" % nfound)
    useful = useful[~already_have]
    if (len(useful) == 0):
        sys.stderr.write("Nothing to do, next object!\n")
        continue

    if (ntodo > 0):
        useful = useful[:ntodo]

    # Download new images in chunks (URL-fetch is slow):
    #sys.stderr.write("making URLs ... ")
    nchunks = int(np.ceil(len(useful) / float(chunksize)))
    sys.stderr.write("nchunks: %d\n" % nchunks)
    uidx = np.arange(len(useful))
    chunkidx = np.array_split(np.arange(len(useful)), nchunks)
    for ii,cidx in enumerate(chunkidx, 1):
        sys.stderr.write("Chunk %d of %d ...\n" % (ii, nchunks))
        snag = useful[cidx]
        imurls = cadc.get_data_urls(snag)
        dlspec = [(uu, ss, tmp_dl_path) for uu,ss in zip(imurls, snag['isave'])]
        fdl.smart_fetch_bulk(dlspec)



######################################################################
# CHANGELOG (fetch_hsa_data.py):
#---------------------------------------------------------------------
#
#  2020-03-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created fetch_CFHT_data.py.
#

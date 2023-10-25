#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Fetch Canada-France-Hawaii Telescope (CFHT) images from the Canadian
# Astronomy Data Centre (CADC).
#
# Rob Siverd
# Created:       2020-03-02
# Last modified: 2023-10-24
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.3"

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
import re
import sys
import time
import numpy as np
import warnings
import requests
import traceback
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
#try:
#    import downloading
#    reload(downloading)
#    fdl = downloading.Downloader()
#except ImportError:
#    sys.stderr.write("\nRequired 'downloading' module not found!\n")
#    #raise ImportError

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
    import astropy.time as astt
    import astropy.io.fits as pf
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
    Download CFHT data from CADC by sky coordinate.
    
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
            help='output folder for CADC files', type=str)
    iogroup.add_argument('-t', '--target_list', required=True, default=None,
            help='input list of target coordinates', type=str)
    iogroup.add_argument('--subdirectory', required=False, default=None,
            help='subfolder (within target) for downloads', type=str)
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
prod_suffix = ['o', 'p', 's']
ntodo = 0
chunksize = 50
tmp_dl_base = 'tmpfetch.%d.fits' % os.getpid()
tmp_dl_path = os.path.join(context.temp_folder, tmp_dl_base)
#fdl.prefer_requests()

## Select the things I want:
def pick_favorites(results):
    # work on a duplicate:
    hits = results.copy()

    # select desired instruments:
    hits = apt.vstack([hits[(hits['instrument_name'] == x)] \
            for x in instr_names])

    # select desired productID suffixes:
    which = np.array([any([pp.endswith(x) for x in prod_suffix]) \
            for pp in hits['productID']])
    hits = hits[which]
    # return whatever remains:
    return hits

## Impose my fixed-width naming convention:
def fixed_width_productID(prodID):
    # get leading digits:
    digits = re.search(r'\d+', prodID).group()
    seqnum = int(digits)
    #sys.stderr.write("%s --> digits: %s\n" % (prodID, digits))
    #flavor = prodID[-1]
    flavor = prodID[len(digits):]
    if len(flavor) > 1:
        sys.stderr.write("Weirdo: %s\n" % prodID)
    #seqnum = int(prodID[:-1])
    return '%07d%s' % (seqnum, flavor) 
    #return '%07d%c' % (seqnum, flavor) 

##--------------------------------------------------------------------------##
##------------------      Download and Validate Images      ----------------##
##--------------------------------------------------------------------------##

def file_is_FITS(filename):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pix = pf.getdata(filename)
        return True
    except:
        return False

def rfetch(url, save_file):
    with open(save_file, 'wb') as f:
        f.write(requests.get(url).content)
    return

def download_from_cadc(dlspec, stream=sys.stderr):
    total = len(dlspec)
    for ii,(url, save_path, tmp_path) in enumerate(dlspec, 1):
        # retrieve data:
        stream.write("Downloading file %d of %d ... " % (ii, total))
        try:
            rfetch(url, tmp_path)
        except (KeyboardInterrupt, SystemExit) as e:
            raise e
        except:
            stream.write("error during download!\n")
            trouble = sys.exc_info()
            etype, evalue, etrace = sys.exc_info()
            sys.stderr.write("etype: %s\n" % str(etype))
            sys.stderr.write("evalue: %s\n" % str(evalue))
            print(etrace)
            print(dir(etrace))
            sys.stderr.write("PROBLEM: %s\n" % str(trouble))
            print(traceback.format_exc())
            #return False
            continue

        # validate file:
        stream.write("validating ... ")
        if not file_is_FITS(tmp_path):
            stream.write("failed!\n")
            os.unlink(tmp_path)
            #return False
            continue

        # move to final destination:
        stream.write("moving ... ") 
        shutil.move(tmp_path, save_path)
        stream.write("done.\n")
        pass
    return

##--------------------------------------------------------------------------##
##------------------         Download Everything            ----------------##
##--------------------------------------------------------------------------##

## Allow saving of CADC query results for debugging:
save_cadc_hits = False
cadc_hits_file = 'cadc_hits.pickle'

## Iterate over targets:
for nn,tinfo in enumerate(targets, 1):
    targ, tname = tinfo
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Target %d of %d: %s\n" % (nn, len(targets), tname))

    # ensure root output folder exists:
    save_dir = os.path.join(context.output_dir, tname)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # set and create subfolder, if requested:
    if context.subdirectory:
        save_dir = os.path.join(save_dir, context.subdirectory)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # retrieve query results
    sys.stderr.write("Querying CFHT database ... ")
    #hits = cadc.query(coord=targ, size=context.search_rad_deg, dataset=1)
    hits = cadc.query_region(targ, radius=srch_deg, collection='CFHT')
    hits['fw_prod_id'] = [fixed_width_productID(x) for x in hits['productID']]
    #hits.sort('productID')
    hits.sort('fw_prod_id')
    sys.stderr.write("done.\n")
    sys.stderr.write("Found %d products at position.\n" % len(hits))

    # save results payload for inspection:
    if save_cadc_hits:
        import pickle
        #import pdb; pdb.set_trace()
        with open(cadc_hits_file, 'wb') as chp:
            pickle.dump(hits, chp)
        sys.exit(1)
        ### reload with:
        ##with open(cadc_hits_file, 'rb') as chp:
        ##    hits = pickle.load(chp)

    # select useful subset:
    sys.stderr.write("Selecting useful subset ... ")
    useful = pick_favorites(hits)
    #useful['ibase'] = ['%s.fits.fz'%x for x in useful['productID']]
    useful['ibase'] = ['%s.fits.fz'%x for x in useful['fw_prod_id']]
    useful['isave'] = [os.path.join(save_dir, x) for x in useful['ibase']]
    #import pdb; pdb.set_trace()
    sys.stderr.write("done. Identified %d images.\n" % len(useful))
    if len(useful) < 1:
        sys.stderr.write("Nothing to fetch!\n")
        continue

    # exclude not-yet-public data (will not work in this script):
    #isot_format = '%Y-%m-%dT%H:%M:%S'
    #public_date = useful['dataRelease']
    #dt.datetime.strptime(public_date[-1].split('.')[0], isot_format)
    pub_date = astt.Time(useful['dataRelease'], scale='utc', format='isot')
    now_sec  = time.time()
    private  = (time.time() < pub_date.unix)
    n_priv   = np.sum(private)
    useful   = useful[~private]
    n_remain = len(useful)
    sys.stderr.write("Dropping %d not-yet-public images.\n" % np.sum(private))
    sys.stderr.write("Have %d public images remaining.\n" % n_remain)
    if (n_remain == 0):
        sys.stderr.write("Nothing to do, next object!\n")
        continue


    # exclude images already retrieved:
    already_have = np.array([os.path.isfile(x) for x in useful['isave']])
    nfound = np.sum(already_have)
    sys.stderr.write("Excluding %d already-retrieved images.\n" % nfound)
    useful = useful[~already_have]
    n_remain = len(useful)
    if (n_remain == 0):
        sys.stderr.write("Nothing to do, next object!\n")
        continue
    sys.stderr.write("Found %d images to fetch.\n" % n_remain)

    if (ntodo > 0):
        useful = useful[:ntodo]

    # Download new images in chunks (URL-fetch is slow):
    #sys.stderr.write("making URLs ... ")
    nchunks = int(np.ceil(len(useful) / float(chunksize)))
    #sys.stderr.write("nchunks: %d\n" % nchunks)
    uidx = np.arange(len(useful))
    chunkidx = np.array_split(np.arange(len(useful)), nchunks)
    for ii,cidx in enumerate(chunkidx, 1):
        sys.stderr.write("Chunk %d of %d ...\n" % (ii, nchunks))
        snag = useful[cidx]
        imurls = cadc.get_data_urls(snag)
        if len(imurls) != len(snag):
            sys.stderr.write("Mismatch in download URL list!\n")
            # SEEN WITH:
            # ['https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/raven/files/cadc:CFHT/2245332o.fits',
            #  'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/raven/files/cadc:CFHT/2245332o.fits.fz']
            #
            continue
        dlspec = [(uu, ss, tmp_dl_path) for uu,ss in zip(imurls, snag['isave'])]
        #fdl.smart_fetch_bulk(dlspec)
        download_from_cadc(dlspec)
        #for uu,ss in zip(imurls, snag['isave']):
        #    success = download_from_cadc(uu, ss, tmp_dl_path)




######################################################################
# CHANGELOG (fetch_CFHT_data.py):
#---------------------------------------------------------------------
#
#  2023-05-24:
#     -- Increased __version__ to 0.2.1.
#     -- Added warning for weird mismatch in URLs from CADC.
#     -- Now exclude not-yet-public images from download list.
#
#  2023-05-23:
#     -- Increased __version__ to 0.2.0.
#     -- Added ability to specify raw downloads subfolder.
#     -- Now skip download loop when useful subset has zero length.
#     -- Added 'o' type products to download queue.
#     -- Tested working with calib1/calib2 and target fields.
#     -- Fixed comments in help menu.
#
#  2020-03-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created fetch_CFHT_data.py.
#

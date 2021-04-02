#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Iterate over a set of stored ExtendedCatalog files. For each file,
# * match sources to Gaia
# * calculate RA- and Dec-offsets
# * nudge ExtendedCatalog values to eliminate offsets
#
# This procedure compensates for significant zero-point errors in the WCS
# provided by Spitzer Heritage Archive.
#
# Rob Siverd
# Created:       2021-03-31
# Last modified: 2021-04-01
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
#from numpy.lib.recfunctions import append_fields
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Angular math tools:
try:
    import angle
    reload(angle)
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Easy Gaia source matching:
try:
    import gaia_match
    reload(gaia_match)
    gm = gaia_match.GaiaMatch()
except ImportError:
    logger.error("failed to import gaia_match module!")
    sys.exit(1)

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

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

## Read ASCII file to list:
def read_column(filename, column=0, delim=' ', strip=True):
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.split(delim)[column] for x in content]
    if strip:
        content = [x.strip() for x in content]
    return content

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
    Perform in-place correction of WCS zero-point errors in
    ExtendedCatalog FITS files.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(imtype=None) #'cbcd') #'clean')
    parser.set_defaults(gaia_tol_arcsec=2.0)
    parser.set_defaults(min_gaia_matches=10)
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-C', '--cat_list', default=None, required=True,
            help='list of full paths to ExtendedCatalog files')
    iogroup.add_argument('-G', '--gaia_csv', default=None, required=True,
            help='CSV file with Gaia source list', type=str)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #iogroup = parser.add_argument_group('File I/O')
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
##------------------       load Gaia sources from CSV       ----------------##
##--------------------------------------------------------------------------##

if context.gaia_csv:
    try:
        logger.info("Loading sources from %s" % context.gaia_csv)
        gm.load_sources_csv(context.gaia_csv)
    except:
        logger.error("failed to load from %s" % context.gaia_csv)
        sys.exit(1)

##--------------------------------------------------------------------------##
##------------------      load ExtendedCatalogs list        ----------------##
##--------------------------------------------------------------------------##

## Read list of ExCat files:
cat_files = read_column(context.cat_list)

##--------------------------------------------------------------------------##
##------------------      Gaia-based offset calculator      ----------------##
##--------------------------------------------------------------------------##

def find_gaia_matches(stars, tol_arcsec, ra_col='wdra', de_col='wdde'):
    tol_deg = tol_arcsec / 3600.0
    matches = []
    for target in stars:
        sra, sde = target[ra_col], target[de_col]
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            #sys.stderr.write("got one!\n")
            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
            matches.append((sra, sde, *gcoords))
            pass
        pass
    return matches
    #have_ra, have_de, gaia_ra, gaia_de = np.array(matches).T

def compute_offset(match_list):
    have_ra, have_de, gaia_ra, gaia_de = np.array(match_list).T
    ra_diffs = gaia_ra - have_ra
    de_diffs = gaia_de - have_de
    delta_ra, delta_ra_sig = rs.calc_ls_med_IQR(ra_diffs)
    delta_de, delta_de_sig = rs.calc_ls_med_IQR(de_diffs)
    ratio_ra = delta_ra / delta_ra_sig
    ratio_de = delta_de / delta_de_sig
    sys.stderr.write("delta_ra: %8.3f (%6.3f)\n" % (3600.*delta_ra, ratio_ra))
    sys.stderr.write("delta_de: %8.3f (%6.3f)\n" % (3600.*delta_de, ratio_de))
    return {'gradelta':delta_ra,    'grasigma':delta_ra_sig,
            'gdedelta':delta_de,    'gdesigma':delta_de_sig,}
    #return (delta_ra, delta_ra_sig, delta_de, delta_de_sig)

##--------------------------------------------------------------------------##
##------------------        update catalog sky coords       ----------------##
##--------------------------------------------------------------------------##

## Catalog object:
ccc = ec.ExtendedCatalog()

## Lists of coordinate keys:
_ra_keys = ['dra', 'wdra', 'ppdra']
_de_keys = ['dde', 'wdde', 'ppdde']

## Update files:
for cfile in cat_files:
    sys.stderr.write("Catalog: %s\n" % cfile)
    goffset = {'gradelta':0.0, 'grasigma':0.0,
               'gdedelta':0.0, 'gdesigma':0.0,}

    # load ExtCat, skip if processed:
    ccc.load_from_fits(cfile)
    if 'gmatches' in ccc._imeta.keys():
        sys.stderr.write("Catalog already nudged!\n")
        continue
    sys.stderr.write("Not yet nudged ...\n")

    # calculate and apply nudge if needed:
    stars = ccc.get_catalog()
    match_list = find_gaia_matches(stars, context.gaia_tol_arcsec)
    nmatched = len(match_list)
    sys.stderr.write("nmatched: %d\n" % nmatched)

    # nudge coordinates if matches sufficient:
    if (nmatched >= context.min_gaia_matches):
        goffset = compute_offset(match_list)
        fixed = stars.copy()
        for kk in _ra_keys:
            fixed[kk] = fixed[kk] + goffset['gradelta']
        for kk in _de_keys:
            fixed[kk] = fixed[kk] + goffset['gdedelta']
        ccc.set_catalog(fixed)
    ccc._imeta['gmatches'] = nmatched
    ccc._imeta.update(goffset)
    ccc.save_as_fits(cfile, overwrite=True)

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

######################################################################
# CHANGELOG (08_inplace_fix_WCS_offsets.py):
#---------------------------------------------------------------------
#
#  2021-03-31:
#     -- Increased __version__ to 0.1.0.
#     -- First created 08_inplace_fix_WCS_offsets.py.
#

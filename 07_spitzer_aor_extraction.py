#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract and save extended object catalogs from the specified data and
# uncertainty images. This version of the script jointly analyzes all
# images from a specific AOR/channel to enable more sophisticated
# analysis.
#
# Rob Siverd
# Created:       2021-02-02
# Last modified: 2021-02-16
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
__version__ = "0.2.5"

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
#import signal
import glob
#import gc
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##
## Spitzer pipeline filesystem helpers:
try:
    import spitz_fs_helpers
    reload(spitz_fs_helpers)
except ImportError:
    logger.error("failed to import spitz_fs_helpers module!")
    sys.exit(1)
sfh = spitz_fs_helpers

## Spitzer pipeline cross-correlation:
try:
    import spitz_xcorr_stacking
    reload(spitz_xcorr_stacking)
except ImportError:
    logger.error("failed to import spitz_xcor_stacking module!")
    sys.exit(1)
sxc = spitz_xcorr_stacking.SpitzerXCorr()

## Catalog pruning helpers:
try:
    import catalog_tools
    reload(catalog_tools)
except ImportError:
    logger.error("failed to import catalog_tools module!")
    sys.exit(1)
xcp = catalog_tools.XCorrPruner()

## Spitzer star detection routine:
try:
    import spitz_extract
    reload(spitz_extract)
    spf = spitz_extract.SpitzFind()
except ImportError:
    logger.error("spitz_extract module not found!")
    sys.exit(1)

##--------------------------------------------------------------------------##
## Fast FITS I/O:
try:
    import fitsio
except ImportError:
    logger.error("fitsio module not found!  Install and retry.")
    sys.stderr.write("\nError: fitsio module not found!\n")
    sys.exit(1)

## Save FITS image with clobber (fitsio):
def qsave(iname, idata, header=None, **kwargs):
    this_func = sys._getframe().f_code.co_name
    parent_func = sys._getframe(1).f_code.co_name
    sys.stderr.write("Writing to '%s' ... " % iname)
    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
    sys.stderr.write("done.\n")

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
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

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
    Extract catalogs from the listed Spitzer data/uncertainty images.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt)
                          #formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    parser.set_defaults(imtype=None) #'cbcd') #'clean')
    parser.set_defaults(sigthresh=3.0)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-I', '--input_folder', default=None, required=True,
            help='where to find input images', type=str)
    iogroup.add_argument('-O', '--output_folder', default=None, required=False,
            help='where to save extended catalog outputs', type=str)
    imtype = iogroup.add_mutually_exclusive_group()
    #imtype.add_argument('--cbcd', required=False, action='store_const',
    #        dest='imtype', const='cbcd', help='use cbcd images')
    imtype.add_argument('--hcfix', required=False, action='store_const',
            dest='imtype', const='hcfix', help='use hcfix images')
    imtype.add_argument('--clean', required=False, action='store_const',
            dest='imtype', const='clean', help='use clean images')
    iogroup.add_argument('-W', '--walk', default=False, action='store_true',
            help='recursively walk subfolders to find CBCD images')
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

    # Unless otherwise specified, output goes into input folder:
    if not context.output_folder:
        context.output_folder = context.input_folder

    # Ensure an image type is selected:
    if not context.imtype:
        sys.stderr.write("\nNo image type selected!\n\n")
        sys.exit(1)

##--------------------------------------------------------------------------##
##------------------         Make Input Image List          ----------------##
##--------------------------------------------------------------------------##

tstart = time.time()
sys.stderr.write("Listing %s frames ... " % context.imtype) 
#im_wildpath = 'SPITZ*%s.fits' % context.imtype
#im_wildcard = os.path.join(context.input_folder, 'SPIT*'
#_img_types = ['cbcd', 'clean', 'cbunc']
#_type_suff = dict([(x, x+'.fits') for x in _im_types])
#img_list = {}
#for imsuff in suffixes:
#    wpath = '%s/SPITZ*%s.fits' % (context.input_folder, imsuff)
#    img_list[imsuff] = sorted(glob.glob(os.path.join(context.
#img_files = sorted(glob.glob(os.path.join(context.input_folder, im_wildpath)))

if context.walk:
    img_files = sfh.get_files_walk(context.input_folder, flavor=context.imtype)
else:
    img_files = sfh.get_files_single(context.input_folder, flavor=context.imtype)
sys.stderr.write("done.\n")

## Abort in case of no input:
if not img_files:
    sys.stderr.write("No input (%s) files found in folder:\n" % context.imtype)
    sys.stderr.write("--> %s\n\n" % context.input_folder)
    sys.exit(1)

n_images = len(img_files)

## List of uncertainty frames (warn if any missing):
#unc_files = [x.replace(context.imtype, 'cbunc') for x in img_files]
#sys.stderr.write("Checking error-images ... ") 
#have_unc = [os.path.isfile(x) for x in unc_files]
#if not all(have_unc):
#    sys.stderr.write("WARNING: some uncertainty frames missing!\n")
#else:
#    sys.stderr.write("done.\n") 

##--------------------------------------------------------------------------##
##------------------       Unique AOR/Channel Combos        ----------------##
##--------------------------------------------------------------------------##

unique_tags = sorted(list(set([sfh.get_irac_aor_tag(x) for x in img_files])))
images_by_tag = {x:[] for x in unique_tags}
for ii in img_files:
    images_by_tag[sfh.get_irac_aor_tag(ii)].append(ii)


##--------------------------------------------------------------------------##
##------------------        Diagnostic Region Files         ----------------##
##--------------------------------------------------------------------------##

def regify_excat_pix(data, rpath, win=False, rr=2.0):
    colnames = ('wx', 'wy') if win else ('x', 'y')
    xpix, ypix = [data[x] for x in colnames]
    with open(rpath, 'w') as rfile:
        for xx,yy in zip(xpix, ypix):
            rfile.write("image; circle(%8.3f, %8.3f, %8.3f)\n" % (xx, yy, rr))
    return

##--------------------------------------------------------------------------##
##------------------         Stack/Image Comparison         ----------------##
##--------------------------------------------------------------------------##

#def xcheck(idata, sdata):
#    nstack = len(sdata)
#    nimage = len(idata)
#    sys.stderr.write("nstack: %d\n" % nstack)
#    sys.stderr.write("nimage: %d\n" % nimage)
#    return

##--------------------------------------------------------------------------##
##------------------           Process All Images           ----------------##
##--------------------------------------------------------------------------##

ntodo = 0
nproc = 0
ntotal = len(img_files)
min_sobj = 10       # bark if fewer than this many found in stack

skip_stuff = False

context.save_registered = True

#for aor_tag,tag_files in images_by_tag.items():
for aor_tag in unique_tags:
    sys.stderr.write("\n\nProcessing images from %s ...\n" % aor_tag)
    tag_files = images_by_tag[aor_tag]

    # File/folder paths:
    aor_dir = os.path.dirname(tag_files[0])
    stack_ibase = '%s_%s_stack.fits' % (aor_tag, context.imtype)
    stack_cbase = '%s_%s_stack.fcat' % (aor_tag, context.imtype)
    medze_ibase = '%s_%s_medze.fits' % (aor_tag, context.imtype)
    stack_ipath = os.path.join(aor_dir, stack_ibase)
    stack_cpath = os.path.join(aor_dir, stack_cbase)
    medze_ipath = os.path.join(aor_dir, medze_ibase)
    #sys.stderr.write("stack_ibase: %s\n" % stack_ibase)

    sys.stderr.write("Cross-correlating and stacking ... ")
    result = sxc.shift_and_stack(tag_files)
    sys.stderr.write("done.\n")
    sxc.save_istack(stack_ipath)
    #istack = sxc.get_stacked()
    #qsave(stack_ipath, istack)

    # Dump registered data to disk:
    if context.save_registered:
        sys.stderr.write("Saving registered frames for inspection ...\n")
        reg_dir = os.path.join(aor_dir, 'zreg')
        if os.path.isdir(reg_dir):
            shutil.rmtree(reg_dir)
        os.mkdir(reg_dir)
        sxc.dump_registered_images(reg_dir)
        sys.stderr.write("\n")

    # Extract stars from stacked image:
    spf.use_images(ipath=stack_ipath)
    stack_cat = spf.find_stars(context.sigthresh)
    stack_cat.save_as_fits(stack_cpath, overwrite=True)
    sdata = stack_cat.get_catalog()
    nsobj = len(sdata)

    if (nsobj < min_sobj):
        sys.stderr.write("Fewer than %d objects found in stack ... \n" % min_sobj)
        sys.stderr.write("Found %d objects.\n\n" % nsobj)
        sys.stderr.write("--> %s\n\n" % stack_ipath)
        sys.exit(1)
    
    # region file for diagnostics:
    stack_rfile = stack_ipath + '.reg'
    regify_excat_pix(sdata, stack_rfile)

    # Make/save 'medianize' stack for comparison:
    sxc.make_mstack()
    sxc.save_mstack(medze_ipath)

    # Set up pruning system:
    xshifts, yshifts = sxc.get_stackcat_offsets()
    xcp.set_master_catalog(sdata)
    xcp.set_image_offsets(xshifts, yshifts)
 
    ## Stop here for now ...
    #if skip_stuff:
    #    continue

    # process individual files with cross-correlation help:
    for ii,img_ipath in enumerate(tag_files, 1):
        sys.stderr.write("%s\n" % fulldiv)
        unc_ipath = img_ipath.replace(context.imtype, 'cbunc')
        if not os.path.isfile(unc_ipath):
            sys.stderr.write("WARNING: file not found:\n--> %s\n" % unc_ipath)
            continue
        img_ibase = os.path.basename(img_ipath)
        #cat_ibase = img_ibase.replace(context.imtype, 'fcat')
        cat_fbase = img_ibase + '.fcat'
        cat_pbase = img_ibase + '.pcat'
        cat_mbase = img_ibase + '.mcat'

        ### FIXME ###
        ### context.output_folder is not appropriate for walk mode ...
        save_dir = context.output_folder    # NOT FOR WALK MODE
        save_dir = os.path.dirname(img_ipath)
        cat_fpath = os.path.join(save_dir, cat_fbase)
        cat_ppath = os.path.join(save_dir, cat_pbase)
        cat_mpath = os.path.join(save_dir, cat_mbase)
        ### FIXME ###

        sys.stderr.write("Catalog %s ... " % cat_fpath)
        if os.path.isfile(cat_ppath):
            sys.stderr.write("exists!  Skipping ... \n")
            continue
        nproc += 1
        sys.stderr.write("not found ... creating ...\n")
        spf.use_images(ipath=img_ipath, upath=unc_ipath)
        result = spf.find_stars(context.sigthresh)
        result.save_as_fits(cat_fpath, overwrite=True)
        nfound = len(result.get_catalog())

        # prune sources not detected in stacked frame:
        pruned = xcp.prune_spurious(result.get_catalog(), img_ipath)
        npruned = len(pruned)
        sys.stderr.write("nfound: %d, npruned: %d\n" % (nfound, npruned))
        if (len(pruned) < 5):
            sys.stderr.write("BARKBARKBARK\n")
            sys.exit(1)
        result.set_catalog(pruned)
        result.save_as_fits(cat_ppath, overwrite=True)

        # stop early if requested:
        if (ntodo > 0) and (nproc >= ntodo):
            break
        #break

    if (ntodo > 0) and (nproc >= ntodo):
        break

tstop = time.time()
ttook = tstop - tstart
sys.stderr.write("Extraction completed in %.3f seconds.\n" % ttook)


#import astropy.io.fits as pf
#
#imra = np.array([hh['CRVAL1'] for hh in sxc._im_hdrs])
#imde = np.array([hh['CRVAL2'] for hh in sxc._im_hdrs])
#
##sys.stderr.write("\n\n\n")
##sys.stderr.write("sxc.shift_and_stack(tag_files)\n")
##result = sxc.shift_and_stack(tag_files)
#sys.exit(0)
#
#layers = sxc.pad_and_shift(sxc._im_data, sxc._x_shifts, sxc._y_shifts)
#tstack = sxc.dumb_stack(layers)
#pf.writeto('tstack.fits', tstack, overwrite=True)
#
#tdir = 'zzz'
#if not os.path.isdir(tdir):
#    os.mkdir(tdir)
#
##tag_bases = [os.path.basename(x) for x in tag_files]
##for ibase,idata in zip(tag_bases, layers):
##    tsave = os.path.join(tdir, 'r' + ibase)
##    sys.stderr.write("Saving %s ... \n" % tsave) 
##    pf.writeto(tsave, idata, overwrite=True)
#
#sys.stderr.write("\n\n\n")
#sys.stderr.write("visual inspection with:\n") 
#sys.stderr.write("flztfs %s\n" % ' '.join(tag_files))

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (07_spitzer_aor_extraction.py):
#---------------------------------------------------------------------
#
#  2021-02-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created 07_spitzer_aor_extraction.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module provides routines to find Spitzer image files on disk and
# and perform various filesystem-related tasks.
#
# Rob Siverd
# Created:       2021-01-29
# Last modified: 2021-02-02
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

## Modules:
#import shutil
#import resource
#import signal
import glob
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##------------------      Input Image List Generators       ----------------##
##--------------------------------------------------------------------------##

def get_files_single(dirpath, flavor):
    want_suffix = '%s.fits' % flavor
    im_wildpath = '%s/SPITZ*_%s' % (dirpath, want_suffix)
    return sorted(glob.glob(im_wildpath))

def get_files_walk(targ_root, flavor):
    files_found = []
    want_suffix = '_%s.fits' % flavor
    for thisdir, subdirs, files in os.walk(targ_root):
        these_files = [x for x in files if x.endswith(want_suffix)]
        files_found += [os.path.join(thisdir, x) for x in these_files]
    return sorted(files_found)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (spitz_fstools.py):
#---------------------------------------------------------------------
#
#  2021-01-29:
#     -- Increased __version__ to 0.1.0.
#     -- First created spitz_fs_helpers.py.
#

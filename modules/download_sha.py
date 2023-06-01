#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module to handle file retrieval from Spitzer Heritage Archive.
#
# Rob Siverd
# Created:       2019-08-27
# Last modified: 2019-08-27
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Python 2/3 compatibility (modules):
from __future__ import absolute_import, division, print_function

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
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

##--------------------------------------------------------------------------##

## Various from astropy:
try:
    from astroquery import sha
    #from astropy import coordinates as coord
    #from astropy import units as uu
except ImportError:
    sys.stderr.write("\nError: astropy/astroquery module not found!\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

class DownloadSHA(object):

    def __init__(self):
        self._save_dir = None
        self._this_obj = None
        self._name_key = 'externalname'
        self._bcd_ukey = 'accessUrl'
        self._anc_ukey = 'accessWithAnc1Url'
        return

    # specify where to save files:
    def set_outdir(self, output_folder):
        if os.path.isdir(output_folder):
            self._save_dir = output_folder
        else:
            sys.stderr.write("Error: folder not found: %s\n" % output_folder)
            sys.stderr.write("Please create folder and retry.\n") 
            return False
        return True

##--------------------------------------------------------------------------##


######################################################################
# CHANGELOG (download_sha.py):
#---------------------------------------------------------------------
#
#  2019-08-27:
#     -- Increased __version__ to 0.1.0.
#     -- First created download_sha.py.
#

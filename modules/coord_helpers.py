#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module provides routines to work with targets lists and image 
# coordinates for parallax pipeline image processing.
#
# Rob Siverd
# Created:       2021-02-12
# Last modified: 2021-02-14
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

## Various from astropy:
try:
    from astropy import coordinates as coord
except ImportError:
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


##--------------------------------------------------------------------------##
##------------------       Target Coordinate Loading        ----------------##
##--------------------------------------------------------------------------##

class CoordFileReader(object):

    def __init__(self):
        return

    def load_coords(self, filename):
        contents = self._read_text_file(filename)
        targets = [self._skycoordify(x) for x in contents]
        targets = [x for x in targets if x]
        return targets

    # -------------------------------
    # Helpers:
    # -------------------------------

    @staticmethod
    def _read_text_file(filename):
        with open(filename, 'r') as f:
            contents = []
            for line in [x.strip() for x in f.readlines()]:
                if line.startswith('#'):
                    continue    # skip comments
                nocomment = line.split('#')[0].strip()
                contents.append(nocomment)
                pass
            pass
        return contents

    @staticmethod
    def _skycoordify(text):
        tcoo = None
        try:
            tcoo = coord.SkyCoord(text)
        except:
            try:
                tcoo = coord.SkyCoord(text, unit="deg")
            except:
                sys.stderr.write("Failed to parse coordinates: '%s'\n" % text)
        return tcoo


##--------------------------------------------------------------------------##
##------------------    Image-in-Field Testing with WCS     ----------------##
##--------------------------------------------------------------------------##

class WCSCoordChecker(object):

    def __init__(self):
        return


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (spitz_fstools.py):
#---------------------------------------------------------------------
#
#  2021-02-12:
#     -- Increased __version__ to 0.1.0.
#     -- First created coord_helpers.py.
#

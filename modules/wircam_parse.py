#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Filename parser for WIRCam data. This is used for name mangling and
# basic image name verification. This module needs to know about:
# * file names provided by CADC for download
# * naming conventions for images and their flavors in pyrallax
# * naming conventions for catalogs and their flavors in pyrallax
#
# Rob Siverd
# Created:       2023-07-24
# Last modified: 2023-07-24
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Modules:
#import shutil
#import glob
#import gc
import os
import sys
import time
#import vaex
#import calendar
#import ephem
#import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#import pandas as pd
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
#_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##
##------------------           Naming Conventions           ----------------##
##--------------------------------------------------------------------------##

# Raw WIRCam images from CADC have a variable-length format consisting of
# an exposure/sequence number, image type, and applicable suffixes. Examples:
# *  887816f.fits.fz        # an early flat
# * 2289531p.fits.fz        # one of the calib1 images (p = processed)

# Processed WIRCam images are regularized. These images have names like:
# wircam_<filter>_<expnum><type>.fits.fz  OR
# wircam_<filter>_<expnum><type>_<flavor>.fits.fz
# where:
# * <filter> is a variable-length tag (e.g., J, H2)
# * <expnum> is a (zero-padded ??) exposure number
# * <type> is the image type provided by CADC
# * <flavor> is a variable-length string indicating degree of processing

# NOTES:
# * RJS scoured CADC by observation date and determined that the earliest
#       data have obsid around 800k or 900k (like above). This means that
#       the shortest obsid is 6 digits long.
# * Fixed-length naming (zero-padding for obsid) can be enforced in the
#       download script for sanity.
# * Since numpy recarrays complain when the columns have mismatched length,
#       there is some value in forcing the <filter> in derived file names
#       to be a fixed length.

##--------------------------------------------------------------------------##
##------------------          WIRCam Name Parsing           ----------------##
##--------------------------------------------------------------------------##

## Base class that recognizes WIRCam image files:
class WIRCamParse(object):
    """File name parser & validator for CFHT WIRCam images."""

    def __init__(self):
        #self._raw_regex = ''
        return

##--------------------------------------------------------------------------##

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



######################################################################
# CHANGELOG (wircam_parse.py):
#---------------------------------------------------------------------
#
#  2023-07-24:
#     -- Increased __version__ to 0.1.0.
#     -- First created wircam_parse.py.
#

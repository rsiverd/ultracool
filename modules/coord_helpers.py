#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module provides routines to work with targets lists and image 
# coordinates for parallax pipeline image processing.
#
# Rob Siverd
# Created:       2021-02-12
# Last modified: 2021-02-25
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
__version__ = "0.3.0"

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

## RJS personal angular routines:
try:
    import angle
except ImportError:
    logger.error("module 'angle' not found!  Install and retry.")
    sys.stderr.write("\nError!  robust_stats module not found!\n"
           "Please install and try again ...\n\n")
    sys.exit(1)


## Various from astropy:
try:
    from astropy import coordinates as coord
    import astropy.wcs as awcs
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
        self.header = None
        self.ishape = None
        self.xlimit = None
        self.ylimit = None
        return

    def set_header(self, imhdr):
        #sys.stderr.write("--------------------------------------\n")
        self.header = imhdr.copy()
        nx = self.header['NAXIS1']
        ny = self.header['NAXIS2']
        self.ishape = (nx, ny)
        self.xlimit = (0.5, nx + 0.5)
        self.ylimit = (0.5, ny + 0.5)
        self.wcs = awcs.WCS(self.header)
        self._set_center_and_diagonal()
        #self._ctr_sep_max = 0.5 * self._diag_deg
        #sys.stderr.write("Adopted image center: %s\n" % str(self._ctr_radec))
        #sys.stderr.write("Maximum sep (degrees): %.4f\n" % self._ctr_sep_max)
        return

    def _set_center_and_diagonal(self):
        corner_xx = np.array([1, self.ishape[0]])
        corner_yy = np.array([1, self.ishape[1]])
        corner_ra, corner_de = self.wcs.all_pix2world(corner_xx, corner_yy, 1,
                                                        ra_dec_order=True)
        self._diag_deg = angle.dAngSep(corner_ra[0], corner_de[0],
                                        corner_ra[1], corner_de[1])
        #self._half_diag_deg = 0.5 * self._diag_deg

        # calculate mid-image RA, DE:
        center_xx = np.average(corner_xx)
        center_yy = np.average(corner_yy)
        self._ctr_radec = self.wcs.all_pix2world(center_xx, center_yy, 1)
        return
    
    # ---------------------------------------------
    # Full-frame position containment checks:
    # 
    # dfrac NOTE: center-edge distance is 0.5 / sqrt(2) =~ 0.35
    # Using dfrac >~ 0.35 allows the target to be off-image.

    def fdiag_covers_position_single(self, coord, dfrac=0.3):
        """Check whether coord is less than dfrac*diagonal degrees
        from the image center."""
        tra, tde = coord.ra.degree, coord.dec.degree
        #sys.stderr.write("Target RA: %8.4f\n" % tra)
        #sys.stderr.write("Target DE: %8.4f\n" % tde)
        ## large angular separation rules out coverage:
        ctr_sep_deg = angle.dAngSep(*self._ctr_radec, 
                            coord.ra.degree, coord.dec.degree)
        #sys.stderr.write("ctr_sep_deg: %.4f\n" % ctr_sep_deg)
        return (ctr_sep_deg < dfrac * self._diag_deg)

    def fdiag_covers_position_multi(self, coord_list, dfrac=0.3):
        return [self.fdiag_covers_position_single(tt, dfrac=dfrac) \
                            for tt in coord_list]

    def fdiag_covers_position_any(self, coord_list, dfrac=0.3):
        return any(self.fdiag_covers_position_multi(coord_list, dfrac=dfrac))

    # ---------------------------------------------
    # Full-frame position containment checks:

    def image_covers_position_single(self, coord):
        return coord.contained_by(self.wcs)

    def image_covers_position_multi(self, coord_list):
        return [self.image_covers_position_single(tt) for tt in coord_list]

    def image_covers_position_any(self, coord_list):
        return any(self.image_covers_position_multi(coord_list))



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


# should be contained:
# SPITZER_I1_57909248_0025_0000_1_cbcd.fits

# tt = targets[0]
# tra = tt.ra.degree
# tde = tt.dec.degree


######################################################################
# CHANGELOG (spitz_fstools.py):
#---------------------------------------------------------------------
#
#  2021-02-12:
#     -- Increased __version__ to 0.1.0.
#     -- First created coord_helpers.py.
#

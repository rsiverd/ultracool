#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract sources from Spitzer images for UCD project.
#
# Rob Siverd
# Created:       2019-10-15
# Last modified: 2019-10-15
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
import os
import sys
import time
import numpy as np
from numpy.lib.recfunctions import append_fields
#from functools import partial
#from collections import OrderedDict
from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##

## Various from astropy:
try:
#    import astropy.io.ascii as aia
    import astropy.io.fits as pf
#    import astropy.table as apt
#    import astropy.time as astt
    import astropy.wcs as awcs
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

## LACOSMIC cosmic ray removal:
try:
    from lacosmic import lacosmic
except ImportError:
    logger.error("failed to import lacosmic module!")
    sys.exit(1)

##--------------------------------------------------------------------------##
## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Star extraction:
try:
    import easy_sep
    reload(easy_sep)
except ImportError:
    logger.error("easy_sep module not found!  Install and retry.")
    sys.stderr.write("Error: easy_sep module not found!\n\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------    Spitzer Star Extraction Class       ----------------##
##--------------------------------------------------------------------------##

_lacos_defaults = {
                  'contrast'  :  12.0,
              'cr_threshold'  :   5.0,
        'neighbor_threshold'  :   4.0,
        }

_spitz_defaults = {
                'minpixels'   :     5,
               'pix_origin'   :   1.0,
                'win_sigma'   :   1.2,
                'calc_wpos'   :  True,
        }

class SpitzFind(object):

    def __init__(self):
        self._pse = easy_sep.EasySEP()
        self._pse.set_options_test(**_spitz_defaults)
        # science image:
        self._ipath = None
        self._idata = None
        self._ihdrs = None
        self._imwcs = None
        self._cdata = None      # optional cleaned image
        # uncertainty image:
        self._upath = None
        self._udata = None
        self._uhdrs = None
        self._have_err_image = False
        return

    def set_pse_options(self, **kwargs):
        return self._pse.set_options_test(**kwargs)
        #return self._pse.set_options(**kwargs)

    # ----------------------------------------

    def use_images(self, ipath=None, upath=None):
        """Load images for analysis. Inputs:
        ipath   --  path to image for analysis
        upath   --  path to uncertainty image
        """
        # data image:
        if ipath:
            logger.info("Loading data image %s" % ipath)
            try:
                self._idata, self._ihdrs = self._get_data_and_header(ipath)
                self._pse.set_image(self._idata, _docopy=False)
                self._pse.set_mask(np.isnan(self._idata))
                self._imwcs = awcs.WCS(self._ihdrs)
                self._pse.set_imwcs(self._imwcs.all_pix2world)
                self._ipath = ipath
            except:
                logger.error("Failed to load file: %s" % ipath)
                self._ipath, self._idata, self._ihdrs = None, None, None

        # error image:
        if upath:
            logger.info("Loading error image %s" % upath)
            try:
                self._udata, self._uhdrs = self._get_data_and_header(upath)
                self._have_err_image = True
                self._pse.set_errs(self._udata, _docopy=False)
                self._upath = upath
            except:
                logger.error("Failed to load file: %s" % ipath)
                self._upath, self._udata, self._uhdrs = None, None, None
                self._have_err_image = False
        return


    @staticmethod
    def _get_data_and_header(filename):
        rdata, rhdrs = pf.getdata(filename, header=True)
        return rdata.astype('float32'), rhdrs.copy(strip=True)

    # ----------------------------------------
    def find_stars(self, thresh, keepall=False, use_err_img=True):
        """Driver routine for star extraction. Required inputs:
        thresh      --  significance threshold for star extraction
        use_err_img --  False disables use of error-image

        Results are reported in an ExtendedCatalog container.
        """
        _err_mode = use_err_img and self._have_err_image
        self._pse.analyze(thresh, rel_err=_err_mode)
        dataset = self._pse.allobjs if keepall else self._pse.useobjs

        result = ec.ExtendedCatalog(data=dataset,
                name=os.path.basename(self._ipath), header=self._ihdrs,
                uname=os.path.basename(self._upath), uheader=self._uhdrs)
        return result

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (spitz_extract.py):
#---------------------------------------------------------------------
#
#  2019-10-15:
#     -- Increased __version__ to 0.1.0.
#     -- First created spitz_extract.py.
#

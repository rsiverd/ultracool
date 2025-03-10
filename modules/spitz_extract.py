#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract sources from Spitzer images for UCD project.
#
# Rob Siverd
# Created:       2019-10-15
# Last modified: 2023-01-19
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

## Spitzer pixel phase correction (cheesy version, IRAC ch1):
try:
    import spitz_pixphase
    reload(spitz_pixphase)
    iracfix = spitz_pixphase.IRACFix()
except ImportError:
    logger.error("failed to import spitz_pixphase module!")
    sys.exit(1)

## Adam Kraus polynomial routines:
try:
    import akspoly
    reload(akspoly)
    akp = akspoly.AKSPoly()
except ImportError:
    logger.error("failed to import akspoly module!")
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
               'pix_origin'   :     1,  # MUST BE INTEGER
                'win_sigma'   :   1.2,
                'calc_wpos'   :  True,
        }

class SpitzFind(object):

    def __init__(self):
        self._pse = easy_sep.EasySEP()
        self._pse.set_options_test(**_spitz_defaults)
        # science image:
        self._reset_iparams()
        # cosmic ray cleaning:
        self._cdata = None
        self._cmask = None
        # uncertainty image:
        self._reset_uparams()
        # miscellany:
        self._confirmed = False
        return

    # Re-initialize image values:
    def _reset_iparams(self):
        self._ipath = None
        self._idata = None
        self._ihdrs = None
        self._imwcs = None
        self._imask = None
        return

    # Re-initialize error-image values:
    def _reset_uparams(self):
        self._upath = None
        self._udata = None
        self._uhdrs = None
        self._have_err_image = False
        return

    # Reset everything:
    def _reset_everything(self):
        self._reset_iparams()
        self._reset_uparams()
        return
 
    # ----------------------------------------

    def set_pse_options(self, **kwargs):
        return self._pse.set_options_test(**kwargs)
        #return self._pse.set_options(**kwargs)

    # ----------------------------------------

    def use_images(self, ipath=None, upath=None):
        """Load images for analysis. Inputs:
        ipath   --  path to image for analysis
        upath   --  path to uncertainty image
        """
        #self._confirmed = False
        self._reset_everything()

        # data image:
        if ipath:
            logger.info("Loading data image %s" % ipath)
            try:
                self._idata, self._ihdrs = self._get_data_and_header(ipath)
                self._imask = np.isnan(self._idata)
                self._imwcs = awcs.WCS(self._ihdrs)
                self._ipath = ipath
                self._pse.set_image(self._idata, _docopy=False)
                self._pse.set_mask(self._imask)
                self._pse.set_imwcs(self._imwcs.all_pix2world)
            except:
                logger.error("Failed to load file: %s" % ipath)
                #self._ipath, self._idata, self._ihdrs = None, None, None
                self._reset_iparams()

        # error image:
        if upath:
            logger.info("Loading error image %s" % upath)
            try:
                self._udata, self._uhdrs = self._get_data_and_header(upath)
                self._upath = upath
                self._have_err_image = True
                self._pse.set_errs(self._udata, _docopy=False)
            except:
                logger.error("Failed to load file: %s" % ipath)
                #self._upath, self._udata, self._uhdrs = None, None, None
                #self._have_err_image = False
                self._reset_uparams()
        return

    ## remove cosmic rays:
    #def remove_cosmics(self):
    #    sys.stderr.write("Removing cosmic rays ... ")
    #    lakw = {}
    #    lakw.update(_lacos_defaults)
    #    lakw['mask'] = self._imask
    #    if self._have_err_image:
    #        lakw['error'] = self._udata
    #    self._cdata, self._cmask = lacosmic(self._idata, **lakw)
    #    self._pse.img_data = self._cdata    # swap in 
    #    #self._pse.set_image(self._cdata, _docopy=False)
    #    sys.stderr.write("done.\n")
    #    return

    # confirm choices to prepare for PSE:
    #def confirm(self):
    #    self._pse.set_image(self._idata, _docopy=False)
    #    self._pse.set_mask(self._imask)
    #    self._pse.set_imwcs(self._imwcs.all_pix2world)
    #    if self._have_err_image:
    #        self._pse.set_errs(self._udata, _docopy=False)

    @staticmethod
    def _get_data_and_header(filename):
        rdata, rhdrs = pf.getdata(filename, header=True)
        return rdata.astype('float32'), rhdrs.copy(strip=True)

    # ----------------------------------------
#    def _ak_dephase_dewarp(self, dataset, mission, aks_inst):
#        x_dephase, y_dephase = akp.dephase(np.atleast_1d(dataset['x']),
#                                        np.atleast_1d(dataset['y']), mission, aks_inst)
#        x_dewarp, y_dewarp = akp.xform_xy(x_dephase, y_dephase, aks_inst)
#        dataset = append_fields(dataset, ('xdp', 'ydp', 'xdw', 'ydw'),
#                (x_dephase, y_dephase, x_dewarp, y_dewarp), usemask=False)
#
#        return True

    def _akp_added_value(self, dataset):
        channel  = self._ihdrs['CHNLNUM']        # SHA provides this in FITS header
        jdtdb    = self._ihdrs['OBS_TIME']       # mid-exposure JDTDB, added to headers previously
        mission  = akspoly.mission_from_jdtdb(jdtdb)
        aks_inst = akspoly._chmap[channel]

        # first update adds dephased and dewarped pixel coordinates
        #success  = self._ak_dephase_dewarp(dataset, mission, aks_inst)
        x_dephase, y_dephase = akp.dephase(np.atleast_1d(dataset['x']),
                                        np.atleast_1d(dataset['y']), mission, aks_inst)
        x_dewarp, y_dewarp = akp.xform_xy(x_dephase, y_dephase, aks_inst)
        dataset = append_fields(dataset, ('xdp', 'ydp', 'xdw', 'ydw'),
                (x_dephase, y_dephase, x_dewarp, y_dewarp), usemask=False)

        # second update transforms to RA/DE using PA, CRVAL1, CRVAL2:
        this_padeg = self._ihdrs['PA']
        this_crv_1 = self._ihdrs['CRVAL1']
        this_crv_2 = self._ihdrs['CRVAL2']
        aks_ra, aks_de = akspoly.xypa2radec(this_padeg, x_dewarp, y_dewarp,
                this_crv_1, this_crv_2, aks_inst)
        dataset = append_fields(dataset, ('akpara', 'akpade'),
                (aks_ra, aks_de), usemask=False)

        #return True
        return dataset

    # ----------------------------------------
    def find_stars(self, thresh, keepall=False, use_err_img=True, include_akp=False):
        """Driver routine for star extraction. Required inputs:
        thresh      --  significance threshold for star extraction
        keepall     --  keep all detections (skip pruning)
        use_err_img --  False disables use of error-image
        include_akp --  if True, add Adam Kraus polynomial columns (requires metadata)

        Results are reported in an ExtendedCatalog container.
        """
        _err_mode = use_err_img and self._have_err_image
        self._pse.analyze(thresh, rel_err=_err_mode)
        dataset = self._pse.allobjs if keepall else self._pse.useobjs

        # tack on pixel-phase corrected coordinates:
        pix_origin = self._pse.settings['pix_origin']
        ppx, ppy = iracfix.fix_centroid(dataset['x'], dataset['y'])
        ppra, ppde = self._pse._wcs_func(ppx, ppy, pix_origin)
        dataset = append_fields(dataset, ('ppx', 'ppy', 'ppdra', 'ppdde'),
                (ppx, ppy, ppra, ppde), usemask=False)

        # Adam Kraus polynomial dephase/dewarp:
        if include_akp:
            #success = self._akp_added_value(dataset)
            dataset = self._akp_added_value(dataset)

        #channel  = self._ihdrs['CHNLNUM']        # SHA provides this in FITS header
        #jdtdb    = self._ihdrs['OBS_TIME']       # mid-exposure JDTDB, added to headers previously
        #mission  = akspoly.mission_from_jdtdb(jdtdb)
        #aks_inst = akspoly._chmap[channel]
        #x_dephase, y_dephase = akp.dephase(np.atleast_1d(dataset['x']),
        #                                np.atleast_1d(dataset['y']), mission, aks_inst)
        #x_dewarp, y_dewarp = akp.xform_xy(x_dephase, y_dephase, aks_inst)
        #dataset = append_fields(dataset, ('xdp', 'ydp', 'xdw', 'ydw'),
        #        (x_dephase, y_dephase, x_dewarp, y_dewarp), usemask=False)

        ## sky coords from Adam Kraus polynomials:
        #this_padeg = self._ihdrs['PA']
        #this_crv_1 = self._ihdrs['CRVAL1']
        #this_crv_2 = self._ihdrs['CRVAL2']
        #aks_ra, aks_de = akspoly.xy2radec(this_padeg, x_dewarp, y_dewarp,
        #        this_crv_1, this_crv_2, aks_inst)
        #dataset = append_fields(dataset, ('akra', 'akde'),
        #        (aks_ra, aks_de), usemask=False)

        # encapsulate result:
        ecopts = {'name':os.path.basename(self._ipath), 'header':self._ihdrs}
        if self._have_err_image:
            ecopts['uname']   = os.path.basename(self._upath)
            ecopts['uheader'] = self._uhdrs
        result = ec.ExtendedCatalog(data=dataset, **ecopts)
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

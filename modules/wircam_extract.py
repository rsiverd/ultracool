#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Extract sources from Spitzer images for UCD project.
#
# Rob Siverd
# Created:       2023-07-06
# Last modified: 2024-09-23
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
__version__ = "0.1.1"

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

### LACOSMIC cosmic ray removal:
#try:
#    from lacosmic import lacosmic
#except ImportError:
#    logger.error("failed to import lacosmic module!")
#    sys.exit(1)

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

### Spitzer pixel phase correction (cheesy version, IRAC ch1):
#try:
#    import spitz_pixphase
#    reload(spitz_pixphase)
#    iracfix = spitz_pixphase.IRACFix()
#except ImportError:
#    logger.error("failed to import spitz_pixphase module!")
#    sys.exit(1)

## WIRCam polynomial routines:
try:
    import wircam_poly
    reload(wircam_poly)
    wcp = wircam_poly.WIRCamPoly()
except ImportError:
    logger.error("failed to import wircam_poly module!")
    sys.exit(1)

##--------------------------------------------------------------------------##
##------------------   CFHT WIRCam Star Extraction Class    ----------------##
##--------------------------------------------------------------------------##

#_lacos_defaults = {
#                  'contrast'  :  12.0,
#              'cr_threshold'  :   5.0,
#        'neighbor_threshold'  :   4.0,
#        }

_wircam_defaults = {
                'minpixels'   :     5,
               'pix_origin'   :     1,  # MUST BE INTEGER
                'win_sigma'   :   2.0,
                'calc_wpos'   :  True,
        }

class WIRCamFind(object):

    def __init__(self):
        self._pse = easy_sep.EasySEP()
        self._pse.set_options_test(**_wircam_defaults)
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

    def use_images(self, ipath=None, upath=None, layer=0):
        """Load images for analysis. Inputs:
        ipath   --  path to image for analysis
        upath   --  path to uncertainty image
        layer   --  which layer to use in case of 3-D cube
        """
        #self._confirmed = False
        self._reset_everything()

        # data image:
        if ipath:
            logger.info("Loading data image %s" % ipath)
            try:
                self._idata, self._ihdrs = self._get_data_and_header(ipath)
                ndims = len(self._idata.shape)
                if (ndims != 2):
                    logger.warning("Not a 2-D image! Image shape: %s" % str(self._idata.shape))
                    logger.warning("Keeping layer %d and discarding the rest." % layer)
                    # for now, just take top layer:
                    #sys.stderr.write("before\n")
                    self._idata = self._idata[layer]
                    #sys.stderr.write("after\n")
                self._imask = np.isnan(self._idata)
                self._imwcs = awcs.WCS(self._ihdrs)
                self._ipath = ipath
                self._pse.set_image(self._idata, _docopy=False)
                self._pse.set_mask(self._imask)
                self._pse.set_imwcs(self._imwcs.all_pix2world)
                ndims = len(self._idata.shape)

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

    @staticmethod
    def _get_data_and_header(filename):
        rdata, rhdrs = pf.getdata(filename, header=True)
        return rdata.astype('float32'), rhdrs.copy(strip=True)

    ## include distortion corrected pixel positions:
    #def _wir_added_value(self, dataset, wparams):
    #    filt = self._ihdrs['FILTER']
    #    #crval1 = self._ihdrs['CRVAL1']
    #    #crval2 = self._ihdrs['CRVAL2']
    #    cdmat  = wparams[:4]
    #    crval1 = wparams[4]
    #    crval2 = wparams[5]

    #    # adjusted standard positions:
    #    tx = np.atleast_1d(dataset['x'])
    #    ty = np.atleast_1d(dataset['y'])
    #    sx_dewarp, sy_dewarp = wcp.calc_corrected_xy(tx, ty, filt) 
    #    pred_ra, pred_de = wcp.predicted_radec(None, tx, ty, crval1, crval2)
    #    dataset = append_fields(dataset, ('xdw', 'ydw', 'dradw', 'ddedw'),
    #            (sx_dewarp, sy_dewarp, pred_ra, pred_de), usemask=False)

    #    # adjusted windowed positions:
    #    tx = np.atleast_1d(dataset['wx'])
    #    ty = np.atleast_1d(dataset['wy'])
    #    wx_dewarp, wy_dewarp = wcp.calc_corrected_xy(tx, ty, filt)
    #    pred_ra, pred_de = wcp.predicted_radec(None, tx, ty, crval1, crval2)
    #    dataset = append_fields(dataset, ('wxdw', 'wydw', 'wdradw', 'wddedw'),
    #            (wx_dewarp, wy_dewarp, pred_ra, pred_de), usemask=False)

    #    # also try calculating 
    #    return dataset

    def _wir_distortion_correction(self, dataset):
        # the models we know about:
        _have_models = wcp.get_dist_models()

        # adjust non-windowed positions:
        tx = np.atleast_1d(dataset['x'])
        ty = np.atleast_1d(dataset['y'])
        for model in _have_models:
            xnudge, ynudge = wcp.calc_xy_nudges(tx, ty, model)
            xcorr = tx + xnudge
            ycorr = ty + ynudge
            xname = 'xdw_%s' % model
            yname = 'ydw_%s' % model
            dataset = append_fields(dataset, (xname, yname),
                    (xcorr, ycorr), usemask=False)
            pass

        # adjust windowed positions:
        tx = np.atleast_1d(dataset['wx'])
        ty = np.atleast_1d(dataset['wy'])
        for model in _have_models:
            xnudge, ynudge = wcp.calc_xy_nudges(tx, ty, model)
            xcorr = tx + xnudge
            ycorr = ty + ynudge
            xname = 'wxdw_%s' % model
            yname = 'wydw_%s' % model
            dataset = append_fields(dataset, (xname, yname),
                    (xcorr, ycorr), usemask=False)
            pass

        # return updated catalog:
        return dataset

    # a more sophisticated "hot row" detector/corrector:
    def _prune_hot_rows(self, dataset, thresh):
        cleaned  = np.copy(dataset)
        rpk_pop  = np.bincount(dataset['ypeak'])
        row_num  = np.arange(rpk_pop.size)
        bad_rows = row_num[(rpk_pop >= thresh)]
        for rr in bad_rows:
            drop_me = (cleaned['ypeak'] == rr)
            cleaned = cleaned[~drop_me]
        return cleaned

    # append additional identifying information to the dataset such as image
    # name, filter, instrument tag, and detection index:
    def _append_instrument_tag(self, dataset):
        filt = self._ihdrs['FILTER']
        itag = 'wircam_%s' % filt
        nsrc = len(dataset)
        dindx = np.arange(nsrc)                     # detection index
        iname = os.path.basename(self._ipath)       # image base name
        dataset = append_fields(dataset, ('filter', 'instrument', 'iname'),
                (nsrc*[filt], nsrc*[itag], nsrc*[iname]), usemask=False)
        return dataset

    # ----------------------------------------
    def find_stars(self, thresh, keepall=False, use_err_img=True, 
            prune_horiz=True, include_poly=False, **fskw):
        """Driver routine for star extraction. Required inputs:
        thresh       --  significance threshold for star extraction
        keepall      --  keep all detections (skip pruning)
        use_err_img  --  False disables use of error-image
        prune_horiz  --  eliminate objects spanning 1 pixel in Y (hot rows)
        include_poly --  include polynomial distortion correction(s)

        Results are reported in an ExtendedCatalog container.
        """
        _err_mode = use_err_img and self._have_err_image
        self._pse.analyze(thresh, rel_err=_err_mode)
        dataset = self._pse.allobjs if keepall else self._pse.useobjs

        # create instrument tag for this data set:
        dataset = self._append_instrument_tag(dataset)

        # snip out single-row detections:
        if prune_horiz:
            n_dirty = len(dataset)
            dataset = self._prune_hot_rows(dataset, thresh=10)
            n_clean = len(dataset)
            #horiz   = dataset['ymin'] == dataset['ymax']
            #dataset = dataset[~horiz]

        # tack on pixel-phase corrected coordinates:
        pix_origin = self._pse.settings['pix_origin']
        #ppx, ppy = iracfix.fix_centroid(dataset['x'], dataset['y'])
        #ppra, ppde = self._pse._wcs_func(ppx, ppy, pix_origin)
        #dataset = append_fields(dataset, ('ppx', 'ppy', 'ppdra', 'ppdde'),
        #        (ppx, ppy, ppra, ppde), usemask=False)

        # distortion correction:
        if include_poly:
            dataset = self._wir_distortion_correction(dataset)
        #if include_poly:
        #    wparams = fskw.pop('wpars')
        #    sys.stderr.write("Got wparams: %s\n" % str(wparams))
        #    dataset = self._wir_added_value(dataset, wparams)

        ## Adam Kraus polynomial dephase/dewarp:
        #if include_akp:
        #    #success = self._akp_added_value(dataset)
        #    dataset = self._akp_added_value(dataset)

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
# CHANGELOG (wircam_extract.py):
#---------------------------------------------------------------------
#
#  2023-07-06:
#     -- Increased __version__ to 0.1.0.
#     -- First created wircam_extract.py.
#

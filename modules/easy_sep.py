#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
#    Quick star extraction with SEP.
#
# Rob Siverd
# Created:       2018-02-19
# Last modified: 2020-09-26
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
__version__ = "0.2.3"

## Modules:
import os
import sys
import time
import numpy as np
from numpy.lib.recfunctions import append_fields
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Star extraction:
try:
    import sep
except ImportError:
    #sys.stderr.write("Error: sep module not installed!\n\n")
    logger.error("sep module not installed!\n\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
## Robust location/scale estimate using median/IQR:
def calc_ls_med_IQR(a, axis=None):
    """Return median and inter-quartile range of *a* (scaled to normal)."""
    pctiles = np.percentile(a, [25, 50, 75], axis=axis)
    med_val = pctiles[1]
    sig_hat = (0.741301109 * (pctiles[2] - pctiles[0]))
    return (med_val, sig_hat)

##--------------------------------------------------------------------------##
## 5x5 kernel:
_five_by_five = np.array([[1., 2., 3., 2., 1.],
                          [2., 3., 5., 3., 2.],
                          [3., 5., 8., 5., 3.],
                          [2., 3., 5., 3., 2.],
                          [1., 2., 3., 2., 1.]])


##--------------------------------------------------------------------------##
## Top hat kernel of arbitrary shape:
def make_tophat(shape):
    tophat = np.ones(shape).astype('float32')
    return (tophat / np.sum(tophat))

##--------------------------------------------------------------------------##
## Flux <---> mag conversion:
def flux2mag(flux, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(flux))

def mag2flux(mag, zeropt=25.0):
    return 10.0**(0.4 * (zeropt - mag))

##--------------------------------------------------------------------------##
## Settings I commonly use:
_default_settings = {
       #'sigthresh'     :        75.0,
        'pix_origin'    :           0,
        'minpixels'     :          10,
        'gain'          :         1.0, 
        'calc_wpos'     :       False,
        'win_sigma'     :        None,
        'have_wcs'      :       False,
        'have_errs'     :       False,
        }

_required_settings = _default_settings.keys()

## Which data columns get augmented by origin:
_pixpos_columns = ('xmin', 'x', 'xmax', 'xpeak', 'xcpeak', 'wx',
                   'ymin', 'y', 'ymax', 'ypeak', 'ycpeak', 'wy',)

##--------------------------------------------------------------------------##
## Simplified star extraction class:
class EasySEP(object):

    def __init__(self):
        self.settings  = _default_settings
        #self.thresh    = None
        #self.minpixels = minpixels
        #self.gain      = gain
        self.minpixels  = 10
        self.gain       = None
        self._reset_image()
        self._reset_objects()
        return

    # ---------------------------------------------
    # Housekeeping routines:
    def _reset_image(self):
        self.img_data   = None
        self.err_data   = None
        self.bkg_data   = None
        self.msk_data   = None
        self.sub_data   = None
        self.im_stats   = {}
        #self.kernel     = _five_by_five
        self.kernel     = None
        self._wcs_func  = None
        return

    def _reset_objects(self):
        self.useobjs   = None
        self.badobjs   = None
        return

    def _check_settings(self):
        nmissing = 0
        present = list(set(_required_settings) & set(self.settings.keys()))
        missing = list(set(_required_settings) ^ set(self.settings.keys()))
        #present = [x for x in _required_settings if x in self.settings.keys()]
        #    if not (item in self.settings.keys()):
        #        sys.stderr.write("Missing required setting: %s\n" % item)
        #        nmissing += 1
        return True # FIXME

    # ---------------------------------------------
    # High-level driver routines:
    # ---------------------------------------------

    # Switch to new image:
    def set_image(self, image, gain=None, _docopy=True):
        self._reset_image()
        self._reset_objects()
        self.settings['gain'] = gain if gain else None
        self.img_data = image.astype('float32') if _docopy else image
        self.im_stats = self._calc_imstats(self.img_data)
        return True

    # Set error/uncertainty image:
    def set_errs(self, image, _docopy=True):
        #self.err_data = np.copy(err_image) if _docopy else err_image
        self.err_data = image.astype('float32') if _docopy else image
        #self._have_errs = True
        self.settings['have_errs'] = True
        return

    # Set bad pixel mask:
    def set_mask(self, mask):
        self.msk_data = np.copy(mask)
        return

    # Provide WCS conversion object:
    def set_imwcs(self, imwcs):
        """Specify an astropy WCS coordinate conversion method. This will
        be used after object detection to provide RA/Dec values in degrees
        ('dra' and 'dde') alongside the standard outputs of SEP.

        Use one of these (or something with matching arguments):
        --> all_pix2world       (does everything)
        --> wcs_pix2world       (ignores SIP)
        """
        self._wcs_func = imwcs
        self.settings['have_wcs'] = True
        return

    # Enabled windowed positions by setting winsig:
    def enable_winpos(self, winsig):
        self.settings['win_sigma'] = winsig
        self.settings['calc_wpos'] = True
        return

    # Disable windowed positions:
    def disable_winpos(self):
        self.settings['win_sigma'] = None
        self.settings['calc_wpos'] = False
        return

    ## Set pixel origin (usually 0 or 1):
    #def set_origin(self, offset):
    #    self.settings['pix_origin'] = offset
    #    return

    # Change settings:
    def set_options(self, gain=None, minpixels=None, pix_origin=None):
        if (gain != None):
            self.settings['gain'] = gain
        if (minpixels != None):
            self.settings['minpixels'] = minpixels
        if (pix_origin != None):
            if not isinstance(pix_origin, int):
                logger.warning("pix_origin (%f) truncated to integer!\n"
                        % pix_origin)
            self.settings['pix_origin'] = int(pix_origin)
        #if (pix_origin != None):
        #    self.settings['pix_origin'] = minpixels
        return

    def set_options_test(self, **kwargs):
        # FIXME: should validate settings before update ...
        self.settings.update(**kwargs)
        return

    # Main driver routine (use this):
    def analyze(self, sigthresh, rel_err=False):
        if not self._check_settings():
            return None
        if rel_err and not self.settings['have_errs']:
            logger.error("No error image set (needed for rel_err=True)!\n")
            return None
        if not isinstance(self.bkg_data, np.ndarray):
            self.bkg_data  = self._estimate_background()
        if not isinstance(self.sub_data, np.ndarray):
            self.sub_data = self.img_data - self.bkg_data
        if not isinstance(self.useobjs, np.ndarray):
            self._extract_stars(sigthresh, rel_err)
        return self.useobjs

    # ---------------------------------------------
    # Workhorse routines:
    # ---------------------------------------------
    
    # Extract stars:
    def _extract_stars(self, sig_thresh, rel_err_mode):
        kwargs = {'gain':self.settings['gain'], 'filter_kernel':self.kernel}
        if rel_err_mode:
            use_thresh = sig_thresh
            kwargs['err'] = self.err_data
        else:
            use_thresh = self._calc_sig_threshold(sig_thresh)   # in ADU

        # perform extraction:
        use_image = self.sub_data     # which image data to analyze
        cat = sep.extract(use_image, use_thresh, **kwargs)
                #gain=self.settings['gain'], filter_kernel=self.kernel)

        # calc/append windowed positions if needed:
        if self.settings['calc_wpos']:
            xyf = sep.winpos(use_image, cat['x'], cat['y'],
                                sig=self.settings['win_sigma'])
            cat = append_fields(cat, ('wx', 'wy', 'wflag'), xyf, usemask=False)

        # UNWINDOWED coordinates from WCS (if possible):
        if self.settings['have_wcs']:
            radec = self._wcs_func(cat['x'], cat['y'], 0.0)
            cat = append_fields(cat, ('dra', 'dde'), radec, usemask=False)
        
        # WINDOWED coordinates from WCS (if possible):
        if self.settings['calc_wpos'] and self.settings['have_wcs']:
            radec = self._wcs_func(cat['wx'], cat['wy'], 0.0)
            cat = append_fields(cat, ('wdra', 'wdde'), radec, usemask=False)

        # impose coordinate system origin:
        for cc in _pixpos_columns:
            if cc in cat.dtype.names:
                cat[cc] += self.settings['pix_origin']

        # sort into reverse-flux order
        flx_ord = np.argsort(cat['flux'])[::-1]
        cat = cat[flx_ord]

        # select keepers using detection size:
        keepers = (cat['tnpix'] >= self.settings['minpixels'])
        self.useobjs = cat[keepers]
        self.badobjs = cat[~keepers]
        self.allobjs = cat
        return

    # Background estimation:
    def _estimate_background(self):
        idata, mask = self.img_data, self.msk_data
        bkg_128 = sep.Background(idata, mask=mask, bw=128, bh=128)
        bkg_064 = sep.Background(idata, mask=mask, bw=64, bh=64)
        bkg_use = 0.5 * (bkg_064.back() + bkg_128.back())
        return bkg_use

    # Noise-based threshold:
    def _calc_sig_threshold(self, sigmas):
        return sigmas * self.im_stats['iqrdev']

    # Image stats:
    @staticmethod
    def _calc_imstats(idata):
        iclean = idata[~np.isnan(idata)]
        pix_med, pix_iqrdev = calc_ls_med_IQR(iclean)
        pix_avg, pix_stddev = np.average(iclean), np.std(iclean)
        return {'med':pix_med, 'iqrdev':pix_iqrdev,
                'avg':pix_avg, 'stddev':pix_stddev,}

##--------------------------------------------------------------------------##






######################################################################
# CHANGELOG (easy_sep.py):
#---------------------------------------------------------------------
#
#  2020-09-26:
#     -- Increased __version__ to 0.2.3.
#     -- Fixed key typo in default settings: calc_winpos --> calc_wpos. This
#           prevented easy_sep from executing with default settings.
#
#  2020-02-12:
#     -- Increased __version__ to 0.2.2.
#     -- Now use logging module for messages instead of sys.stderr.
#     -- pix_offset setting is now required to be integer type so that it
#           can be added to integer pixel positions (e.g., xpeak) without
#           violating 'same_kind' casting rules. Added a warning to user
#           about truncation whenever a non-float value is provided.
#
#  2019-11-11:
#     -- Increased __version__ to 0.2.1.
#     -- Added xmin, xmax, xpeak, xcpeak and y-coordinate analogs to list of
#           pixel positions for post-extraction nudging.
#
#  2019-10-29:
#     -- Increased __version__ to 0.2.0.
#     -- Moved _win_sigma, _calc_wpos, _have_errs, _have_wcs, and related
#           procedural constants into settings dictionary.
#     -- Default kernel is now None (filtering changes SNR behavior!).
#     -- Can now calculate windowed positional parameters in analyze().
#
#  2019-10-15:
#     -- Increased __version__ to 0.1.7.
#     -- Now note numpy version in module.
#     -- Can now provide an initialized WCS object. When provided, results
#           will include dra/dde celestial coordinates in decimal degrees.
#
#  2019-10-13:
#     -- Increased __version__ to 0.1.6.
#     -- Can now set pix_origin with set_options(). x,y in output table are
#           updated correspondingly.
#
#  2019-10-11:
#     -- Increased __version__ to 0.1.5.
#     -- Gain no longer required.
#     -- Added support for an error image.
#
#  2019-09-11:
#     -- Increased __version__ to 0.1.4.
#     -- NaN values are now avoided in _calc_imstats().
#
#  2018-03-02:
#     -- Increased __version__ to 0.1.3.
#     -- Added make_tophat() method.
#
#  2018-02-26:
#     -- Increased __version__ to 0.1.2.
#     -- Now store allobjs in case external access is needed.
#     -- Added set_mask() method.
#
#  2018-02-20:
#     -- Increased __version__ to 0.1.1.
#     -- SEP import is now wrapped in a try with sensible error message.
#
#  2018-02-19:
#     -- Increased __version__ to 0.1.0.
#     -- First created easy_sep.py.
#

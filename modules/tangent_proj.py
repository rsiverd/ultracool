#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Generalized tangent projection helper module.
#
# Rob Siverd
# Created:       2023-06-02
# Last modified: 2023-06-02
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
__version__ = "0.0.1"

## Modules:
#import argparse
#import shutil
import os
import sys
import time
#import vaex
#import calendar
#import ephem
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import fov_rotation
reload(fov_rotation)
rfov = fov_rotation.RotateFOV()

## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

## Matrix printer:
def mprint(matrix):
    for row in matrix:
        sys.stderr.write("  %s\n" % str(row))
    return

## Reflection matrices:
xref_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
yref_mat = np.array([[-1.0, 0.0], [0.0, 1.0]])
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array([[1.0, 0.0], [0.0, 1.0]])

## Radian-to-degree converter:
_radeg = 180.0 / np.pi

## Tangent projection:
def _tanproj(prj_xx, prj_yy):
    prj_rr = np.hypot(prj_xx, prj_yy)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    #prj_rr = np.sqrt(prj_xx**2 + prj_yy**2)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    useful = (prj_rr > 0.0)
    prj_theta = np.ones_like(prj_xx) * np.pi * 0.5
    prj_theta[useful] = np.arctan(np.degrees(1.0 / prj_rr[useful]))
    #prj_theta[useful] = np.arctan(_radeg / prj_rr[useful])
    #prj_phi = np.arctan2(prj_xx, prj_yy)
    prj_phi = np.arctan2(prj_xx, -prj_yy)
    #return prj_phi, prj_theta
    return np.degrees(prj_phi), np.degrees(prj_theta)

## Low-level WCS tangent processor:
def _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=False):
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    if debug:
        sys.stderr.write("%.3f < prj_xx < %.3f\n" % (prj_xx.min(), prj_xx.max()))
        sys.stderr.write("%.3f < prj_yy < %.3f\n" % (prj_yy.min(), prj_yy.max()))

    # Perform tangent projection:
    prj_phi, prj_theta = _tanproj(prj_xx, prj_yy)
    if debug:
        sys.stderr.write("%.3f < prj_theta < %.3f\n"
                % (prj_theta.min(), prj_theta.max()))
        sys.stderr.write("%.3f < prj_phi   < %.3f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Change variable names to avoid confusion:
    rel_ra, rel_de = prj_phi, prj_theta
    if debug:
        phi_range = prj_phi.max() - prj_phi.min()
        sys.stderr.write("phi range: %.4f < phi < %.4f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Shift to 
    old_fov = (0.0, 90.0, 0.0)
    new_fov = (crval1, crval2, 0.0)
    stuff = rfov.migrate_fov_deg(old_fov, new_fov, (rel_ra, rel_de))
    return stuff

## Convert X,Y to RA, Dec using CD matrix and CRVAL pair:
#def xycd2radec(cdmat, xpix, ypix, crval1, crval2, debug=False):
def xycd2radec(cdmat, rel_xx, rel_yy, crval1, crval2, debug=False):
    thisCD = np.array(cdmat).reshape(2, 2)
    #rel_xx = xpix - _aks_crpix1
    #rel_yy = ypix - _aks_crpix2
    relpix = np.array([xpix - _aks_crpix1, ypix - _aks_crpix2])
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)

## Convert X,Y to RA, Dec (single-value), uses position angle and fixed pixel scale:
#def xypa2radec(pa_deg, xpix, ypix, crval1, crval2, channel, debug=False):
#def xypa2radec(pa_deg, rel_xx, rel_yy, crval1, crval2, channel, debug=False):
#    pa_rad = np.radians(pa_deg)
#    #rel_xx = xpix - _aks_crpix1
#    #rel_yy = ypix - _aks_crpix2
#    relpix = np.array([xpix - _aks_crpix1, ypix - _aks_crpix2])
#    #pscale = _pxscale[channel]
#    #sys.stderr.write("pscale: %.4f\n" % pscale)
#    #rotmat = rotation_matrix(pa_rad)
#    ##sys.stderr.write("rotmat:\n")
#    ##mprint(rotmat)
#    #rscale = (pscale / 3600.0) * rotmat
#    #sys.stderr.write("rscale:\n")
#    #mprint(rscale)
#    thisCD = np.matmul(xflip_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
#    #thisCD = np.dot(ident_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
#    #thisCD = np.matmul(ident_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
#    if debug:
#        sys.stderr.write("thisCD:\n")
#        mprint(thisCD)
#    #rel_ra, rel_de = np.dot(thisCD, relpix)
#
#    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)


## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#    warnings.simplefilter('error')    # halt on warnings
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2


##--------------------------------------------------------------------------##
## Projections with cartopy:
#try:
#    import cartopy.crs as ccrs
#    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#    from cartopy.feature.nightshade import Nightshade
#    #from cartopy import config as cartoconfig
#except ImportError:
#    sys.stderr.write("Error: cartopy module not found!\n")
#    sys.exit(1)

##--------------------------------------------------------------------------##
## Disable buffering on stdout/stderr:
#class Unbuffered(object):
#   def __init__(self, stream):
#       self.stream = stream
#   def write(self, data):
#       self.stream.write(data)
#       self.stream.flush()
#   def __getattr__(self, attr):
#       return getattr(self.stream, attr)
#
#sys.stdout = Unbuffered(sys.stdout)
#sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##

unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
if (resource.getrlimit(resource.RLIMIT_DATA) == unlimited):
    resource.setrlimit(resource.RLIMIT_DATA,  (3e9, 6e9))
if (resource.getrlimit(resource.RLIMIT_AS) == unlimited):
    resource.setrlimit(resource.RLIMIT_AS, (3e9, 6e9))

## Memory management:
#def get_memory():
#    with open('/proc/meminfo', 'r') as mem:
#        free_memory = 0
#        for i in mem:
#            sline = i.split()
#            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
#                free_memory += int(sline[1])
#    return free_memory
#
#def memory_limit():
#    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

### Measure memory used so far:
#def check_mem_usage_MB():
#    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#    return max_kb_used / 1000.0

##--------------------------------------------------------------------------##

## Home-brew robust statistics:
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

## Home-brew KDE:
#try:
#    import my_kde
#    reload(my_kde)
#    mk = my_kde
#except ImportError:
#    logger.error("module my_kde not found!  Install and retry.")
#    sys.stderr.write("\nError!  my_kde module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

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
## Save FITS image with clobber (astropy / pyfits):
#def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    if header:
#        while (len(header) < padkeys):
#            header.append() # pad header
#    if os.path.isfile(iname):
#        os.remove(iname)
#    pf.writeto(iname, idata, header=header, **kwargs)
#    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
## Save FITS image with clobber (fitsio):
#def qsave(iname, idata, header=None, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    #if os.path.isfile(iname):
#    #    os.remove(iname)
#    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
#    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()




##--------------------------------------------------------------------------##
## New-style string formatting (more at https://pyformat.info/):

#oldway = '%s %s' % ('one', 'two')
#newway = '{} {}'.format('one', 'two')

#oldway = '%d %d' % (1, 2)
#newway = '{} {}'.format(1, 2)

# With padding:
#oldway = '%10s' % ('test',)        # right-justified
#newway = '{:>10}'.format('test')   # right-justified
#oldway = '%-10s' % ('test',)       #  left-justified
#newway = '{:10}'.format('test')    #  left-justified





######################################################################
# CHANGELOG (tangent_proj.py):
#---------------------------------------------------------------------
#
#  2023-06-02:
#     -- Increased __version__ to 0.0.1.
#     -- First created tangent_proj.py.
#

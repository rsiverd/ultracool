#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module contains a set of routines to simplify the retrieval and
# use of JPL HORIZONS ephemerides for later augmentation of FITS image
# headers. Routines are provided to fetch ephemerides using astroquery,
# store ephemerides on disk, and to recall/lookup stored data.
#
# Rob Siverd
# Created:       2021-03-16
# Last modified: 2021-03-16
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
#import calendar
#import ephem
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##

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
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()





##--------------------------------------------------------------------------##
##------------------       Quick SST Ephemeris Recall       ----------------##
##--------------------------------------------------------------------------##

class SSTEph(object):

    def __init__(self):
        return

    def load(self, filename):
        self._eph_file = filename
        gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
        gftkw.update({'names':True, 'autostrip':True})
        gftkw.update({'delimiter':',', 'comments':'%0%0%0%0'})
        self._eph_data = np.genfromtxt(filename, dtype=None, **gftkw)
        self._im_names = self._eph_data['filename'].tolist()
        return

    def retrieve(self, image_names):
        if not np.all([x in self._im_names for x in image_names]):
            sys.stderr.write("Yikes ... images not found??\n")
            return None
        #which = np.array([(x in self._im_names) for x in image_names])
        which = [self._im_names.index(x) for x in image_names]
        tdata = self._eph_data.copy()[which]
        return append_fields(tdata, 't', tdata['jdtdb'], usemask=False)



######################################################################
# CHANGELOG (horizons_eph_helpers.py):
#---------------------------------------------------------------------
#
#  2021-03-16:
#     -- Increased __version__ to 0.0.1.
#     -- First created horizons_eph_helpers.py.
#

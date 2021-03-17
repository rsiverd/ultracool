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
try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
    import astropy.table as apt
    import astropy.time as astt
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
    sys.exit(1)

## HORIZONS queries:
try:
    from astroquery.jplhorizons import Horizons
except ImportError:
    logger.error("Unable to load astroquery/Horizons module!")
    sys.exit(1)

##--------------------------------------------------------------------------##
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()





##--------------------------------------------------------------------------##
##------------------     Ephemeris Retrieval and Storage    ----------------##
##--------------------------------------------------------------------------##

class FetchHorizEphem(object):

    def __init__(self, vlevel=0):
        # Miscellany:
        self._vlevel = vlevel
        self._stream = sys.stderr

        # Query config:
        self._qmax = 50             # max data points per query batch
        self._refplane = 'earth'    # XY-plane parallel to J2000 Earth equator
        self._location = '@0'       # use Solar System Barycenter as origin
        self._target   = {}         # HORIZONS target body identifier dict

        # Image info:
        self._filenames = None
        self._timestamps = None

        # Storage config:
        self._fname_col = 'filename'
        self._drop_cols = ['targetname', 'datetime_str']
        return

    # ----------------------------------------------------

    # Adjust verbosity:
    def set_vlevel(self, vlevel):
        self._vlevel = vlevel

    # vlevel-specific messages:
    def _vlwrite(self, vlmin, vlmax, txt):
        if (vlmin <= self._vlevel <= vlmax):
            self._stream.write(txt)

    # ----------------------------------------------------
    # ----------    High-Level User Routines    ----------
    # ----------------------------------------------------

    # Choose target:
    def set_target(self, targkw):
        if isinstance(targkw, dict):
            self._target = targkw
        else:
            sys.stderr.write("Not a dictionary: %s\n" % str(targkw))
        return

    # Provide filenames and observation times:
    def set_imdata(self, filenames, timestamps):
        """Provide a list of filenames (for later lookup) and corresponding
        time stamps (astropy.Time objects) for the HORIZONS query."""
        if not isinstance(filenames, list):
            sys.stderr.write("Expected list as 'filenames' input\n")
            return False
        if not isinstance(timestamps, astt.Time):
            sys.stderr.write("Expected astropy.Time object times input\n")
            return False
        self._filenames = [x for x in filenames]
        self._timestamps = timestamps.copy()
        return

    # Query ephemeris and return augmented results as astropy Table:
    def get_ephdata(self):
        """Retrieve HORIZONS ephemeris for specified target object at
        the specified times (with matching filenames) and return augmented
        result as astropy Table object."""

        # input sanity check:
        if not self._target:
            sys.stderr.write("No target specified!\n")
            return None
        if (not self._filenames) or (not self._timestamps):
            sys.stderr.write("Filenames/timestamps not specified!\n")
            return None

        # run query:
        hrz_ephem = self._do_query()

        # update and return table:
        return self._update_eph_table(hrz_ephem)

    # ----------------------------------------------------
    # ----------    Low-Level Helper Routines    ---------
    # ----------------------------------------------------

    # Perform batched HORIZONS query:
    def _do_query(self):
        _query_kw = {'location':self._location, **self._target}

        # query in batches to avoid 2000-char URL length limit:
        tik = time.time()
        nchunks = (self._timestamps.tdb.jd.size // self._qmax) + 1
        batches = np.array_split(self._timestamps.tdb.jd, nchunks)
        results = []
        for ii,batch in enumerate(batches, 1):
            self._vlwrite(1, 99, "\rQuery batch %d of %d ... " % (ii, nchunks))
            hrz_query = Horizons(**_query_kw, epochs=batch.tolist())
            batch_eph = hrz_query.vectors(refplane=self._refplane)
            results.append(batch_eph)
        tok = time.time()
        self._vlwrite(1, 99, "done (took %.3f sec).\n" % (tok-tik))
        # combine results:
        return apt.vstack(results)

    # Clean up query results:
    def _update_eph_table(self, eph_table):
        # adjust/remove columns:
        new_table = eph_table.copy()
        new_table.rename_column('datetime_jd', 'jdtdb')
        for cc in self._drop_cols:
            if cc in new_table.keys():
                new_table.remove_column(cc)

        # want filename first in CSV for human readability:
        col_order = [self._fname_col] + new_table.keys()

        # append file names:
        fname_col = apt.Column(data=self._filenames, name=self._fname_col)
        new_table.add_column(fname_col)

        # return reordered result:
        return new_table[col_order]
        

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

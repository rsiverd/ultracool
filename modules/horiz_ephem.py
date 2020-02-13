#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Simple wrapper to interpolate ephemerides from HORIZONS.
#
# Rob Siverd
# Created:       2020-01-01
# Last modified: 2020-01-01
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Logging setup:
import logging
##logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
##logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

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
#import shutil
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##


##--------------------------------------------------------------------------##
##------------------         HORIZONS Base Class            ----------------##
##--------------------------------------------------------------------------##

class HorizEphem(object):

    _astdiv = 79 * '*'
    _estart = "$$SOE"
    _eend   = "$$EOE"
    #_required_lines = [_astdiv, _estart, _eend]
    #_required_count = [8, 1, 1]
    _required_lines = {_astdiv:8, _estart:1, _eend:1}
    _time_vec = 'JDTDB'
    _pos_vecs = [ 'X',  'Y',  'Z']
    _vel_vecs = ['VX', 'VY', 'VZ']
    _itp_vecs = _pos_vecs + _vel_vecs

    def __init__(self, debug=False):
        self.debug = debug
        self._raw_text = None
        self._sections = None
        self._eph_data = None
        return

    # ----------------------------------------- #
    #    Ephemeris Interpolation Routine(s)     #
    # ----------------------------------------- #

    def get_vectors(self, timestamp):
        _when = timestamp.tdb.jd
        _tvec = self._eph_data[self._time_vec]
        result = {}
        for item in self._itp_vecs:
            result[item] = np.interp(_when, _tvec, self._eph_data[item])
        return result

    # ----------------------------------------- #
    #   Ephemeris Loading Drivers and Helpers   #
    # ----------------------------------------- #

    def load_ascii_ephemeris(self, eph_file):
        if not os.path.isfile(eph_file):
            logger.error("file not found: %s" % eph_file)
            return False
        logger.debug("data file located")
        with open(eph_file, 'r') as f:
            content = [x.strip() for x in f.readlines()]
        self._raw_text = content
        if not self._format_looks_okay(content):
            logger.error("unexpected formatting in %s" % eph_file)
            logger.error("probably not a HORIZONS ephemeris ...")
            return False

        # Split up file sections:
        self._sections = self._split_sections(content)
        if not (self._time_vec in self._sections['colnames']):
            logger.error("JDTDB not found among column names!")
            return False

        # Read data into numpy record array:
        self._eph_data = self._make_rec_array(self._sections['npy_load'])
        return True

    # Look for asterisk dividers and $$SEO / $$EOE tags:
    def _format_looks_okay(self, txt_lines, want_divs=8):
        if not txt_lines:
            return False
        nlines = len(txt_lines)
        logger.debug("nlines: %d" % nlines)

        # count occurrences of required items:
        for item,nwant in self._required_lines.items():
            count = len([x for x in txt_lines if x ==item])
            if (count != nwant):
                logger.error("found %d of %d expected %s occurrences"
                        % (count, nwant, item))
                return False
        return True

    def _split_sections(self, txt_lines):
        e1 = txt_lines.index(self._estart)
        e2 = txt_lines.index(self._eend)
        logger.debug("e1=%d, e2=%d" % (e1, e2))
        chunks = {}
        chunks['settings'] = txt_lines[:(e1-2)]
        chunks['colnames'] = txt_lines[e1-2]
        chunks['eph_data'] = txt_lines[e1+1:e2]
        chunks['coordsys'] = txt_lines[(e2+1):]
        chunks['npy_load'] = [x.rstrip(',') for x in \
                [chunks['colnames']] + chunks['eph_data']]
        return chunks

    @staticmethod
    def _make_rec_array(txt_lines):
        gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
        gftkw.update({'names':True, 'autostrip':True})
        gftkw.update({'delimiter':',', 'comments':'%0%0%0%0'})
        return np.genfromtxt([x.rstrip(',') for x in txt_lines],
                        dtype=None, **gftkw)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

data_file = 'horizons_earth.au.txt'
data_file = 'horizons_spitzer.au.txt'
heph = HorizEphem()
success = heph.load_ascii_ephemeris(data_file)
sys.stderr.write("succes: %s\n" % str(success))




######################################################################
# CHANGELOG (horiz_ephem.py):
#---------------------------------------------------------------------
#
#  2020-01-01:
#     -- Increased __version__ to 0.1.0.
#     -- First created horiz_ephem.py.
#

#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# A fairly simple class for making and tracking vertices of a square.
#
# Rob Siverd
# Created:       2026-03-27
# Last modified: 2026-04-02
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.1"

## Modules:
#import glob
#import io
#import gc
import os
import sys
import math
import time
#import pprint
#import pickle
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
#import scipy.stats as scst
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
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

##--------------------------------------------------------------------------##

class QSquare(object):

    # Known origins and which vertex they correspond to:
    _ORIGINS = ['lower-left', 'lower-right',
                'upper-right', 'upper-left', 'center']
    _ORI2VTX = {'center'       :  None,
                'lower-left'   :     0,
                'lower-right'  :     1,
                'upper-right'  :     2,
                'upper-left'   :     3,
               }

    def __init__(self, diam=1.0, xpos=0.0, ypos=0.0, origin='center',
                 theta=0.0, vlevel=0, debug=False):
        #self.orig = 'center'
        #self._nidx = None
        #self.diam = diam
        self._defaults()
        self.debug = False
        self.vlevel = vlevel
        self.set_position(xpos, ypos)
        self.set_origin(origin)
        self.set_rotation_rad(theta)
        #self._update()
        return

    # Default settings for the class:
    def _defaults(self):
        self.diam = 1.0
        self.orig = 'center'
        self._nidx = None
        self.set_position(0.0, 0.0)
        self.set_rotation_rad(0.)

    # Recalculation driver routine (USE BEFORE GETTING):
    def _update(self):
        self._vtx = self._calc_vtx()
        self._ctr = self._calc_center(self._vtx)

    # Retrieve the current vertices:
    def get_vertices(self):
        #self._vtx = self._calc_vtx()
        self._update()
        return self._vtx

    # Retrieve a specific vertex:
    def get_vertex(self, which):
        if not which in self._ORI2VTX.keys():
            sys.stderr.write("Unknown origin: %s\n" % origin)
            sys.stderr.write("Choices are: %s\n" % str(self._ORI2VTX.keys()))
            return
        self._update()
        if which == 'center':
            return self._ctr
        else:
            return self._vtx[:, self._ORI2VTX.get(which)]
        #self._vtx = self._calc_vtx()

    # Calculate central X,Y from first 4 data points:
    @staticmethod
    def _calc_center(vertices):
        return np.mean(vertices[:, :4], axis=1)

    # Update the origin coordinate chooser.
    def set_origin(self, origin):
        #if not origin in self._ORIGINS:
        if not origin in self._ORI2VTX.keys():
            sys.stderr.write("Unknown origin: %s\n" % origin)
            sys.stderr.write("Choices are: %s\n" % str(self._ORI2VTX.keys()))
            return
        prev_vtxs = self._calc_vtx()
        self.orig = origin
        self._nidx = self._ORI2VTX.get(origin)
        if self.debug:
            sys.stderr.write("New origin: %s\n" % self.orig)
            sys.stderr.write("New  _nidx: %s\n" % self._nidx)
        # nudge position to compensate for any origin change:
        curr_vtxs = self._calc_vtx()
        dx, dy = curr_vtxs[:, 0] - prev_vtxs[:, 0]
        self.set_position_rel(-dx, -dy)


    # Update the X,Y position:
    def set_position(self, xpos, ypos):
        self.xpos = xpos
        self.ypos = ypos
        self.posn = np.array([xpos, ypos])
    def set_position_rel(self, rel_xpos, rel_ypos):
        self.xpos = self.xpos + rel_xpos
        self.ypos = self.ypos + rel_ypos
        self.posn = np.array([self.xpos, self.ypos])

    # Update rotation angle:
    def set_rotation_rad(self, r_theta):
        self.theta = r_theta
        self._rmat = self._vector_zrot(r_theta)
    def set_rotation_deg(self, d_theta):
        return self.set_rotation_rad(np.radians(d_theta))

    # Make 0-center vertices:
    def _mkvtx0(self):
        LL = BB = -0.5 * self.diam      # left, bottom
        RR = TT =  0.5 * self.diam      # right, top
        sqvtx = np.array([(LL, BB),     # lower-left
                          (RR, BB),     # lower-right
                          (RR, TT),     # upper-right
                          (LL, TT),     # upper-left
                          (LL, BB)])    # lower-left (closure)
        return sqvtx.T.copy()

    # Recalculate the square
    def _calc_vtx(self):
        vtx = self._mkvtx0()            # make 0-centered square
        if self._nidx is not None:      # compensate origin
            vtx -= vtx[:, self._nidx][:, None]
        vtx = self._rmat @ vtx
        return vtx + self.posn[:, None]

    # Z-axis rotation matrix:
    @staticmethod
    def _vector_zrot(r_theta):
        c, s = math.cos(r_theta), math.sin(r_theta)
        return np.array((c, -s, s, c)).reshape(2, 2)

######################################################################
# CHANGELOG (qsquare.py):
#---------------------------------------------------------------------
#
#  2026-03-27:
#     -- Increased __version__ to 0.1.0.
#     -- First created qsquare.py.
#

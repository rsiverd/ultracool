#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module contains the routines to transform pixel coordinates
# in the NW,SE,SW sensors to/from NE pixel coordinates.
#
# Rob Siverd
# Created:       2026-04-02
# Last modified: 2026-04-02
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Modules:
#import argparse
#import shutil
#import resource
#import signal
#import glob
#import io
#import gc
import os
import sys
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
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Parameter matrix:
## | MCD11   MCD12   Ox  |
## | MCD21   MCD22   Oy  |
## |     0       0    1  |

## BRIEF BACKGROUND SUMMARY:
##    We expect the four detectors in the WIRCam mosaic to be rigidly coupled
## to one another by construction. We expect time-varying flexure within the 
## instrument to be negligible and do not consider it here. For a detector
## mosaic with this rigid construction, transformations between the pixel
## coordinates of the different detectors SHOULD BE CONSTANT IN TIME. We 
## believe that these can be fit once for the entire data set and used for
## the lifetime of the instrument with minimal (or zero) modification. 
##    It is not clear whether a full 6-parameter affine transformation is
## needed for the WIRCam mosaic. It might be possible to use scale, rotation,
## and translation (SRT) alone. In this case, there would be 5 parameters per
## detector instead of 6 but the affine transformation math is unchanged. In
## other words, a single affine transformation formula can accommodate all
## of the parameterizations we intend to use.
##
## GOAL:
## We aim to estimate transformation parameters between the pixel coordinates
## of the various WIRCam detectors. Using these transformations, X,Y pixel
## coordinates of all four sensors can be "unified" onto a single, continuous
## grid. We adopt the NE detector as the reference grid. Detector positions
## on the other three sensors (NW, SE, SW) will be transformed onto the NE
## detector system. This NE system will then have a SINGLE WCS transformation
## between the detector plane and the sky. 
## 
## DISTORTION:
## We expect to apply distortion to the "unified" mosaic in mosaic (NE pixel) 
## coordinates before projecting positions onto the sky. The simplest case we
## consider is radial distortion but there is an option to extend this to the
## elliptic case. Elliptical distortion could arise from a tip/tilt of the
## mosaic itself with respect to the focal plane (expected to vary by RUNID)
## or it could relate to telescope attitude, which might require a per-image
## solution to the distortion tip/tilt/angle. In either case, we aim to
## separately fit for a radial distortion profile and potential tip/tilt/angle
## of that profile with respect to the mosaic.

## For more background, see the cdm_crpix.pdf document. 

## Parameter format in 1D for a single detector:
## [MCD11, MCD12, MCD21, MCD22, Ox, Oy]

## Given an 18-element array of transformation parameters, break out with:
## * pars_nw, pars_se, pars_sw = pars_all.reshape(3, -1)

## Parameters sets will concatenate in NW, SE, SW order for consistency
## with existing software. Optimization The base 6-parameter affine transformation model 
## works for the general case of 
## codes and scripts. The global fit needs to fit 6 parameters per sensor
## (18 total parameters) in the most complete case.

## The parameters for a detector are rearranged as follows before
## performing matrix multiplication:
#xform = np.array([[par[0], par[1], par[4]],
#                  [par[2], par[3], par[5]],
#                  [   0.0,    0.0,    1.0]])

def mkaffine(par):
    return np.array([[par[0], par[1], par[4]],
                     [par[2], par[3], par[5]],
                     [   0.0,    0.0,    1.0]])

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##




######################################################################
# CHANGELOG (sensor_transform.py):
#---------------------------------------------------------------------
#
#  2026-04-02:
#     -- Increased __version__ to 0.1.0.
#     -- First created sensor_transform.py.
#

#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Try to measure the pixel scale from WCS solutions.
#
# Rob Siverd
# Created:       2023-07-13
# Last modified: 2023-07-13
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
#import glob
import os
import sys
import time
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
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##
## Quick ASCII I/O:
data_file = '20230629--final_wcs_params.csv'
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = np.atleast_1d(np.genfromtxt(data_file, dtype=None, **gftkw))
#all_data = np.genfromtxt(fix_hashes(data_file), dtype=None, **gftkw)
#all_data = aia.read(data_file)

pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
#all_data = pd.read_csv(data_file)
all_data = pd.read_csv(data_file, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)


cd11 = all_data.CD11
cd12 = all_data.CD12
cd21 = all_data.CD21
cd22 = all_data.CD22

npts = len(cd11)
every_cdmat = np.column_stack((cd11, cd12, cd21, cd22)).reshape(npts, 2, 2)

every_scale = np.sqrt(np.sum(every_cdmat**2, axis=-1))


avg_pscale = 3600.0 * np.average(every_scale)
med_pscale = 3600.0 *  np.median(every_scale)

sys.stderr.write("avg_pscale: %.7f arcsec/pix\n" % avg_pscale)
sys.stderr.write("med_pscale: %.7f arcsec/pix\n" % med_pscale)

## One-at-a-time method:
#for td11,td12,td21,td22 in zip(cd11, cd12, cd21, cd22):
cdm_rot_deg = []
for this_cdm in every_cdmat:
    #cd_pscales = np.sqrt(np.sum(
    cd_pscales = np.sqrt(np.sum(this_cdm**2, axis=1))
    norm_cdmat = this_cdm / cd_pscales
    cd_ang_rad = np.arccos(norm_cdmat[0, 0])
    cd_ang_deg = np.degrees(cd_ang_rad)
    cdm_rot_deg.append(cd_ang_deg)
    pass
cdm_rot_deg = np.array(cdm_rot_deg)



######################################################################
# CHANGELOG (01_measure_pixel_scale.py):
#---------------------------------------------------------------------
#
#  2023-07-13:
#     -- Increased __version__ to 0.0.1.
#     -- First created 01_measure_pixel_scale.py.
#

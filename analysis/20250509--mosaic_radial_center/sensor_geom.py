#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Sensor-to-sensor geometry helper module.
#
# Rob Siverd
# Created:       2025-06-03
# Last modified: 2025-06-03
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
import gc
import os
import sys
import time
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

### Compute CRPIX1,2 for NW,SE,SW sensors given trial NE CRPIX1,2:
##def get_nwsesw_crpix(ne_crpix1, ne_crpix2):
#def get_4sensor_crpix(ne_crpix1, ne_crpix2):
#    result = {'NE':(ne_crpix1, ne_crpix2)}
#
#    # Distance off the NE inner corner in NE pixel coordinates is:
#    dx = ne_crpix1 - 2048
#    dy = ne_crpix2 -    1
#    
#    # For NW sensor, start with pixel coordinate of NE inner corner and offset:
#    #NW_CRPIX1 = -134.580 + NENW_cd11 * DX + NENW_cd12 * DY
#    #NW_CRPIX2 =    9.697 + NENW_cd21 * DX + NENW_cd22 * DY
#    nw_crpix1 = -134.580 + 0.9999522 * dx + 0.0042017 * dy
#    nw_crpix2 =    9.697 + 0.0063453 * dx + 0.9998133 * dy
#    result['NW'] = (nw_crpix1, nw_crpix2)
# 
#    # For SE sensor, start with pixel coordinate of NE inner corner and offset:
#    #SE_CRPIX1 = 2052.228 + NESE_cd11 * DX + NESE_cd12 * DY
#    #SE_CRPIX2 = 2192.539 + NESE_cd21 * DX + NESE_cd22 * DY
#    se_crpix1 = 2052.228 + 1.0022410 * dx + 0.0027602 * dy
#    se_crpix2 = 2192.539 + 0.0055770 * dx + 0.9998227 * dy
#    result['SE'] = (se_crpix1, se_crpix2)
#    
#    # For SW sensor, start with pixel coordinate of NE inner corner and offset:
#    #SW_CRPIX1 = -137.002 + NESW_cd11 * DX +  NESW_cd12 * DY
#    #SW_CRPIX2 = 2195.456 + NESW_cd21 * DX +  NESW_cd22 * DY
#    sw_crpix1 = -137.002 + 1.0026923 * dx + -0.0001494 * dy
#    sw_crpix2 = 2195.456 + 0.0024810 * dx +  1.0004179 * dy
#    result['SW'] = (sw_crpix1, sw_crpix2)
#
#    # The whole set:
#    return result

##--------------------------------------------------------------------------##

## Compute CRPIX1,2 for NW,SE,SW sensors given trial NE CRPIX1,2:
#def get_nwsesw_crpix(ne_crpix1, ne_crpix2):
def get_4sensor_crpix(ne_crpix1, ne_crpix2):
    result = {'NE':(ne_crpix1, ne_crpix2)}

    # Distance off the NE inner corner in NE pixel coordinates is:
    dx = ne_crpix1 - 2048
    dy = ne_crpix2 -    1
    
    # For NW sensor, start with pixel coordinate of NE inner corner and offset:
    #NW_CRPIX1 = -134.580 + NENW_cd11 * DX + NENW_cd12 * DY
    #NW_CRPIX2 =    9.697 + NENW_cd21 * DX + NENW_cd22 * DY
    #nw_crpix1 = -134.580 + 0.9999515 * dx + 0.0063534 * dy
    #nw_crpix2 =    9.697 + 0.0041989 * dx + 0.9998084 * dy
    nw_crpix1 = -138.889 + 0.9999515 * dx + 0.0063534 * dy
    nw_crpix2 =    8.497 + 0.0041989 * dx + 0.9998084 * dy
    result['NW'] = (nw_crpix1, nw_crpix2)
 
    # For SE sensor, start with pixel coordinate of NE inner corner and offset:
    #SE_CRPIX1 = 2052.228 + NESE_cd11 * DX + NESE_cd12 * DY
    #SE_CRPIX2 = 2192.539 + NESE_cd21 * DX + NESE_cd22 * DY
    #se_crpix1 = 2052.228 + 1.0000901 * dx + 0.0055722 * dy
    #se_crpix2 = 2192.539 + 0.0027586 * dx + 0.9998193 * dy
    se_crpix1 = 2052.118 + 1.0000901 * dx + 0.0055722 * dy
    se_crpix2 = 2195.059 + 0.0027586 * dx + 0.9998193 * dy
    result['SE'] = (se_crpix1, se_crpix2)
    
    # For SW sensor, start with pixel coordinate of NE inner corner and offset:
    #SW_CRPIX1 = -137.002 + NESW_cd11 * DX +  NESW_cd12 * DY
    #SW_CRPIX2 = 2195.456 + NESW_cd21 * DX +  NESW_cd22 * DY
    #sw_crpix1 = -137.002 +  1.0005394 * dx + 0.0024817 * dy
    #sw_crpix2 = 2195.456 + -0.0001453 * dx + 1.0004068 * dy
    sw_crpix1 = -140.980 +  1.0005394 * dx + 0.0024817 * dy
    sw_crpix2 = 2208.131 + -0.0001453 * dx + 1.0004068 * dy
    result['SW'] = (sw_crpix1, sw_crpix2)

    # The whole set:
    return result


#def get_4sensor_crpix(ne_crpix1, ne_crpix2):
#    return {
#            'NE': (2122.690779, -81.67888761),
#            'NW': ( -69.6750095, -77.65628599),
#            'SE': (2121.88321,  2122.762079),
#            'SW': ( -76.9217609, 2128.412456),
#            }


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
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##





######################################################################
# CHANGELOG (sensor_geom.py):
#---------------------------------------------------------------------
#
#  2025-06-03:
#     -- Increased __version__ to 0.1.0.
#     -- First created sensor_geom.py.
#

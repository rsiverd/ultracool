#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module stores my current best guess for per-sensor CD matrix and
# CRPIX values as a function of RUNID (and possibly filter). Solutions
# here will preempt the dumb initial guesses in 12_factor_fit_4pack.py
# and elsewhere.
#
# Rob Siverd
# Created:       2026-02-25
# Last modified: 2026-02-25
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

from importlib import reload
import copy
import numpy as np

## Solution parameter helpers:
import slv_par_tools
reload(slv_par_tools)
spt = slv_par_tools

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


## J-band solutions:
J_wcs_pars = {}

J_wcs_pars['17AQ07'] = {
    'cdmat': {
        'NE': np.array([-0.0000851 , -0.00000029, -0.00000024,  0.00008499]),
        'NW': np.array([-0.00008496, -0.00000006, -0.00000005,  0.00008496]),
        'SE': np.array([-0.00008512, -0.00000028, -0.00000028,  0.00008519]),
        'SW': np.array([-0.00008492,  0.00000001, -0.00000001,  0.0000852 ]),
        },
    'crpix': {
        'NE': np.array([2084.0954883 ,  149.17508844]),
        'NW': np.array([ -99.3582719 ,  156.76241002]),
        'SE': np.array([2081.73451733, 2343.34218411]),
        'SW': np.array([-103.75574761, 2353.83891436]),
        },
    'crval':[0.0, 0.0],
    'rpars':np.array([]),
}

J_wcs_pars['18AQ15'] = {
    'cdmat': {
        'NE': np.array([-0.00008473,  0.00000057,  0.00000058,  0.00008522]),
        'NW': np.array([-0.00008515,  0.00000084,  0.00000081,  0.00008505]),
        'SE': np.array([-0.00008503,  0.00000069,  0.00000086,  0.00008521]),
        'SW': np.array([-0.00008475,  0.00000074,  0.00000078,  0.00008515]),
        },
    'crpix': {
        'NE': np.array([1934.53753273, -131.19247409]),
        'NW': np.array([-252.03722642, -128.58649563]),
        'SE': np.array([1927.33743675, 2056.16868595]),
        'SW': np.array([-262.95867008, 2066.53157482]),
        },
    'crval':[0.0, 0.0],
    'rpars':np.array([]),
}

J_wcs_pars['21AQ18'] = {
    'cdmat': {
        'NE': np.array([-0.00008508,  0.00000052,  0.00000052,  0.00008508]),
        'NW': np.array([-0.00008509,  0.00000071,  0.00000071,  0.0000851 ]),
        'SE': np.array([-0.00008509,  0.00000064,  0.00000063,  0.00008508]),
        'SW': np.array([-0.00008506,  0.00000073,  0.00000073,  0.00008504]),
        },
    'crpix': {
        'NE': np.array([2124.0630655 ,  -91.96674513]),
        'NW': np.array([ -59.94684968,  -83.17030954]),
        'SE': np.array([2127.98554353, 2100.62029866]),
        'SW': np.array([ -62.49889049, 2113.12048686]),
        },
    'crval':[0.0, 0.0],
    'rpars':np.array([]),
}

J_wcs_pars['23AQ21'] = {
    'cdmat': {
        'NE': np.array([-0.00008475,  0.00000068,  0.00000051,  0.0000852 ]),
        'NW': np.array([-0.00008514,  0.0000009 ,  0.00000099,  0.00008518]),
        'SE': np.array([-0.00008499,  0.00000088,  0.0000009 ,  0.00008499]),
        'SW': np.array([-0.00008495,  0.00000081,  0.00000084,  0.0000851 ]),
        },
    'crpix': {
        'NE': np.array([1955.10248175, -252.48731546]),
        'NW': np.array([-231.054922  , -248.66432523]),
        'SE': np.array([1951.31480627, 1936.5387148 ]),
        'SW': np.array([-239.88924116, 1943.69715484]),
        },
    'crval':[0.0, 0.0],
    'rpars':np.array([]),
}

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


## Check whether or not we have a solution:
def solution_is_known(runid):
    return runid in J_wcs_pars

## Unsift prior parameters for a single image with CRVAL overwrite:
def single_image_params(runid, crval1, crval2):
    pars = copy.deepcopy(J_wcs_pars[runid])
    pars['crval'] = [crval1, crval2]    # overwrite dummy CRVAL
    return spt.unsift_params(pars)



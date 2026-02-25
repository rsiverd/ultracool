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



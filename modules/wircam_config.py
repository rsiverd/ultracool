#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# General WIRCam information to simplify processing and analysis.
#
# Rob Siverd
# Created:       2024-10-21
# Last modified: 2024-10-21
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Modules:
#import gc
#import os
#import sys
#import time
#import numpy as np
#from numpy.lib.recfunctions import append_fields


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## WIRCam configuration:

quad_exts = {
        'NW'    :   (1, 'HAWAII-2RG-#77'),
        'SW'    :   (2, 'HAWAII-2RG-#52'),
        'SE'    :   (3, 'HAWAII-2RG-#54'),
        'NE'    :   (4, 'HAWAII-2RG-#60'),
}




######################################################################
# CHANGELOG (wircam_config.py):
#---------------------------------------------------------------------
#
#  2024-10-21:
#     -- Increased __version__ to 0.1.0.
#     -- First created wircam_config.py.
#

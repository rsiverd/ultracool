#!/usr/bin/env python
#
# Test the astrometry.net facility provided by astroquery.
#

from importlib import reload
from astroquery.astrometry_net import AstrometryNet
import numpy as np
import os, sys, time

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    sys.stderr.write("failed to import extended_catalog module!")
    sys.exit(1)

## File to use:
fcat_path = "/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid/17BQ03/wircam_J_2166830p.fits.fz.fcat"

## Make objects:
ast = AstrometryNet()
ccc = ec.ExtendedCatalog()

## Image size:
nxpix, nypix = 2048, 2048

## List the allowed settings:
ast.show_allowed_settings()

## Solver settings:
solver_settings = {
        'parity'        :             2,      # for CFHT
        'scale_units'   : 'arcminwidth',
        'scale_lower'   :           5.0,
        'scale_upper'   :          15.0,
}

## Lists of coordinate keys to fix (if seen):
_ra_keys = ['dra', 'wdra', 'ppdra', 'akra']
_de_keys = ['dde', 'wdde', 'ppdde', 'akde']


## Load catalog:
ccc.load_from_fits(fcat_path)
imcat = ccc.get_catalog()

## Attempt a solve:
wcs_header = ast.solve_from_source_list(imcat['x'], imcat['y'], 
        nxpix, nypix, **solver_settings)



## Parameters to use to REMOVE distortion from detector positions measured
## with e.g., SExtractor. Use these for data reduction.
pars_det2sky = np.array([ 1.50436502e-01,  1.20381196e-03,  1.01549191e-06,
                          8.44221727e-11,  4.15778286e-13, -9.85209032e-17])

## Parameter to use to ADD the distortion relative X,Y positions produced
## from sky catalogs. These are needed to *predict* the detector coordinates
## that correspond to catalog positions that have already gone through most
## of an inverse WCS transformation.
pars_sky2det = np.array([ 1.60019291e-01,  1.14388049e-03,  1.15982724e-06,
                         -5.87754793e-11,  4.90335898e-13, -1.10582796e-16])

## Polynomial evaluator:
def radial_poly_eval(r, c0, c1, c2, c3, c4, c5):
    return c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r*c5))))

## -----------------------------------------------------------------------
## The radial distortion polynomial connects the *magnitude* of the distortion
## correction to a radial position. It does not capture the correction SIGN.
## As a result, the way to apply the correction depends on which direction
## the transformation is going. The two routines below are what you need.
##
## In both cases, "xrel" and "yrel" are relative to the CRPIX1,CRPIX2 origin.
## The relative coordinates can be distorted or not. Explicitly,
##
## The undistorted coordinates are called 'flat' in variables. Undistorted
## X,Y coordinates are ready for WCS transformation into RA/DE positions.
## * xrel_flat = undistorted_x - CRPIX1     # from inverse_wcs(catalog_pos)
## * yrel_flat = undistorted_y - CRPIX2     # from inverse_wcs(catalog_pos)
##
## The distorted coordinates are called 'dist' in variables. Distorted
## X,Y coordinates are what you measure on the detector with SEP/SExtractor.
## * xrel_dist = raw_measured_x - CRPIX1 
## * yrel_dist = raw_measured_y - CRPIX2

## The following is a general-purpose correction calculator. It
## returns the magnitude of the correction for input relative X,Y
## coordinates. It does not handle the sign convention and does
## not choose the correct model on its own. Use the wrappers below
## to get the desired results.
def calc_rdist_corrections(xrel, yrel, model):
    rdist = np.hypot(xrel, yrel)     # distance from CRPIX
    rcorr = poly_eval(rdist, model)  # total correction magnitude
    theta = np.arctan2(yrel, xrel)
    xcorr = rcorr * np.cos(theta)
    ycorr = rcorr * np.sin(theta)
    return xcorr, ycorr

## The following routine ADDS detector distortion to 'flat' relative X,Y
## coordinates. The input arrays are UNDISTORTED xrel, yrel. This routine
## is used to predict the detector X,Y positions when working backwards
## from catalog sky positions --> WCS --> detector positions.
## The appropriate parameter array (pars_sky2det) is hard-coded inside.
def redistort_sky_positions(xrel_flat, yrel_flat):
    xnudge, ynudge = calc_rdist_corrections(xrel_flat, yrel_flat, pars_sky2det)
    xrel_dist = xrel_flat + xnudge
    yrel_dist = yrel_flat + ynudge
    return xrel_dist, yrel_dist

## The following routine REMOVES detector distortion from 'dist' relative X,Y
## coordinates. The input arrays are DISTORTED xrel, yrel. This routine is
## used to remove distortion from measured X,Y positions before apply the
## WCS transformation (working detector -> WCS -> sky direction).
## The appropriate parameter array (pars_det2sky) is hard-coded inside.
def undistort_det_positions(xrel_dist, yrel_dist):
    xnudge, ynudge = calc_rdist_corrections(xrel_dist, yrel_dist, pars_det2sky)
    xrel_dist = xrel_dist - xnudge
    yrel_dist = yrel_dist - ynudge
    return xrel_flat, yrel_flat


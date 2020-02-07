## The following illustrates how to compute parallax-induced corrections to both
## RA and Dec from nominal position and solar system barycenter XYZ position.

import numpy as np

## Compute parallax factors:
def parallax_factors(meanRA, meanDE, X, Y, Z):
    """Compute RA offset due to parallax given solar system XYZ position.

    Inputs:
    meanRA      --  mean/nominal RA coordinate (degrees)
    meanDE      --  mean/nominal DE coordinate (degrees)
    X, Y, Z     --  solar system barycentric coordinates (in AU)

    Returns:
    pfRA, pfDE -- tuple of RA, DE parallax factors
    """
    rra = np.radians(meanRA)
    rde = np.radians(meanDE)
    ra_factor = (X * np.sin(rra) - Y * np.cos(rra)) / np.cos(rde)
    de_factor =   X * np.cos(rra) * np.sin(rde) \
                + Y * np.sin(rra) * np.sin(rde) \
                - Z * np.cos(rde)
    return ra_factor, de_factor

def calc_parallax_factors(RA_rad, DE_rad, X_au, Y_au, Z_au):
    sinRA, cosRA = np.sin(RA_rad), np.cos(RA_rad)
    sinDE, cosDE = np.sin(DE_rad), np.cos(DE_rad)
    ra_factor = (X_au * sinRA - Y_au * cosRA) / cosDE
    de_factor =  X_au * cosRA * sinDE \
              +  Y_au * sinRA * sinDE \
              -  Z_au * cosDE
    return ra_factor, de_factor

## Compute RA offset due to parallax:
def parallax_offsets(meanRA, meanDE, X, Y, Z, parallax):
    """Compute RA offset due to parallax given solar system XYZ position.

    Inputs:
    meanRA      --  mean/nominal RA coordinate (degrees)
    meanDE      --  mean/nominal DE coordinate (degrees)
    X, Y, Z     --  solar system barycentric coordinates (in AU)
    parallax    --  target parallax

    Returns:
    RA_shift    --  parallax adjustment in RA (same units as parallax)
    DE_shift    --  parallax adjustment in DE (same units as parallax)
    """
    ra_factor, de_factor = parallax_factors(meanRA, meanDE, X, Y, Z)
    return (parallax * ra_factor, parallax * de_factor)


## Compute apparent positions given true positions and time of observation:
_ARCSEC_PER_RADIAN = 180. * 3600.0 / np.pi
def apparent_radec(t_ref, astrom_pars, eph_obs):
    """
    t_ref       --  chosen reference epoch
    astrom_pars --  five astrometric parameters specified at the
                    reference epoch: meanRA (rad), meanDE (rad),
                    pmRA*cos(DE), pmDE, and parallax
    eph_obs     --  dict with x,y,z,t elements describing the times
                    and places of observations (numpy arrays)
    FOR NOW, assume
                [t_ref] = JD (TDB)
                [t]     = JD (TDB)
                [pars]  = rad, rad, arcsec/yr, arcsec/yr, arcsec
                                   *no cos(d)*
    """

    rra, rde, pmra, pmde, prlx = astrom_pars
    
    t_diff_yr = (eph_obs['t'] - t_ref) / 365.25     # units of years

    pfra, pfde = calc_parallax_factors(rra, rde,
            eph_obs['x'], eph_obs['y'], eph_obs['z'])

    delta_ra = (t_diff_yr * pmra / _ARCSEC_PER_RADIAN) + (prlx * pfra)
    delta_de = (t_diff_yr * pmde / _ARCSEC_PER_RADIAN) + (prlx * pfde)

    return (rra + delta_ra, rde + delta_de)

import astropy.units as uu
_par_keys = ['ra', 'de', 'pmra', 'pmde', 'plx']
_par_vals = [123.45, 54.32, 1.23, 4.56, 0.4]
_par_unit = [uu.deg, uu.deg, uu.arcsec/uu.year,
        uu.arcsec/uu.year, uu.arcsec]
data = np.array(_par_vals) * np.array(_par_unit)


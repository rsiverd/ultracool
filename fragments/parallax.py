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




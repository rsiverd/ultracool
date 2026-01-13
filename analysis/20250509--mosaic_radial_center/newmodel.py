

# new parameters:
# ne_pa, ne_crpix1, ne_crpix2
# nw_pa, nw_crpix1, nw_crpix2
# se_pa, se_crpix1, se_crpix2
# sw_pa, sw_crpix1, sw_crpix2
# pixscale_deg
# crval1, crval2

guessmo = np.array(
        [8.5e-5, 294.59363733,   35.11815553,
         0.0,   2090.62211249,  -82.73910775,
         0.0,    -93.49623245,  -73.89771061,
         0.0,   2094.58739653, 2109.68403443,
         0.0,    -96.14395827, 2122.3897682,])

##--------------------------------------------------------------------------##

## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array((c, -s, s, c)).reshape(2, 2)

## Reflection matrices:
xref_mat  = np.array((( 1.0, 0.0), (0.0, -1.0)))
yref_mat  = np.array(((-1.0, 0.0), (0.0,  1.0)))
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array((( 1.0, 0.0), (0.0,  1.0)))

import itertools as itt

## Radian-to-degree converter:
_radeg = 180.0 / np.pi

## Make a CD matrix given pixel scale and PA:
def make_cdmat(pa_rad, pxscale_deg):
    #rmat = rotation_matrix(pa_rad)
    return pxscale_deg * np.dot(xflip_mat, rotation_matrix(pa_rad))

def make_qpars(pxscale_deg, pa_rad, crpix1, crpix2):
    return make_cdmat(pa_rad, pxscale_deg).ravel().tolist() + [crpix1, crpix2]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

def mutate_params(newpars):
    pscale, cv1, cv2 = newpars[:3]
    pacrpix = newpars[3:].reshape(4, 3)
    return np.array(
            list(itt.chain.from_iterable([make_qpars(pscale, *stuff) for stuff in pacrpix])) \
                    + [cv1, cv2])
    
    for pa,cp1,cp2 in pacrpix:
        asdf = make_cdmat(pa, pscale).ravel()
    
    #[make_qpars(pscale, *stuff) for stuff in pacrpix] + [cv1, cv2]
    #itt.chain.from_iterable([make_qpars(pscale, *stuff) for stuff in pacrpix])  + [cv1, cv2]
    list(itt.chain.from_iterable([make_qpars(pscale, *stuff) for stuff in pacrpix]))  + [cv1, cv2]



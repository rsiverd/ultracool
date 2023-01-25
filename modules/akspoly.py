#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Module to provide Adam Kraus' Spitzer distortion solution.
#
# Rob Siverd
# Created:       2019-09-04
# Last modified: 2022-09-14
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.4.0"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
#import shutil
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
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import fov_rotation
reload(fov_rotation)
rfov = fov_rotation.RotateFOV()

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Channel mapping:
_chmap = {
        1:1,  '1':1, 'ch1':1,
        2:2,  '2':2, 'ch2':2,
        3:3,  '3':3, 'ch3':3,
        4:4,  '4':4, 'ch4':4,
}

## Dephasing parameter dictionaries:
_ppa = {'cryo':{}, 'warm':{}}
_ppb = {'cryo':{}, 'warm':{}}
#_ppa, _ppb = {}, {}
#_aaa, _bbb = {}, {}

## Cryo mission parameters:
_ppa['cryo'][1] = np.array([+2.783938E-02, +2.046231E-02, -3.609477E-03, -7.669185E-01, +3.232491E-02, +1.276753E-01, +5.724266E-01, -1.491381E-01, -5.018244E-01, +2.857436E-01, +1.270852E+00, +7.535925E-02, +1.313992E-03, -1.956625E-01, -4.658775E-01, -2.770535E+00, +6.981738E-01, +1.554831E+00, -6.419691E-02, +7.382683E-01, -1.262995E+00])
_ppb['cryo'][1] = np.array([-2.831900E-03, +7.436390E-05, -4.321248E-02, -5.004743E-02, +2.975831E-02, -2.752005E-02, +7.136074E-02, -3.121356E-01, +3.678197E-01, +9.041720E-02, +1.646660E-01, -2.555968E-02, -1.893717E-01, -1.369838E-01, -2.143417E-01, -3.711832E-01, +2.288625E-01, -9.326759E-01, +8.925945E-01, -9.874694E-01, +1.274447E-01])
_ppa['cryo'][2] = np.array([+1.560036E-02, -8.046778E-02, +4.347387E-03, -4.658847E-01, +2.773967E-03, +5.260310E-03, +6.406896E-01, -1.433339E-01, +1.236054E-02, +8.189350E-02, +3.777830E-01, -4.622371E-05, +6.629448E-02, -6.734899E-02, -1.666788E-01, -1.160999E+00, +5.561046E-01, +1.101566E-02, -3.533757E-02, -8.704616E-01, -3.367478E-01])
_ppb['cryo'][2] = np.array([-4.911884E-03, -1.597777E-02, -8.055713E-02, -1.993140E-02, +1.408297E-03, -1.959913E-01, +1.738639E-01, -1.320856E-01, +6.978489E-02, +2.124757E-01, -7.466133E-02, -8.647830E-02, +1.998463E-01, -3.775696E-02, +1.012916E-01, -5.557408E-01, -4.995205E-01, +6.448413E-02, +9.568401E-01, -1.708307E-01, +1.892472E-01])
_ppa['cryo'][3] = np.array([-2.958072E-02, +2.026849E-02, -7.964571E-02, -3.297237E-01, +7.959587E-02, +6.688262E-04, +4.893981E-02, +3.205900E-01, +1.701601E+00, +8.494186E-01, +1.167754E+00, -1.866597E-01, -6.758463E-02, -9.820143E-02, -1.786983E-01, -4.945196E-01, -6.789277E-01, -3.755359E+00, -1.419667E+00, -6.513492E+00, -2.141430E+00])
_ppb['cryo'][3] = np.array([+4.889141E-03, +4.892661E-02, -2.633427E-02, -9.054284E-02, -1.235002E-01, -3.679879E-01, -8.481535E-01, +2.337722E-02, -9.665800E-01, +4.488875E-01, +3.975577E-01, +2.139587E-01, -1.885543E-01, +7.415889E-01, +7.630252E-01, +3.191154E+00, +4.774575E-01, +4.692128E+00, -4.824242E-01, +1.946208E+00, -1.692083E+00])
_ppa['cryo'][4] = np.array([-7.716998E-02, +3.117923E-01, +5.264274E-02, +1.326105E-02, +1.599413E-03, -2.830490E-01, -2.083720E+00, -3.426859E-01, +1.393718E+00, -4.865320E-01, -4.885021E-01, -6.302262E-01, +1.673146E+00, +2.399258E-01, +1.044011E+00, +3.354562E+00, -1.340646E+00, +2.386917E-01, +5.725999E+00, -9.903208E+00, +3.530657E-01])
_ppb['cryo'][4] = np.array([-3.592878E-02, +3.136689E-02, +2.694646E-01, -4.197915E-01, +1.979760E-01, +3.747094E-02, -1.068913E+00, +8.309793E-01, +1.578866E-02, -9.650013E-01, +1.333649E+00, -9.932210E-01, +1.172701E+00, -8.285775E-01, -1.223195E+00, +2.984574E+00, -4.530653E+00, +2.838048E+00, -2.905586E+00, -8.142194E-01, +3.327983E-01])
#_ppa['cryo'] = _aaa
#_ppb['cryo'] = _bbb

## Warm mission parameters:
_ppa['warm'][1] = np.array([+5.298110E-02, -1.031251E-01, +3.433206E-04, -7.190475E-01, -5.493761E-03, +3.039381E-02, +1.368275E+00, -4.923920E-02, -2.766547E-01, +9.866965E-03, +7.050155E-01, +6.951617E-02, -2.790677E-02, +3.136094E-02, +4.057089E-02, -3.810908E+00, +1.158134E-01, +7.586575E-01, -4.799565E-02, +1.531817E-01, -3.483019E-02])
_ppb['warm'][1] = np.array([+1.243723E-02, -2.326365E-03, -1.560887E-01, -6.648988E-02, +3.912225E-02, -1.279622E-01, +2.325948E-02, +2.361877E-02, +5.564555E-02, +2.068253E-01, +1.747259E-01, -1.176636E-01, +1.084111E-01, -1.289356E-01, -8.817560E-03, -1.606868E-01, -2.181416E-01, -1.361416E-01, -1.593799E-01, +9.579846E-02, +1.682768E+00])
_ppa['warm'][2] = np.array([+4.050731E-02, -1.887555E-01, -1.919032E-02, -4.133482E-01, +3.262207E-04, +2.888772E-03, +8.406737E-01, +1.438462E-01, +3.580412E-02, +1.716415E-01, -9.744711E-02, +5.304771E-02, -3.940930E-02, -9.872928E-04, -4.662450E-02, -3.890232E-01, -4.835268E-01, +5.128984E-01, -1.795073E-01, -6.409911E-01, -4.390934E-01])
_ppb['warm'][2] = np.array([+1.260093E-02, -1.665688E-02, -1.549771E-01, -2.083848E-02, +3.977243E-02, -2.426269E-01, +1.287132E-01, -1.318016E-02, +6.500871E-02, +3.146763E-01, +1.297098E-02, +6.563227E-03, +3.919026E-02, -1.990932E-01, +6.797772E-02, -2.070237E-01, -5.477868E-01, -2.156148E-01, +7.003856E-01, -2.158017E-01, +1.085267E+00])
#_ppa['warm'] = _aaa
#_ppb['warm'] = _bbb

## Cleanup:
#del _aaa, _bbb

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Distortion parameter arrays (one set for cryo+warm Spitzer):
_ka, _kb = {}, {}
_ka[1] = np.array([-1.987895E-03, +4.848572E-04, -1.909555E-04, -1.870311E-05, +2.044248E-05, +3.535586E-06, -2.312227E-07, +1.145814E-09, -2.420678E-07, +1.881476E-08, -3.250621E-10, +9.190708E-11, -1.040106E-10, -4.774988E-11, +4.641016E-11, +4.242499E-12, -4.978284E-13, +2.999052E-12, -2.219252E-13, +3.029946E-12, -5.917890E-13])
_kb[1] = np.array([+7.389430E-04, -8.399943E-04, -4.172204E-05, -9.536498E-07, -2.196779E-05, +2.340636E-05, +3.181545E-08, -1.927947E-07, +3.026354E-09, -1.725757E-07, +2.378517E-10, -9.549596E-11, -2.028054E-11, -9.499672E-11, -2.431012E-11, -3.160712E-12, -2.879781E-14, +2.327181E-13, +2.085189E-12, -6.830374E-13, -2.113676E-13])
_ka[2] = np.array([-5.741790E-05, +1.140810E-03, -8.599681E-05, +2.108325E-05, +2.941094E-05, +4.087327E-07, -1.599348E-07, +1.558857E-08, -2.030561E-07, +1.573384E-08, -1.081135E-10, +5.459438E-11, -2.647650E-12, -1.476344E-11, +1.651644E-11, -6.800286E-13, -7.283961E-13, +4.910761E-14, -4.495334E-13, +3.622077E-13, -1.006904E-12])
_kb[2] = np.array([+1.168688E-03, +6.795485E-04, +1.038700E-03, +5.767053E-06, +1.947397E-05, +3.492700E-05, -2.845218E-09, -2.057702E-07, -9.306303E-09, -1.744583E-07, -5.081412E-11, -2.867238E-11, -1.835628E-11, -2.576269E-11, -1.027837E-11, +1.565123E-13, +1.199605E-12, +4.593593E-13, +4.872015E-13, +6.879575E-13, -5.461638E-13])
_ka[3] = np.array([+5.368236E-03, +1.227243E-03, -2.090547E-04, -2.399988E-05, +3.384537E-05, -6.221500E-07, -4.867351E-09, +2.078667E-08, -1.638714E-07, +1.772616E-08, +2.996680E-10, -4.028182E-11, +5.767441E-11, +1.641023E-10, -1.128943E-10, -7.751036E-12, +7.377048E-13, -1.783869E-12, -1.278090E-12, +1.748810E-12, -9.794058E-13])
_kb[3] = np.array([+4.876272E-03, -3.383015E-03, -1.587592E-03, -9.295833E-06, -9.674543E-06, +2.349712E-05, -1.761487E-08, -1.578380E-07, +2.460535E-08, -1.182984E-07, +1.890677E-10, -3.246752E-11, +4.169965E-11, +6.119891E-12, +6.183478E-11, +7.396466E-13, +9.370199E-13, +1.502810E-12, -2.389407E-13, -2.048480E-12, -2.115334E-12])
_ka[4] = np.array([+7.229849E-03, +2.662657E-03, -4.308010E-04, +3.110385E-05, +5.271226E-05, +1.301581E-05, -2.549827E-07, +1.120662E-08, -1.888047E-07, -1.372281E-08, -9.923747E-11, -4.245489E-10, -1.116774E-10, -8.149722E-11, -1.238051E-10, +5.009643E-12, +3.457558E-12, -4.395776E-12, -5.377011E-13, +3.728625E-12, +5.916105E-13])
_kb[4] = np.array([+5.199843E-03, +3.041185E-03, -2.016066E-03, -3.051650E-06, +2.050006E-05, +4.345048E-05, +1.209439E-07, -1.190427E-07, +3.520365E-08, -1.806912E-07, +4.478562E-12, -4.181739E-10, -2.695066E-10, -2.332946E-10, -1.768701E-11, -5.066513E-12, -5.723793E-13, +4.107264E-13, -2.581602E-12, -6.170103E-13, +7.363110E-13])
for kk in _ka.keys():
    _ka[kk][0]  = 0.0
    _ka[kk][1] += 1.0
    _kb[kk][0]  = 0.0
    _kb[kk][2] += 1.0

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Pixelscale in each channel:
_pxscale = {
    1   :   1.2232,
    2   :   1.2144,
    3   :   1.2247,
    4   :   1.2217,
}

## CRPIX pixel coordinate we use:
_aks_crpix1 = 128.0
_aks_crpix2 = 128.0

## Cryo/warm cutoff:
## From:
## --> divider_time = astt.Time('2011-01-01T00:00:00', format='isot', scale='utc')
## --> jdtdb_cutoff = divider_time.tdb.jd
_cryo_warm_cutoff = 2455562.5007660175      # JD (TDB)

def mission_from_jdtdb(jdtdb):
    return 'cryo' if jdtdb < _cryo_warm_cutoff else 'warm'

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Rotation matrix builder:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

## Matrix printer:
def mprint(matrix):
    for row in matrix:
        sys.stderr.write("  %s\n" % str(row))
    return

## Reflection matrices:
xref_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
yref_mat = np.array([[-1.0, 0.0], [0.0, 1.0]])
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array([[1.0, 0.0], [0.0, 1.0]])

## Radian-to-degree converter:
_radeg = 180.0 / np.pi

## Tangent projection:
def _tanproj(prj_xx, prj_yy):
    prj_rr = np.hypot(prj_xx, prj_yy)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    #prj_rr = np.sqrt(prj_xx**2 + prj_yy**2)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    useful = (prj_rr > 0.0)
    prj_theta = np.ones_like(prj_xx) * np.pi * 0.5
    prj_theta[useful] = np.arctan(np.degrees(1.0 / prj_rr[useful]))
    #prj_theta[useful] = np.arctan(_radeg / prj_rr[useful])
    #prj_phi = np.arctan2(prj_xx, prj_yy)
    prj_phi = np.arctan2(prj_xx, -prj_yy)
    #return prj_phi, prj_theta
    return np.degrees(prj_phi), np.degrees(prj_theta)

## Low-level WCS tangent processor:
def _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=False):
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    if debug:
        sys.stderr.write("%.3f < prj_xx < %.3f\n" % (prj_xx.min(), prj_xx.max()))
        sys.stderr.write("%.3f < prj_yy < %.3f\n" % (prj_yy.min(), prj_yy.max()))

    # Perform tangent projection:
    prj_phi, prj_theta = _tanproj(prj_xx, prj_yy)
    if debug:
        sys.stderr.write("%.3f < prj_theta < %.3f\n"
                % (prj_theta.min(), prj_theta.max()))
        sys.stderr.write("%.3f < prj_phi   < %.3f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Change variable names to avoid confusion:
    rel_ra, rel_de = prj_phi, prj_theta
    if debug:
        phi_range = prj_phi.max() - prj_phi.min()
        sys.stderr.write("phi range: %.4f < phi < %.4f\n" 
                % (prj_phi.min(), prj_phi.max()))

    # Shift to 
    old_fov = (0.0, 90.0, 0.0)
    new_fov = (crval1, crval2, 0.0)
    stuff = rfov.migrate_fov_deg(old_fov, new_fov, (rel_ra, rel_de))
    return stuff

## Convert X,Y to RA, Dec using CD matrix and CRVAL pair:
def xycd2radec(cdmat, xpix, ypix, crval1, crval2, debug=False):
    thisCD = np.array(cdmat).reshape(2, 2)
    rel_xx = xpix - _aks_crpix1
    rel_yy = ypix - _aks_crpix2
    relpix = np.array([xpix - _aks_crpix1, ypix - _aks_crpix2])
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)

## Convert X,Y to RA, Dec (single-value), uses position angle and fixed pixel scale:
def xypa2radec(pa_deg, xpix, ypix, crval1, crval2, channel, debug=False):
    pa_rad = np.radians(pa_deg)
    rel_xx = xpix - _aks_crpix1
    rel_yy = ypix - _aks_crpix2
    relpix = np.array([xpix - _aks_crpix1, ypix - _aks_crpix2])
    pscale = _pxscale[channel]
    #sys.stderr.write("pscale: %.4f\n" % pscale)
    #rotmat = rotation_matrix(pa_rad)
    ##sys.stderr.write("rotmat:\n")
    ##mprint(rotmat)
    #rscale = (pscale / 3600.0) * rotmat
    #sys.stderr.write("rscale:\n")
    #mprint(rscale)
    thisCD = np.matmul(xflip_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
    #thisCD = np.dot(ident_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
    #thisCD = np.matmul(ident_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
    if debug:
        sys.stderr.write("thisCD:\n")
        mprint(thisCD)
    #rel_ra, rel_de = np.dot(thisCD, relpix)

    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)

    ##prj_xx, prj_yy = np.matmul(thisCD, relpix)
    ###sys.stderr.write("prj_xx: %s\n" % str(prj_xx))
    ###sys.stderr.write("prj_yy: %s\n" % str(prj_yy))
    ###prj_rr = np.sqrt(prj_xx, prj_yy)
    ##if debug:
    ##    sys.stderr.write("%.3f < prj_xx < %.3f\n" % (prj_xx.min(), prj_xx.max()))
    ##    sys.stderr.write("%.3f < prj_yy < %.3f\n" % (prj_yy.min(), prj_yy.max()))

    ##prj_phi, prj_theta = _tanproj(prj_xx, prj_yy)

    ##if debug:
    ##    sys.stderr.write("%.3f < prj_theta < %.3f\n"
    ##            % (prj_theta.min(), prj_theta.max()))
    ##    sys.stderr.write("%.3f < prj_phi   < %.3f\n"
    ##            % (prj_phi.min(), prj_phi.max()))
    ###prj_rr = np.hypot(prj_xx, prj_yy)
    ###useful = (prj_rr > 0.0)
    ###prj_theta = np.ones_like(xpix) * np.pi * 0.5
    ###prj_theta[useful] = np.arctan(np.radians(prj_rr[useful]))
    ####prj_phi = np.arctan2(prj_xx, -prj_yy)
    ###prj_phi = np.arctan2(prj_xx, prj_yy)
    ###sys.stderr.write("prj_theta: %s\n" % str(prj_theta))
    ###sys.exit(0)
    ###rel_ra, rel_de = np.matmul(thisCD, relpix)
    ##rel_ra, rel_de = prj_phi, prj_theta
    ###derp_ra = thisCD[0][0] * rel_xx + thisCD[0][1] * rel_yy
    ###derp_de = thisCD[1][0] * rel_xx + thisCD[1][1] * rel_yy
    ###sys.stderr.write("rel_ra: %s\n" % str(rel_ra))
    ###sys.stderr.write("drp_ra: %s\n" % str(derp_ra))
    ###sys.stderr.write("rel_de: %s\n" % str(rel_de))
    ###sys.stderr.write("drp_de: %s\n" % str(derp_de))
    ##if debug:
    ##    phi_range = prj_phi.max() - prj_phi.min()
    ##    sys.stderr.write("phi range: %.4f < phi < %.4f\n" 
    ##            % (prj_phi.min(), prj_phi.max()))
    ###old_fov = (0.0, 0.0, 0.0)
    ###old_fov = (180.0, 90.0, 0.0)
    ##old_fov = (0.0, 90.0, 0.0)
    ##new_fov = (crval1, crval2, 0.0)
    ##stuff = rfov.migrate_fov_deg(old_fov, new_fov, (rel_ra, rel_de))
    ###stuff = rfov.migrate_fov_deg(old_fov, new_fov, (prj_phi, prj_theta))
    ###sys.stderr.write("stuff: %s\n" % str(stuff))
    ### for speed, try:
    ### stuff = rfov.roll_sky_deg(rel_ra, rel_de, crval1, crval2)
    ##return stuff
    ###return rel_ra + crval1, rel_de + crval2

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Expose through class:
class AKSPoly(object):

    def __init__(self):
        self._dpar_a = _ka
        self._dpar_b = _kb
        self._crpix1 = _aks_crpix1
        self._crpix2 = _aks_crpix2
        self._ppar_a = _ppa
        self._ppar_b = _ppb
        self._missions = list(self._ppar_a.keys())
        self._cwcutoff = _cryo_warm_cutoff
        return

    def xy2focal(self, xpix, ypix, channel):
        """Convert X,Y detector positions (IN PIXEL UNITS!!!) to focal
        plane coordinates using Adam Kraus' solution."""
        prev_shape = xpix.shape
        xrel = xpix.flatten() - self._crpix1
        yrel = ypix.flatten() - self._crpix2
        xpar = self._dpar_a[channel]
        ypar = self._dpar_b[channel]
        xfoc = self._eval_axis(xrel, yrel, xpar)
        yfoc = self._eval_axis(xrel, yrel, ypar)
        #import pdb; pdb.set_trace()
        #xfoc = self._crpix1 + self._eval_axis(xrel, yrel, xpar)
        #yfoc = self._crpix2 + self._eval_axis(xrel, yrel, ypar)
        #return xrel + xfoc.reshape(prev_shape), yrel + yfoc.reshape(prev_shape)
        return xfoc.reshape(prev_shape), yfoc.reshape(prev_shape)

    def xform_xy(self, xpix, ypix, channel):
        """Convert X,Y detector positions (IN PIXEL UNITS!!!) to modified
        X',Y' (truer?) pixel coordinates using Adam Kraus' solution."""
        xfoc, yfoc = self.xy2focal(xpix, ypix, channel)
        newx = self._crpix1 + xfoc
        newy = self._crpix2 + yfoc
        return newx, newy

    def dephase(self, xpix, ypix, mission, channel):
        if not self._have_dephase_pars(mission, channel):
            sys.stderr.write("Unsupported mission/channel combo: %s/%d\n"
                    % (mission, channel))
            return None, None
        #prev_shape = xpix.shape
        #xrel = (xpix - np.floor(xpix)).flatten() - 0.5
        #yrel = (ypix - np.floor(ypix)).flatten() - 0.5
        xrel = (xpix - np.floor(xpix)) - 0.5
        yrel = (ypix - np.floor(ypix)) - 0.5
        xpar = self._ppar_a[mission][channel]
        ypar = self._ppar_b[mission][channel]
        newx = xpix + self._eval_axis(xrel, yrel, xpar)
        newy = ypix + self._eval_axis(xrel, yrel, ypar)
        return newx, newy

    def _have_dephase_pars(self, mission, channel):
        if not mission in self._missions:
            sys.stderr.write("Unsupported mission: %s\n" % mission)
            sys.stderr.write("Available options: %s\n" % str(self._missions))
            return False
        if not channel in self._ppar_a[mission].keys():
            sys.stderr.write("No channel %d for %s mission!\n"
                    % (channel, mission))
            return False
        return True

    @staticmethod
    def _eval_axis(x, y, pp):
        newpix  = np.zeros_like(x)
        newpix += pp[0]
        newpix += pp[1]*x + pp[2]*y
        newpix += pp[3]*x**2 + pp[4]*x*y + pp[5]*y**2
        newpix += pp[6]*x**3 + pp[7]*x**2*y + pp[8]*x*y**2 + pp[9]*y**3
        newpix += pp[10]*x**4 + pp[11]*x**3*y + pp[12]*x**2*y**2 \
                            + pp[13]*x*y**3 + pp[14]*y**4
        newpix += pp[15]*x**5 + pp[16]*x**4*y + pp[17]*x**3*y**2 \
                + pp[18]*x**2*y**3 + pp[19]*x*y**4 + pp[20]*y**5
        return newpix

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##

######################################################################
# CHANGELOG (akspoly.py):
#---------------------------------------------------------------------
#
#  2022-09-14:
#     -- Increased __version__ to 0.4.0.
#     -- Added dephasing parameters for warm and cryo missions.
#
#  2019-11-10:
#     -- Increased __version__ to 0.3.0.
#     -- Fixed critical error in parameter list causing gross miscalculation
#           of Y-coordinates. Module should work as intended now.
#
#  2019-09-05:
#     -- Increased __version__ to 0.2.0.
#     -- Older xform_xy() now relies on xy2focal().
#     -- Added xy2focal() method to produce focal plane coordinates. This new
#           routine also handles flatten/reshape internally to better mimic
#           the API of astropy WCS routines.
#
#  2019-09-04:
#     -- Increased __version__ to 0.1.0.
#     -- First created akspoly.py.
#

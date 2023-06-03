#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
#    Straightforward RA, Dec manipulation with 3-D rotation matrices and
# minimal (numpy) dependencies.
#
# Rob Siverd
# Created:       2018-02-21
# Last modified: 2019-01-30
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.2"

## Modules:
import os
import sys
import time
import numpy as np

##--------------------------------------------------------------------------##



##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## 3D rotation class:
class Rotate3D(object):

    """
    3-D rotation matrices for Cartesian coordinates. The sense of rotation is
    (I believe) such that positive rotation angles turn data COUNTER-CLOCKWISE
    as seen by an observer at +infinity on the specified axis.
    
    Rotation methods for each axis:
    --> xrot(angle_radians, xyz_list)
    --> yrot(angle_radians, xyz_list)
    --> zrot(angle_radians, xyz_list)

    NOTES:
    * all angles are specified in RADIANS
    * xyz_list can be made from individual arrays by np.vstack((x, y, z))

    """
    def __init__(self):
        return

    # ----------------------------------
    # Rotation matrices (radians):
    @staticmethod
    def _calc_xrot_rad(ang):
        return np.matrix([[  1.0,          0.0,          0.0],
                          [  0.0,  np.cos(ang), -np.sin(ang)],
                          [  0.0,  np.sin(ang),  np.cos(ang)]])
    @staticmethod
    def _calc_yrot_rad(ang):
        return np.matrix([[ np.cos(ang),   0.0,  np.sin(ang)],
                          [         0.0,   1.0,          0.0],
                          [-np.sin(ang),   0.0,  np.cos(ang)]])
 
    @staticmethod
    def _calc_zrot_rad(ang):
        return np.matrix([[ np.cos(ang), -np.sin(ang),   0.0],
                          [ np.sin(ang),  np.cos(ang),   0.0],
                          [         0.0,          0.0,   1.0]])

    # How to rotate an array of vectors:
    def rotate_xyz(rmatrix, xyz_list):
        return np.dot(rmatrix, xyz_list)

    def xrotate_xyz(self, ang_rad, xyz_list):
        rmatrix = self._calc_xrot_rad(ang_rad)
        return np.dot(rmatrix, xyz_list)
 
    def yrotate_xyz(self, ang_rad, xyz_list):
        rmatrix = self._calc_yrot_rad(ang_rad)
        return np.dot(rmatrix, xyz_list)
    
    def zrotate_xyz(self, ang_rad, xyz_list):
        rmatrix = self._calc_zrot_rad(ang_rad)
        return np.dot(rmatrix, xyz_list)
    
    def xrot(self, ang_rad, xyz_list):
        return self.xrotate_xyz(ang_rad, xyz_list)
    
    def yrot(self, ang_rad, xyz_list):
        return self.yrotate_xyz(ang_rad, xyz_list)
    
    def zrot(self, ang_rad, xyz_list):
        return self.zrotate_xyz(ang_rad, xyz_list)

    def make_zyx_rmatrix(self, angles):
        """Dot together Rz * Ry * Rx"""
        tz, ty, tx = angles
        Rz = self._calc_zrot_rad(tz)
        Ry = self._calc_yrot_rad(ty)
        Rx = self._calc_xrot_rad(tx)
        return np.dot(Rz, np.dot(Ry, Rx))

##--------------------------------------------------------------------------##
## FOV manipulation with rotation matrices:
class RotateFOV(object):

    """
    NOTES:
    x-axis points to RA,DE =  0.0,  0.0 deg
    y-axis points to RA,DE = 90.0,  0.0 deg
    z-axis points to RA,DE =    *, 90.0 deg
    """

    # ----------------------------------
    def __init__(self):
        self.r3d = Rotate3D()
        return

    # ----------------------------------
    # Reference frame transformations:
    def deg_rade2xyz(self, ra_d, de_d):
        return self._equ2xyz((np.radians(ra_d), np.radians(de_d)))

    def rad_rade2xyz(self, ra_r, de_r):
        return self._equ2xyz((ra_r, de_r))

    # Convert RA/DE to Cartesian:
    @staticmethod
    def _equ2xyz(equ_pts):
        """RA / DE in radians."""
        ra, de = equ_pts
        x = np.cos(de) * np.cos(ra)
        y = np.cos(de) * np.sin(ra)
        z = np.sin(de)
        return np.vstack((x, y, z))

    # Convert Cartesian to RA/DE:
    @staticmethod
    def _xyz2equ(xyz_pts):
        # Shape/dimension sanity check:
        if ((xyz_pts.ndim != 2) or (xyz_pts.shape[0] != 3)):
            sys.stderr.write("XYZ points have wrong shape!\n")
            return (0,0)
        tx = np.array(xyz_pts[0]).flatten()
        ty = np.array(xyz_pts[1]).flatten()
        tz = np.array(xyz_pts[2]).flatten()
        ra = np.arctan2(ty, tx)
        de = np.arcsin(tz)
        return (ra, de)
        #return np.vstack((ra, de))

    # ----------------------------------
    # 3-D FOV migration:
    def _fov_migration_matrix(self, coo1, coo2):
        roll_to_origin = self._rotation_to_origin(coo1) # return to origin
        roll_to_target = self._rotation_to_target(coo2) # rotate to target
        return np.dot(roll_to_target, roll_to_origin)

    def _rotation_to_origin(self, fov1):
        """Create rotation matrix that moves FOV (RA, DE, PA) 
        from its current location back to origin (0, 0, 0)."""
        ra1, de1, pa1 = fov1
        Rz1 = self.r3d._calc_zrot_rad(-ra1)     # rotate 'back' to RA = 0
        Ry1 = self.r3d._calc_yrot_rad( de1)     # rotate 'back' to Dec = 0
        Rx1 = self.r3d._calc_xrot_rad( pa1)     # rotate FOV to PA = 0
        return np.dot(Rx1, np.dot(Ry1, Rz1))    # note XYZ order

    def _rotation_to_target(self, fov2):
        """Create rotation matrix that moves FOV (RA, DE, PA) 
        from origin (0, 0, 0) to target."""
        ra2, de2, pa2 = fov2
        Rx2 = self.r3d._calc_xrot_rad(-pa2)     # rotate to target PA
        Ry2 = self.r3d._calc_yrot_rad(-de2)     # rotate to target Dec
        Rz2 = self.r3d._calc_zrot_rad( ra2)     # rotate to target RA 
        return np.dot(Rz2, np.dot(Ry2, Rx2))    # note ZYX order

    # ----------------------------------
    # On-sky coordinate rolling:
    #def roll_sky_deg(self, ra_roll_d, de_roll_d, pa_roll_d):
    #def roll_rade_deg(self, old_ra, old_de, ra_roll_d, de_roll_d):

    def roll_sky_deg(self, old_ra, old_de, ra_roll_d, de_roll_d):
        old_xyz = self.deg_rade2xyz(old_ra, old_de)
        rangles = np.radians([ra_roll_d, -de_roll_d, 0.0])
        #rangles = np.radians([ra_roll_d, -de_roll_d, -pa_roll_d])
        rmatrix = self.r3d.make_zyx_rmatrix(rangles)
        new_xyz = np.dot(rmatrix, old_xyz)
        return np.degrees(self._xyz2equ(new_xyz))

    def migrate_fov_rad(self, fov1, fov2, rade_coords):
        old_xyz = self._equ2xyz(rade_coords)
        rmatrix = self._fov_migration_matrix(fov1, fov2)
        new_xyz = np.dot(rmatrix, old_xyz)
        return self._xyz2equ(new_xyz)

    def migrate_fov_deg(self, fov1, fov2, coords):
        rfov1, rfov2, rcoords = [np.radians(x) for x in [fov1, fov2, coords]]
        newRA, newDE = self.migrate_fov_rad(rfov1, rfov2, rcoords)
        return (np.degrees(newRA), np.degrees(newDE))

    # ----------------------------------
    # Shortcuts for common operations:
    def roll_to_origin_deg(self, fov, coords):
        origin = (0.0, 0.0, 0.0)
        return self.migrate_fov_deg(fov, origin, coords)

######################################################################
# CHANGELOG (skyrotation.py):
#---------------------------------------------------------------------
#
#  2019-01-30:
#     -- Increased __version__ to 0.2.2.
#     -- Added docstring to Rotate3D() that lists methods and specifies units.
#
#  2018-02-23:
#     -- Increased __version__ to 0.2.1.
#     -- Added shortcut method roll_to_origin_deg().
#     -- Added a few comments.
#
#  2018-02-22:
#     -- Increased __version__ to 0.2.0.
#     -- Implemented working version of RotateFOV() class.
#
#  2018-02-21:
#     -- Increased __version__ to 0.1.0.
#     -- Implemented Rotate3D() class, started work on celestial components.
#     -- First created skyrotation.py.
#

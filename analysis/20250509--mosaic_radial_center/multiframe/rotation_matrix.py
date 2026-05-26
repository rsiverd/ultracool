#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module contains classes to perform 3-D rotations in both the
# active (vector) and passive (frame) sense. Naming conventions are
# chosen to (hopefully) minimize confusion when I use this.
#
# NOTES:
# * Passive/frame rotations differ from active/vector rotations by a sign.
# * Active/vector rotation is also referred to as 'extrinsic' rotation. This
#       operation corresponds to the rotation of a vector about axes in some
#       external reference frame. This is often useful in computer graphics
#       and/or rendering where an existing object is reoriented in the fixed
#       coordinates of the screen or viewport. 
# * Passive/frame rotation is also referred to as 'intrinsic' rotation. This
#       operation corresponds to rotation of coordinate axes while leaving
#       some vector quantity unchanged. The result is the original vector 
#       quantity expressed in the new (rotated) coordinate system. 
#
# Rob Siverd
# Created:       2026-05-26
# Last modified: 2026-05-26
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Modules:
import math
import numpy as np

##--------------------------------------------------------------------------##
## Convention 1: active rotation (a.k.a. vector rotation):
class ActiveRotation(object):
    def __init__(self):
        return

    # 3-D rotation about the X-axis (RADIANS):
    @staticmethod
    def Rx(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0,   c,  -s],
                         [0.0,   s,   c]])

    # 3-D rotation about the Y-axis (RADIANS):
    @staticmethod
    def Ry(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[  c, 0.0,   s],
                         [0.0, 1.0, 0.0],
                         [ -s, 0.0,   c]])

    # 3-D rotation about the Z-axis (RADIANS):
    @staticmethod
    def Rz(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[  c,  -s, 0.0],
                         [  s,   c, 0.0],
                         [0.0, 0.0, 1.0]])

VectorRotation = ActiveRotation

##--------------------------------------------------------------------------##
## Convention 2: passive rotation (a.k.a. frame rotation):
class PassiveRotation(object):
    def __init__(self):
        return

    # 3-D rotation about the X-axis (RADIANS):
    @staticmethod
    def Rx(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0,   c,   s],
                         [0.0,  -s,   c]])

    # 3-D rotation about the Y-axis (RADIANS):
    @staticmethod
    def Ry(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[  c, 0.0,  -s],
                         [0.0, 1.0, 0.0],
                         [  s, 0.0,   c]])

    # 3-D rotation about the Z-axis (RADIANS):
    @staticmethod
    def Rz(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[  c,   s, 0.0],
                         [ -s,   c, 0.0],
                         [0.0, 0.0, 1.0]])

FrameRotation = PassiveRotation

######################################################################
# CHANGELOG (rotation_matrix.py):
#---------------------------------------------------------------------
#
#  2026-05-26:
#     -- Increased __version__ to 0.1.0.
#     -- First created rotation_matrix.py.
#

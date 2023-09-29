#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
#    This module provides simple routines to read, write, and use DS9 region
# files. These are particularly useful for quick inspection of SEP extraction
# and custom astrometric fitting results.
#
# Rob Siverd
# Created:       2018-02-14
# Last modified: 2018-07-14
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.8"

## Modules:
import os
import sys
import time
import numpy as np

##--------------------------------------------------------------------------##
##                      Miscellaneous Configuration:                        ##
##--------------------------------------------------------------------------##

colorset = ['red', 'magenta', 'green', 'blue', 'cyan', 'orange', 'yellow']


##--------------------------------------------------------------------------##
##                      Quick region file creation:                         ##
##--------------------------------------------------------------------------##

## Make region files on sky:
def regify_sky(save_file, ra, dec, rdeg=0.0015, colors=None, vlevel=0):
    if (vlevel >= 1):
        sys.stderr.write("Saving region '%s' ... " % save_file)
    xtra = ""
    with open(save_file, 'w') as f:
        #if color:
        #    f.write("global color=%s\n" % color)
        for i,(sx,sy) in enumerate(zip(ra, dec)):
            if isinstance(colors, list):
                xtra = "color=%s" % colors[i % len(colors)]
            f.write("fk5; circle(%10.4fd, %10.4fd, %.5fd) # %s\n" \
                    % (sx, sy, rdeg, xtra))
    if (vlevel >= 1):
        sys.stderr.write("done.\n")
    return

## Make region file in CCD frame:
def regify_ccd(save_file, xpix, ypix, rpix=8, colors=None, vlevel=0, origin=1):
    if (vlevel >= 1):
        sys.stderr.write("Saving region '%s' ... " % save_file)
    xtra = ""
    with open(save_file, 'w') as f:
        #if color:
        #    f.write("global color=%s\n" % color)
        for i,(sx,sy) in enumerate(zip(xpix, ypix)):
            sx, sy = sx + 1.0 - origin, sy + 1.0 - origin
            if isinstance(colors, list):
                xtra = "color=%s" % colors[i % len(colors)]
            f.write("image; circle(%8.3f, %8.3f, %.2f) # %s\n" \
                    % (sx, sy, rpix, xtra))
    if (vlevel >= 1):
        sys.stderr.write("done.\n")
    return

## Produce region file with line segments:
def regify_with_lines(rfile, objdata, idxpairs, colors=None, frame='image',
        origin=1.0):
    _allowed_frames = ['image', 'fk5']
    if not frame in _allowed_frames:
        sys.stderr.write("Unsupported region frame: '%s'\n" % frame)
        sys.stderr.write("Supported values: %s\n" % str(_allowed_frames))
        return False
    with open(rfile, 'w') as f:
        for i,(idxbri, idxfnt) in enumerate(idxpairs):
            xbri, ybri, _ = objdata[idxbri]
            xfnt, yfnt, _ = objdata[idxfnt]
            if (frame == 'image'):
                xbri, ybri = xbri + 1.0 - origin, ybri + 1.0 - origin
                xfnt, yfnt = xfnt + 1.0 - origin, yfnt + 1.0 - origin
                pass
            if isinstance(colors, list):
                xtra = "color=%s" % colors[i % len(colors)]
                pass
            else:
                xtra = ""
                pass
            f.write("%s; line(%8.3f, %8.3f, %8.3f, %8.3f) # %s\n" \
                    % (frame, xbri, ybri, xfnt, yfnt, xtra))
            pass
        pass
    return True

## Regify with vectors:
def regify_ccd_segments(rfile, segments, origin=1.0):
    with open(rfile, 'w') as f:
        for i,segspec in enumerate(segments):
            #sys.stderr.write("segspec: %s\n" % str(segspec))
            xbri, ybri, ldist, angdeg, mdiff = segspec
            xbri, ybri = xbri + 1.0 - origin, ybri + 1.0 - origin
            pixdist = 10.0**ldist
            #xtra = "vector=1, color=%s" % colorset[i % len(colorset)]
            xtra = "vector=1, color=%s" % "green"
            f.write("image; vector(%8.3f, %8.3f, %8.3f, %8.3f) # %s\n" \
                    % (xbri, ybri, pixdist, angdeg, xtra))
            pass
        pass
    return

## Connect-the-dots region to illustrate source correspondence:
def regify_connect_dots(rfile, coords1, coords2, colors=None, frame=None,
        origin=1.0):
    _allowed_frames = ['image', 'fk5']
    if not frame in _allowed_frames:
        sys.stderr.write("Unsupported region frame: '%s'\n" % frame)
        sys.stderr.write("Supported values: %s\n" % str(_allowed_frames))
        return False
    with open(rfile, 'w') as f:
        for i,data in enumerate(zip(coords1, coords2)):
            coords = tuple([cc for star in data for cc in star])
            if (frame == 'image'):
                coords = [x+(1.0 - origin) for x in coords]
                pass
            if isinstance(colors, list):
                xtra = "color=%s" % colors[i % len(colors)]
                pass
            else:
                xtra = ""
                pass
            args = tuple([frame] + list(coords) + [xtra])
            f.write("%s; line(%8.3f, %8.3f, %8.3f, %8.3f) # %s\n" % args)
            pass
        pass
    return True

## Announce command to inspect region file(s):
def reg_announce(descr, imgfile, regions, stream=sys.stderr):
    rstring = " ".join(["-r "+x for x in regions])
    #stream.write("\n")
    stream.write("--------------------------------------------------\n")
    stream.write("%s:\n" % descr)
    stream.write("ztf --lin %s %s\n" % (imgfile, rstring))
    stream.write("--------------------------------------------------\n")
    stream.write("\n")



######################################################################
# CHANGELOG (region_utils.py):
#---------------------------------------------------------------------
#
#  2018-07-14:
#     -- Increased __version__ to 0.1.8.
#     -- Changed module name from regify to region_utils to avoid bad behavior
#           due to naming conflict with existing script.
#
#  2018-02-26:
#     -- Increased __version__ to 0.1.7.
#     -- Added origin=1.0 arguments to regify_connect_dots(),
#           regify_ccd_segments(), and regify_ccd().
#
#  2018-02-25:
#     -- Increased __version__ to 0.1.6.
#     -- regify_with_lines() now accepts origin=1.0 argument. This applies
#           a +1,+1 offset when origin=0 is given (for image frame).
#
#  2018-02-19:
#     -- Increased __version__ to 0.1.5.
#     -- Added regify_connect_dots() routine to illustrate source matches.
#
#  2018-02-14:
#     -- Increased __version__ to 0.1.0.
#     -- First created regify.py. Routines taken from segmatch.py.
#

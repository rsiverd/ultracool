#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#### vim: set fileencoding=utf-8 :
#
#    Robust slope/line fitting with the Theil-Sen estimator. Unweighted and
# weighted variants are provided.
#
# NOTES:
#    -- Implementation inspired by:
# http://waxworksmath.com/Authors/N_Z/Wilcox/BasicStatistics/Code/Chapter8/theil_sen.py
#
# Rob Siverd
# Created:       2013-12-12
# Last modified: 2019-05-24
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "1.4.2"

## Modules:
#import signal
import time
import numpy as np
from sys import stderr
#from functools import partial
#from multiprocessing import Pool
import resource

##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## Return index of array element nearest specified value:
def _argnear(vec, val):
    return (np.abs(vec - val)).argmin()

##--------------------------------------------------------------------------##
## Calculate weighted median:
#def _calc_weighted_median(svals, wvals):
#   order = svals.argsort()
#   tdata = svals[order]
#   #w_cum = wvals[order].cumsum()
#   w_cum = np.cumsum(wvals[order])
#   which = _argnear(w_cum, 0.5)
#   return tdata[which]

##--------------------------------------------------------------------------##
## Calculate and select positive differences:
#def _calc_diffs(xvals, yvals):
#   t1 = time.time()
#   n_pts = xvals.size
#   order = np.argsort(xvals)
#   x_srt = xvals[order]       # sorting somehow improves speed
#   y_srt = yvals[order]       # sorting somehow improves speed
#   t2 = time.time()
#   stderr.write("Sort vals: %.3f s\n" % (t2 - t1))
#   #v_idx = np.arange(n_pts)
#   #ii,jj = np.meshgrid(np.arange(xvals.size), np.arange(yvals.size))
#   ii,jj = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
#   #ii,jj = np.meshgrid(np.arange(n_pts), np.arange(n_pts))
#   xdiff = xvals[ii] - xvals[jj]
#   ydiff = yvals[ii] - yvals[jj]
#   t3 = time.time()
#   stderr.write("Grid vals: %.3f s\n" % (t3 - t2))
#   which = (xdiff > 0.0)
#   xdiff = xdiff[which]
#   ydiff = ydiff[which]
#   t4 = time.time()
#   stderr.write("Selection: %.3f s\n" % (t4 - t3))
#   #return np.vstack((xdiff, ydiff))
#   return (xdiff, ydiff)

##--------------------------------------------------------------------------##
## Driver routine for multi-threaded diffs calculation:
#def _multi_diff(xvec, yvec, cores=1):
#   pool = Pool(processes=cores)
#   to_run = np.array_split(xvec, cores)
#   result = pool.map(partial(_calc_diffs, yvals=yvec), to_run)
#   result = np.concatenate(result)
#   #print "len:",len(result)
#   #print "result:",result
#   #print "size:",result.size
#   #print "shape:",result.shape
#   #print "result[0]:",result[0]
#   return (result[0], result[1])

#fudge = 1.465


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## An example of sensible workaround for large data sets follows below (first
## developed for cde-synth-mag-zero-point.py). Steps are:
## * The user specifies a maximum number of data points to analyze in a single
##      chunk. Subdivision is necessary to avoid RAM overconsumption.
## * A 'stride' through the data is calculated from the ratio of the size of
##      the total data set to that of the max-size chunks.
## * Theil-Sen fit parameters are separately computed on each strided subset
##      and appended to a list.
## * Once all chunks are completed, perform a median across chunks to obtain
##      a (hopefully) more precise result.
##
## 
## Fit offset with Theil-Sen estimator (use chunks to avoid memory problems):
#max_tspts = 1500
#stride = int(np.sum(fit_me) / float(max_tspts))
#tsbig = []
#for i in range(stride):
#    tsbig.append(ts.linefit(fkmag[i::stride], fdiff[i::stride]))
#tsmod = np.median(tsbig, axis=0)


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Estimate memory footprint of this algorithm:
def guess_memsize_MB(npts, data_bytes=8.0, fudge=0.52, wei=False):
    # vectors (npts**1): order, x_srt, y_srt, x_vec, y_vec
    # 2dgrids (npts**2): ii, jj, ydiff, xdiff, which
    # biggest size seems to be:
    halfgrid_pnts = 0.5 * npts**2
    n_hgrids = 4.0 if wei else 2.125
    #stderr.write("hgrids: %.3f\n" % n_hgrids)
    max_data_pnts = n_hgrids * npts**2 + 3.0 * npts
    #max_data_pnts *= fudge      # extra factor (measured)
    max_data_megabytes = max_data_pnts * data_bytes / 1e6
    return max_data_megabytes

## Expected memory capacity increase:

## Sanity check on data set size:
def memory_checks_ok(npts, warn_MB, halt_MB, wei=False, data_bytes=8):
    """Pre-check memory needs. Warn/halt as needed. Inputs should be MB."""
    need_MB = guess_memsize_MB(npts, data_bytes=data_bytes, wei=wei)
    if need_MB > halt_MB:
        msg = "Theil-Sen memory critical: %.2f MB RAM needed!\n" % need_MB
        msg += "Aborting task!\n"
        stderr.write(msg)
        return False
    if need_MB > warn_MB:
        msg = "Theil-Sen memory warning: %.2f MB RAM needed!\n" % need_MB
        stderr.write(msg)
    return True

## Measure memory used so far:
def check_mem_usage_MB():
    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_kb_used / 1000.0

##--------------------------------------------------------------------------##
## Theil-Sen slope estimator:
#def TS_slope_noweight(x_vec, y_vec, sortfirst=True):
def TS_slope_noweight(x_vec, y_vec, timer=False, debug=False):
    """Estimate line slope with Theil-Sen estimator (unweighted median)."""

    # Prevent insane memory use:
    if not memory_checks_ok(x_vec.size, warn_MB=1e2, halt_MB=1e3, wei=False):
        return float("nan")

    # Note start time, sort inputs by x_vec:
    t1 = time.time()
    n_pts = x_vec.size
    order = np.argsort(x_vec)
    x_srt = x_vec[order]         # sorting somehow improves speed
    y_srt = y_vec[order]         # sorting somehow improves speed
    del order                    # free some RAM

    t2 = time.time()
    if timer:
        stderr.write("Sort vals: %.3f s\n" % (t2 - t1))

    # RAM-conscious evaluation of unique pairs:
    ii,jj = np.meshgrid(np.arange(n_pts), np.arange(n_pts), sparse=True)
    which = ((x_srt[ii] - x_srt[jj]) > 0.0)
    xdiff = np.float_((x_srt[ii] - x_srt[jj])[which])
    ydiff = np.float_((y_srt[ii] - y_srt[jj])[which])
    del ii, jj, x_srt, y_srt, which     # free memory!

    t3 = time.time()
    if timer:
        stderr.write("Selection: %.3f s\n" % (t3 - t2))

    # Get slope:
    slope = np.median(ydiff / xdiff)

    t4 = time.time()
    if timer:
        stderr.write("Get slope: %.3f s\n" % (t4 - t3))
    return slope

##--------------------------------------------------------------------------##
## Theil-Sen slope estimate using weighted median:
#def TS_slope_weighted(x_vec, y_vec, sortfirst=True):
def TS_slope_weighted(x_vec, y_vec, timer=False, debug=False):
    """Estimate line slope with Theil-Sen estimator (using weighted median)."""

    # Prevent insane memory use:
    if not memory_checks_ok(x_vec.size, warn_MB=1e2, halt_MB=1e3, wei=True):
        return float("nan")

    # Note start time, sort inputs by x_vec:
    t1 = time.time()
    n_pts = x_vec.size
    order = np.argsort(x_vec)
    x_srt = x_vec[order]         # sorting somehow improves speed
    y_srt = y_vec[order]         # sorting somehow improves speed
    del order                    # free some RAM

    t2 = time.time()
    if timer:
        stderr.write("Sort vals: %.3f s\n" % (t2 - t1))

    # RAM-conscious evaluation of unique pairs:
    ii,jj = np.meshgrid(np.arange(n_pts), np.arange(n_pts), sparse=True)
    which = ((x_srt[ii] - x_srt[jj]) > 0.0)
    xdiff = np.float_((x_srt[ii] - x_srt[jj])[which])
    ydiff = np.float_((y_srt[ii] - y_srt[jj])[which])
    del ii, jj, x_srt, y_srt, which     # free memory!

    t3 = time.time()
    if timer:
        stderr.write("Grid+selection: %.3f s\n" % (t3 - t2))

    # Slopes and weights:
    svals = ydiff / xdiff
    wvals = xdiff / xdiff.sum()   # weight by X-separation

    # Sort and cumulate:
    order = svals.argsort()
    svals = svals[order]
    w_cum = wvals[order].cumsum()
    which = _argnear(w_cum, 0.5)
    slope = svals[which]

    t4 = time.time()
    if timer:
        stderr.write("Get slope: %.3f s\n" % (t4 - t3))
    return slope

##--------------------------------------------------------------------------##
## Linear regression using Theil-Sen slope estimator:
#def linefit(x_vec, y_vec, weighted=False, joint=True, sortfirst=True):
def linefit(x_vec, y_vec, weighted=False, joint=True, timer=False):
    """
    Robustly fit line slope/intercept using Thiel-Sen estimator.

    Returns (intercept, slope).
    """

    # Robust slope fit:
    if weighted:
        slope = TS_slope_weighted(x_vec, y_vec, timer)
        #slope = TS_slope_weighted(x_vec, y_vec, sortfirst)
    else:
        slope = TS_slope_noweight(x_vec, y_vec, timer)
        #slope = TS_slope_noweight(x_vec, y_vec, sortfirst)

    # Matching intercept:
    if joint:
        icept = np.median(y_vec - slope * x_vec)
    else:
        icept = np.median(y_vec) - slope * np.median(x_vec)

    return np.array([icept, slope])


######################################################################
# CHANGELOG (theil_sen.py):
#---------------------------------------------------------------------
#
#  2019-05-24:
#     -- Increased __version__ to 1.4.2.
#     -- Added example of strided, chunk-wise Theil-Sen estimation that
#           seems to be a useful option for large data sets. This routine
#           is not yet available in this module but has been noted and
#           documented to be enabled in the future.
#
#  2017-05-21:
#     -- Increased __version__ to 1.4.1.
#     -- Removed memory reporting code within fitting routines. Memory
#           consumption behavior of these routines is now well understood.
#
#  2017-05-17:
#     -- Increased __version__ to 1.4.0.
#     -- Now use np.meshgrid() with sparse=True. This *significantly* reduces
#           the amount of memory required for fitting with no apparent change
#           in result.
#     -- Increased __version__ to 1.3.0.
#     -- Indenting is now 4 spaces.
#     -- Added memory use sanity check with warning and halting levels.
#     -- Added explicit del commands to minimize memory footprint:
#           --> 'order' removed after x_srt,y_srt created
#           --> remove ii, jj, x_srt, y_srt once xdiff,ydiff created
#     -- Added memory requirements estimation routine.
#
#  2014-09-25:
#     -- Increased __version__ to 1.2.5.
#     -- Commented out partial/Pool imports (not currently used).
#     -- Fixed wrong array name in non-joint intercept estimate.
#
#  2014-08-29:
#     -- Increased __version__ to 1.2.0.
#     -- Now return parameters in numpy array (compatibility).
#     -- Reversed order of returned parameters (now icept, slope) for
#           compatibility with other numpy/scipy routines.
#
#  2014-03-04:
#     -- Increased __version__ to 1.1.1.
#     -- Now timing information is only reported if timer=True.
#     -- Improved linefit() docstring (return values now indicated).
#
#  2014-01-09:
#     -- Increased __version__ to 1.1.0.
#     -- Eliminated sortfirst argument to linefit().
#     -- sortfirst is no longer optional for weighted TS slope fit.
#
#  2013-12-15:
#     -- Increased __version__ to 1.0.5.
#     -- sortfirst is no longer optional for unweighted TS slope fit.
#
#  2013-12-12:
#     -- Increased __version__ to 1.0.0.
#     -- First created theil_sen.py.
#

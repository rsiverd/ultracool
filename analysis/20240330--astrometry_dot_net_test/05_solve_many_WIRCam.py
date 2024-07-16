#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Solve a bunch of WIRCam images using positions from fcat files I already
# have lying around. The list of images to solve is taken from a file
# called 'fcat_paths.txt' in this directory.
#
# Rob Siverd
# Created:       2024-03-30
# Last modified: 2024-03-30
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
#import argparse
import shutil
#import glob
import gc
import os
import sys
import time
import pickle
import numpy as np
import signal
import random
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## WCS solutions from astroquery/ast.net:
from astroquery.astrometry_net import AstrometryNet
from astroquery.exceptions import TimeoutError

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    sys.stderr.write("failed to import extended_catalog module!")
    sys.exit(1)

## Make objects:
ast = AstrometryNet()
ccc = ec.ExtendedCatalog()

##--------------------------------------------------------------------------##
## Disable buffering on stdout/stderr:
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##
## Catch interruption cleanly:
def signal_handler(signum, frame):
    sys.stderr.write("\nInterrupted!\n\n")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

##--------------------------------------------------------------------------##

## Various from astropy:
#try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
#except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
#    sys.exit(1)

## astrometry.net from astroquery:
from astroquery.astrometry_net import AstrometryNet

##--------------------------------------------------------------------------##
##------------------       Process Configuration            ----------------##
##--------------------------------------------------------------------------##

## Where to stash solutions:
output_dir = 'solutions'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


##--------------------------------------------------------------------------##
##------------------       Load Image To-Do List            ----------------##
##--------------------------------------------------------------------------##

fcat_list_file = 'fcat_paths.txt'
with open(fcat_list_file, 'r') as flf:
    fcats_todo = [x.rstrip() for x in flf.readlines()]

## Randomize order if desired:
randomize = True
if randomize:
    random.shuffle(fcats_todo)

##--------------------------------------------------------------------------##
##------------------     Solution Store/Load Helpers        ----------------##
##--------------------------------------------------------------------------##

def stash_as_pickle(filename, thing):
    with open(filename, 'wb') as sapf:
        pickle.dump(thing, sapf)
    return

def load_pickled_object(filename):
    with open(filename, 'rb') as lpof:
        thing = pickle.load(lpof)
    return thing

##--------------------------------------------------------------------------##
##------------------          Solver Configuration          ----------------##
##--------------------------------------------------------------------------##

## Get polynomial order from command line:
#sys.stderr.write("len(sys.argv): %d\n" % len(sys.argv))
if len(sys.argv) < 2:
    sys.stderr.write("\nSyntax: %s poly_order\n\n" % sys.argv[0])
    sys.exit(1)
use_poly_order = int(sys.argv[1])
sys.stderr.write("Got polynomial order: %d\n" % use_poly_order)

## Image size:
nxpix, nypix = 2048, 2048

## List the allowed settings:
sys.stderr.write("\n------------------------------------------\n")
ast.show_allowed_settings()
sys.stderr.write("------------------------------------------\n\n")

## Solver settings:
ast.TIMEOUT = 300
timeout_sec = ast.TIMEOUT
#use_poly_order  = 2
solver_settings = {
        'parity'        :                 2,      # for CFHT
        'scale_units'   :     'arcminwidth',
        'scale_lower'   :               5.0,
        'scale_upper'   :              15.0,
        'tweak_order'   :    use_poly_order,
}

#sys.exit(0)
##--------------------------------------------------------------------------##
##------------------         Failures Load/Store            ----------------##
##--------------------------------------------------------------------------##

## Set failures file and ensure existence:
failures_file = 'failures.p%d.txt' % solver_settings['tweak_order']
if not os.path.isfile(failures_file):
    with open(failures_file, 'w'):
        pass

def load_failures():
    with open(failures_file, 'r') as ff:
        content = [x.rstrip() for x in ff.readlines()]
    return set([x for x in content if x.endswith('pickle')])

def append_failure(failure):
    with open(failures_file, 'a') as ff:
        ff.write("%s\n" % failure)
    return

##--------------------------------------------------------------------------##
##------------------       Submit/Solve with Retry          ----------------##
##--------------------------------------------------------------------------##

def submit_and_wait_for_solve(dets_x, dets_y, solver_settings={}):
    """
    dets_x          --   array of detection X positions
    dets_y          --   array of detection Y positions
    solver_settings --   dict of solver parameters
    """
    imsize_x = nxpix
    imsize_y = nypix

    try_again = True
    submission_id = None
    while try_again:
        pass


##--------------------------------------------------------------------------##
##------------------         Solve Listed Images            ----------------##
##--------------------------------------------------------------------------##

## Max detections to send up:
nsrc_max = 300
nsrc_min =  50

## Perform the solves:
total = len(fcats_todo)
#count = 0
ntodo = 0
nproc = 0
for count,fcat_path in enumerate(fcats_todo, 1):
    sys.stderr.write("\nSolving image %d of %d ...\n" % (count, total))
    fcat_base = os.path.basename(fcat_path)
    soln_base = '%s.p%d.pickle' % (fcat_base, solver_settings['tweak_order'])
    save_file = os.path.join(output_dir, soln_base)
    sys.stderr.write("save_file: %s\n" % save_file)
    if os.path.isfile(save_file):
        sys.stderr.write("Image already solved!\n")
        continue

    # update list of known failures:
    known_failures = load_failures()

    # skip anything known to have failed with these params:
    if save_file in known_failures:
        sys.stderr.write("Image already FAILED!\n")
        continue

    # solution needed:
    #sys.stderr.write("\n")
    nproc += 1

    # load catalog:
    ccc.load_from_fits(fcat_path)
    imcat = ccc.get_catalog()
    nsrcs = len(imcat)

    sys.stderr.write("Image catalog has %d sources.\n" % nsrcs)
    stars = imcat[:nsrc_max]
    nstar = len(stars)
    sys.stderr.write("Solving with %d brightest.\n" % nstar)
    if (nstar < nsrc_min):
        sys.stderr.write("Too few sources for solution ... skip!\n")
        continue

    try_again = True
    submission_id = None

    # attempt a solve:
    tik = time.time()
    try:
        wcs_header = ast.solve_from_source_list(imcat['x'], imcat['y'],
                nxpix, nypix, **solver_settings)
        tok = time.time()
        taken = tok - tik
        sys.stderr.write("done. Took %.3f seconds.\n" % taken)
    except TimeoutError as e:
        tok = time.time()
        taken = tok - tik
        submission_id = e.args[1]
        sys.stderr.write("timed out after %.3f seconds.\n" % taken)
        sys.stderr.write("Got submission_id: %d\n" % submission_id)
        sys.stderr.write("Updating failures file ... ")
        append_failure(save_file)
        sys.stderr.write("done.\n")
        continue
    except Exception as e:
        error_text = f"An error occurred: {e}"
        sys.stderr.write("%s\n" % error_text)
        sys.stderr.write("Problem solving image ... try next one.\n")
        continue

    # check for error:
    if not wcs_header:
        sys.stderr.write("SOLUTION FAILURE!\n")
        append_failure(save_file)
        continue

    # stuff answer into a pickle object:
    sys.stderr.write("Saving successful solution.\n")
    stash_as_pickle(save_file, wcs_header)

    # stop early if requested:
    if (ntodo > 0) and (nproc >= ntodo):
        break

sys.stderr.write("Image solving complete!\n")


######################################################################
# CHANGELOG (05_solve_many_WIRCam.py):
#---------------------------------------------------------------------
#
#  2024-03-30:
#     -- Increased __version__ to 0.1.0.
#     -- First created 05_solve_many_WIRCam.py.
#

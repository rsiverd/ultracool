#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Print best-fit whole-QRUNID results for incorporation into prior_fit.py.
#
# Rob Siverd
# Created:       2026-03-17
# Last modified: 2026-03-17
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
#import shutil
import glob
#import io
import gc
import os
import ast
import sys
import time
import pprint
#import pickle
#import vaex
#import calendar
#import ephem
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Solution parameter helpers:
import slv_par_tools
reload(slv_par_tools)
spt = slv_par_tools
quads = spt._quads

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

## Home-brew robust statistics:
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

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


## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

##--------------------------------------------------------------------------##
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Parse arguments and run script:
#class MyParser(argparse.ArgumentParser):
#    def error(self, message):
#        sys.stderr.write('error: %s\n' % message)
#        self.print_help()
#        sys.exit(2)
#
### Enable raw text AND display of defaults:
#class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
#                        argparse.RawDescriptionHelpFormatter):
#    pass
#
### Parse the command line:
#if __name__ == '__main__':
#
#    # ------------------------------------------------------------------
#    prog_name = os.path.basename(__file__)
#    descr_txt = """
#    PUT DESCRIPTION HERE.
#    
#    Version: %s
#    """ % __version__
#    parser = argparse.ArgumentParser(
#            prog='PROGRAM_NAME_HERE',
#            prog=os.path.basename(__file__),
#            #formatter_class=argparse.RawTextHelpFormatter)
#            description='PUT DESCRIPTION HERE.')
#            #description=descr_txt)
#    parser = MyParser(prog=prog_name, description=descr_txt)
#                          #formatter_class=argparse.RawTextHelpFormatter)
#    # ------------------------------------------------------------------
#    parser.set_defaults(thing1='value1', thing2='value2')
#    # ------------------------------------------------------------------
#    parser.add_argument('firstpos', help='first positional argument')
#    parser.add_argument('-w', '--whatever', required=False, default=5.0,
#            help='some option with default [def: %(default)s]', type=float)
#    parser.add_argument('-s', '--site',
#            help='Site to retrieve data for', required=True)
#    parser.add_argument('-n', '--number_of_days', default=1,
#            help='Number of days of data to retrieve.')
#    parser.add_argument('-o', '--output_file', 
#            default='observations.csv', help='Output filename.')
#    parser.add_argument('--start', type=str, default=None, 
#            help="Start time for date range query.")
#    parser.add_argument('--end', type=str, default=None,
#            help="End time for date range query.")
#    parser.add_argument('-d', '--dayshift', required=False, default=0,
#            help='Switch between days (1=tom, 0=today, -1=yest', type=int)
#    parser.add_argument('-e', '--encl', nargs=1, required=False,
#            help='Encl to make URL for', choices=all_encls, default=all_encls)
#    parser.add_argument('-s', '--site', nargs=1, required=False,
#            help='Site to make URL for', choices=all_sites, default=all_sites)
#    parser.add_argument('remainder', help='other stuff', nargs='*')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    #iogroup = parser.add_argument_group('File I/O')
#    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
#    #        help='Output filename', type=str)
#    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
#    #        help='KELT image with WCS')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    ofgroup = parser.add_argument_group('Output format')
#    fmtparse = ofgroup.add_mutually_exclusive_group()
#    fmtparse.add_argument('--python', required=False, dest='output_mode',
#            help='Return Python dictionary with results [default]',
#            default='pydict', action='store_const', const='pydict')
#    bash_var = 'ARRAY_NAME'
#    bash_msg = 'output Bash code snippet (use with eval) to declare '
#    bash_msg += 'an associative array %s containing results' % bash_var
#    fmtparse.add_argument('--bash', required=False, default=None,
#            help=bash_msg, dest='bash_array', metavar=bash_var)
#    fmtparse.set_defaults(output_mode='pydict')
#    # ------------------------------------------------------------------
#    # Miscellany:
#    miscgroup = parser.add_argument_group('Miscellany')
#    miscgroup.add_argument('--debug', dest='debug', default=False,
#            help='Enable extra debugging messages', action='store_true')
#    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
#            help='less progress/status reporting')
#    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
#            help='more progress/status reporting')
#    # ------------------------------------------------------------------
#
#    context = parser.parse_args()
#    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
#    context.prog_name = prog_name
#
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Extract RUNID from filename:
def runid_from_filename(filename):
    return os.path.basename(filename).split('_')[1]

## Load parameter set from file:
def load_parameters(filename):
    with open(filename, 'r') as fff:
        return ast.literal_eval(fff.read())

## Extract CD matrix and CRPIX from parameter list:
def get_cdm_crpix(parameters):
    return np.array(parameters[:24]).reshape(4, -1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## List of available J-only joint parameter files:
par_flist = sorted(glob.glob('joint_pars/jpars_??????_J.txt'))

runid_list = [runid_from_filename(x) for x in par_flist]
par_files = dict(zip(runid_list, par_flist))

## Load those files:
#raw_params = {kk:load_parameters(vv) for kk,vv in par_files.items()}
raw_params = {}
raw_inames = {} 
for runid,fname in par_files.items():
    raw_params[runid], raw_inames[runid] = load_parameters(fname)

## Params over time:
#par_stack = np.dstack([get_cdm_crpix(raw_params[x]) for x in runid_list])
#ne_pstack, nw_pstack, se_pstack, sw_pstack = par_stack

##--------------------------------------------------------------------------##
##------------------          fancy param formatter         ----------------##
##--------------------------------------------------------------------------##

def fancy_format(partext):
    # build start-of-line for NE adjustment:
    lines_with_nw = [x for x in partext.split('\n') if 'NW' in x]
    nw_line_start = lines_with_nw[0].split(':')[0]
    ne_line_start = nw_line_start.replace('NW', 'NE')
    # start of edits
    ptxt = partext.replace('array', 'np.array')
    ptxt = '{\n ' + ptxt.lstrip('{')        #  leading { on its own line
    ptxt = ptxt.rstrip('}') + ',\n}'        # trailing } on its own line
    ptxt = ptxt.replace('},\n', ',\n        },\n')    # cdmat/crpix end braces
    ## move NE elements to next line
    ptxt = ptxt.replace("{'NE'", '{\n'+ne_line_start)
    ## adjust indenting:
    ptxt = ptxt.replace("\n '", "\n    '")
    for qq in ['NE', 'NW', 'SE', 'SW']:
        ptxt = ptxt.replace("   '%s'"%qq, "'%s'"%qq)
    return ptxt

##--------------------------------------------------------------------------##
##------------------          dump params to screen         ----------------##
##--------------------------------------------------------------------------##

## Since we really only want the CD matrix and CRPIX items (CRVAL is specific
## to each image), we can just use the first 26 parameters from each par set.
## This keeps per-sensor CD matrix and CRPIX plus the CRVAL from the first
## image in the RUNID.
nkeep = 26

## Which ones we want to print:
selection = raw_params.keys()
#selection = ['18AQ15']
#selection = ['23AQ21']
#selection = ['17AQ07']

## Print everything:
for qrunid in selection:
    pars = np.array(raw_params.get(qrunid)[:nkeep])
    sys.stderr.write("\n%s\n" % fulldiv)
    sys.stderr.write("qrunid: %s\n" % qrunid)
    sifted = spt.sift_params(pars)
    #sys.stderr.write("%s\n" % str(sifted))
    #pprint.pprint(sifted, stream=sys.stderr)
    ptxt = fancy_format(pprint.pformat(sifted))
    #ptxt = '{\n ' + ptxt.lstrip('{')        #  leading { on its own line
    #ptxt = ptxt.rstrip('}') + ',\n}'        # trailing } on its own line

    # grab a copy of 
    #ptxt = ptxt.replace('}
    sys.stderr.write("%s\n" % ptxt)
    sys.stderr.write("\n")
    pass


#oldway = '%s %s' % ('one', 'two')
#newway = '{} {}'.format('one', 'two')

#all_data = append_fields(all_data, ('ra', 'de'), 
#         np.vstack((ra, de)), usemask=False)
#all_data = append_fields(all_data, cname, cdata, usemask=False)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##



######################################################################
# CHANGELOG (26_dump_joint_J_params.py):
#---------------------------------------------------------------------
#
#  2026-03-17:
#     -- Increased __version__ to 0.1.0.
#     -- First created 26_dump_joint_J_params.py.
#

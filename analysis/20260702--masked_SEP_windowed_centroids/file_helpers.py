#!/usr/bin/env python3
#
# A few small routines shared among scripts.
#
# -----------------------------------------------------------------------

import os
import sys
import glob
import time
#import astropy.io.fits as pf

##--------------------------------------------------------------------------##

## Method from filename:
def get_filemethod(filename):
    fbase = os.path.basename(filename)
    froot = fbase.split('.cat')[0]      # strip trailing .cat
    froot = froot.split('.fits')[0]     # strip trailing .fits
    return froot.split('psf_')[1]       # take the part after _

##--------------------------------------------------------------------------##

## List contents of folder (anything with 'psf' in it). 
## NOTE: this MUST return things in a list (not a dict keyed to method) 
## because multiple listed files with have the same method.
def get_psfdir_contents(psf_dir):
    if not os.path.isdir(psf_dir):
        sys.stderr.write("Error: folder not found: %s\n" % psf_dir)
        return {}
    return sorted(glob.glob('%s/*psf*' % psf_dir))

## Find image files in the PSF folder. Return as a list:
def get_psfdir_imglist(psf_dir):
    everything = get_psfdir_contents(psf_dir)
    return [x for x in everything if x.endswith('.fits')]
    #imgfiles = [x for x in everything if x.endswith('.fits')]
    #return {k:v for k,v in everything.items() if v.endswith('.fits')}

## Find image files in the PSF folder. Return as a dictionary:
def get_psfdir_imgdict(psf_dir):
    imgfiles = get_psfdir_imglist(psf_dir)
    return {get_filemethod(x):x for x in imgfiles}

## Find catalog files in the PSF folder. Return as a list:
def get_psfdir_catlist(psf_dir):
    everything = get_psfdir_contents(psf_dir)
    return [x for x in everything if x.endswith('.cat')]

## Find catalog files in the PSF folder. Return as a dictionary:
def get_psfdir_catdict(psf_dir):
    catfiles = get_psfdir_catlist(psf_dir)
    return {get_filemethod(x):x for x in catfiles}

### Find PSF images:
#psf_dir = 'PSF'
#if not os.path.isdir(psf_dir):
#    sys.stderr.write("Error: folder not found: %s\n" % psf_dir)
#    sys.exit(1)
#psf_files = sorted(glob.glob('%s/psf_*.fits' % psf_dir))
#psf_paths = {method_from_filename(x):x for x in psf_files}
#
### Load and normalize images:
#psf_data = {mm:pf.getdata(pp) for mm,pp in psf_paths.items()}
##psf_norm = {mm:(pp / np.sum(pp)) for mm,pp in psf_data.items()}

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## How to load SE catalog data:
def load_se_catalog(filename):
    with open(filename, 'r') as ff:
        content = [x.rstrip() for x in ff.readlines()]
    data = [x for x in content if not x.startswith('#')]
    if len(data) != 1:
        sys.stderr.write("Unexpected content!!\n")
        sys.exit(1)
    cols = ['XWIN_IMAGE', 'YWIN_IMAGE', 'MAG_ISO', 'FLUX_ISO',
            'FWHM_IMAGE', 'BACKGROUND', 'ELLIPTICITY', 'FLAGS', 'FLAGS_WIN']
    ctypes = [float]*7 + [int]*2
    values = [kk(xx) for kk,xx in zip(ctypes, data[0].split())]
    return dict(zip(cols, values))



#!/usr/bin/env python

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import angle
import robust_stats as rs
import astropy.time as astt
from numpy.lib.recfunctions import append_fields
from functools import partial

#reload(angle)

attach = partial(append_fields, usemask=False)

def dateobs2jd(date_obs):
    return astt.Time(date_obs, format='isot', scale='utc').jd

def fcat2cbcd(x):
    return x.replace('ucd_fcat', 'ucd_data').replace('fcat.fits', 'cbcd.fits')

## Targets of interest:
ntarg_ra = 277.12595833    # WISE1828
ntarg_de = +26.84355556    # WISE1828
starg_ra =  63.83208333    # 2M0415
starg_de =  -9.58497222    # 2M0415

toler = 5.0 / 60.0         # 5 arcmin

## Load header metadata:
csv_file = 'fcat_hdrs.csv'
_RA = 'RA_RQST'
_DE = 'DEC_RQST'
data = np.genfromtxt(csv_file, dtype=None, names=True, 
        delimiter=',', encoding=None)

## Append observation date/time as JD:
obstime = astt.Time(data['DATE_OBS'], format='isot', scale='utc')
data = attach(data, 'jdutc', obstime.jd)
del obstime

## Append file name variants:
cln_fcat = np.array([x.split('[')[0] for x in data['FILENAME']])
cln_cbcd = np.array([fcat2cbcd(x) for x in cln_fcat])
data = attach(data, ('fcat', 'cbcd'), (cln_fcat, cln_cbcd))
del cln_fcat, cln_cbcd

## Remove first-in-series exposures:
is_first = np.array([('_0000_0000_' in x) for x in data['cbcd']])
data = data[~is_first]
del is_first

## Sort images by observation date/time:
order = np.argsort(data['jdutc'])
data  = data[order]
del order
#rdata = data.copy() # before exclusions

## Exclude super-short exposures:
old_size = len(data)
#data  = rdata[(rdata['EXPTIME'] > 1.1)]
data  = data[(data['EXPTIME'] > 1.1)]
new_size = len(data)
sys.stderr.write("Droppe %d of %d sources with short exposures.\n"
        % (old_size - new_size, old_size))

north = data[_DE] > 0.0
ndata = data[north]
sdata = data[~north]

## Proximity to northern target:
nsep = angle.dAngSep(ntarg_ra, ntarg_de, ndata[_RA], ndata[_DE])
ssep = angle.dAngSep(starg_ra, starg_de, sdata[_RA], sdata[_DE])
ndata = append_fields(ndata, 'sep', nsep, usemask=False)
sdata = append_fields(sdata, 'sep', ssep, usemask=False)

nnear = (nsep < 0.03)
snear = (ssep < 0.03)

nkeep = ndata[nnear]
skeep = sdata[snear]

def qline(target):
    #keys = ['cln_fcat', 'cln_cbcd', 'jdutc', 'EXPTIME', _RA, _DE, 'sep']
    keys = ['fcat', 'cbcd', 'jdutc', 'EXPTIME', _RA, _DE, 'sep']
    vals = tuple([target[kk] for kk in keys])
    return "%s %s %16.7f %6.2f %12.7f %12.7f %7.4f" % vals


# Save nearby image lists for each object:
n_save = 'cat_wi1828.txt'
s_save = 'cat_2m0415.txt'
sys.stderr.write("Saving %s ... " % n_save)
with open(n_save, 'w') as ff:
    for thing in nkeep:
        ff.write("%s\n" % qline(thing))
sys.stderr.write("%s ... " % s_save)
with open(s_save, 'w') as ff:
    for thing in skeep:
        ff.write("%s\n" % qline(thing))
sys.stderr.write("done.\n") 



sys.exit(0)

fig = plt.figure(1, figsize=(16,8))
fig.clf()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#ax1.scatter(ndata[_RA], ndata[_DE], lw=0, s=2)
ax1.scatter(ndata[_RA], ndata[_DE], lw=0, s=2, c=ndata['jdutc'])
ax1.scatter(ntarg_ra, ntarg_de, c='r', s=30)

#ax2.scatter(sdata[_RA], sdata[_DE], lw=0, s=2)
ax2.scatter(sdata[_RA], sdata[_DE], lw=0, s=2, c=sdata['jdutc'])
ax2.scatter(starg_ra, starg_de, c='r', s=30)

fig.tight_layout()


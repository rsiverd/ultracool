#!/usr/bin/env python3
import os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import robust_stats as rs

import extended_catalog

ecl = extended_catalog.ExtendedCatalog()

some_fcat = 'wircam_H2_1319416p.fits.fz.fcat'

ecl.load_from_fits(some_fcat)

stars = ecl.get_catalog()

wxdiff = stars['wx'] - stars['x']
wydiff = stars['wy'] - stars['y']

dx_pctiles = np.percentile(wxdiff, [25, 50, 75])
dy_pctiles = np.percentile(wydiff, [25, 50, 75])

dx_med, dx_iqrn = rs.calc_ls_med_IQR(dx_pctiles)
dy_med, dy_iqrn = rs.calc_ls_med_IQR(dy_pctiles)

#dx_lab = 'med=%.2f, IQRN=%.3f' % (dx_med, dx_iqrn)
#dy_lab = 'med=%.2f, IQRN=%.3f' % (dy_med, dy_iqrn)
dx_lab = 'IQRN=%.3f' % (dx_iqrn)
dy_lab = 'IQRN=%.3f' % (dy_iqrn)

# Make a figure to illustrate both distributions:
fig = plt.figure(1, figsize=(11,5))
fig.clf()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

hopts = {'range':(-2, 2), 'bins':51}
hopts = {'range':(-1.5, 1.5), 'bins':51}

## Decorations:
def add_pctiles(ax, vals):
    p25, med, p75 = vals
    ax.axvline(p25, c='r', ls=':', label='25th pctile (%.3f)' % p25)
    ax.axvline(med, c='k', ls='--', label='median (%.3f)' % med)
    ax.axvline(p75, c='r', ls=':', label='75th pctile (%.3f)' % p75)

## X-difference distribution:
ax1.hist(wxdiff, label=dx_lab, **hopts)
ax1.set_xlabel('$\Delta$X (pixels)')
ax1.set_title("stars['wx'] - stars['x']")
add_pctiles(ax1, dx_pctiles)
ax1.legend(loc='upper left')

## Y-difference distribution:
ax2.hist(wydiff, label=dy_lab, **hopts)
ax2.set_xlabel('$\Delta$Y (pixels)')
ax2.set_title("stars['wy'] - stars['y']")
add_pctiles(ax2, dy_pctiles)
ax2.legend(loc='upper left')


fig.tight_layout()
fig.savefig('windowing_CFHT.png')


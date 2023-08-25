#!/usr/bin/env python3

import os, sys, time
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import append_fields

import robust_stats as rs

## Convenient, percentile-based plot limits:
def nice_limits(vec, pctiles=[1,99], pad=1.2):
    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
    middle = np.average(ends)
    return (middle + pad * (ends - middle))

## Six-panel plot illustrating astrometric fit:
def plot_ast_summary(targfit, fignum=1, ref_jdtdb=None, mas_lims=None):
    data = np.copy(targfit.dataset)
    data['calc_ra'] = data['calc_ra'] % 360.0   # just in case
    jd_ref = ref_jdtdb if ref_jdtdb else data['jdtdb'][0]
    jd_lab = "JD (TDB) - %.2f" % jd_ref
    mra_prmot, mde_prmot = targfit.get_radec_minus_prmot_mas(cos_dec_mult=True)
    mra_model, mde_model = targfit.get_radec_minus_model_mas(cos_dec_mult=True)
    data = append_fields(data, 
            ('reljd', 'mra_prmot', 'mde_prmot', 'mra_model', 'mde_model'), 
            (data['jdtdb']-jd_ref, mra_prmot, mde_prmot, mra_model, mde_model),
            usemask=False)
 
    # summary information:
    mra_stats = rs.calc_ls_med_IQR(mra_model)
    mde_stats = rs.calc_ls_med_IQR(mde_model)
    sys.stderr.write("RA residual (mas): %6.2f +/- %6.2f\n" % mra_stats)
    sys.stderr.write("DE residual (mas): %6.2f +/- %6.2f\n" % mde_stats)

    # lookup by filter:
    have_filters = np.unique(data['filter']).tolist()
    fdata = {x:data[data['filter'] == x] for x in have_filters}

    fmap = {'J':'green', 'H2':'cornflowerblue'}
    fig, axs = plt.subplots(3, 2, sharex=True, num=fignum, clear=True)
    #p_colors = [fmap[x] for x in data['filter']]
    #skw = {'lw':0, 's':5, 'c':p_colors}
    skw = {'lw':0, 's':5}
    for ax in axs.flatten():
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.patch.set_facecolor((0.8, 0.8, 0.8)) 

    # ----------------------------------------
    # RA, Dec raw sky positions:
    axra, axde = axs[0]
    for ff,subset in fdata.items():
        axra.scatter(subset['reljd'], subset['calc_ra'], 
                c=fmap[ff], label=ff, **skw)
        axde.scatter(subset['reljd'], subset['calc_de'],
                c=fmap[ff], label=ff, **skw)
    for ax in (axra, axde):
        ax.legend(loc='upper left')
        #ax.legend(loc='best')
    axra.set_ylabel("RA (deg)")
    axde.set_ylabel("Dec (deg)")

    # ----------------------------------------
    # RA, Dec residuals w.r.t. proper motion:
    axra, axde = axs[1]
    for ff,subset in fdata.items():
        axra.scatter(subset['reljd'], subset['mra_prmot'], 
                c=fmap[ff], label=ff, **skw)
        axde.scatter(subset['reljd'], subset['mde_prmot'],
                c=fmap[ff], label=ff, **skw)
    #axra.set_ylabel(r'$\Delta\alpha \dot cos\delta$ (mas)')
    #axde.set_ylabel(r'$\Delta\delta$ (mas)')

    # -----------------------------------------
    # RA, Dec residuals w.r.t. total model fit:
    axra, axde = axs[2]
    for ff,subset in fdata.items():
        axra.scatter(subset['reljd'], subset['mra_model'], 
                c=fmap[ff], label=ff, **skw)
        axde.scatter(subset['reljd'], subset['mde_model'],
                c=fmap[ff], label=ff, **skw)
    #axra.set_ylabel(r'$\Delta\alpha \dot cos\delta$ (mas)')
    #axde.set_ylabel(r'$\Delta\delta$ (mas)')

    # -----------------------------------------
    # Tweaks for the residual plots in mas units:
    for axra,axde in axs[1:]:
        axra.set_ylabel(r'$\Delta\alpha \dot cos\delta$ (mas)')
        axde.set_ylabel(r'$\Delta\delta$ (mas)')
    for rax in axs[1:].flatten():
        rax.axhline(0.0, c='k', ls='--', lw=0.5)
        rax.set_ylim(mas_lims)

    # Adjustments for all panels:
    for ax in axs.flatten():
        ax.legend(loc='upper left')

    # JD labels for the bottom row:
    for bax in axs[-1]:
        bax.set_xlabel(jd_lab)

    # Beautify:
    fig.tight_layout()
    plt.draw()
    return


#!/usr/bin/env python3

import os, sys, time
import gaia_match
import numpy as np

gm = gaia_match.GaiaMatch()
gm.load_sources_csv('/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/gaia_calib1_NE.csv')

gaia_epoch_jdutc = gm._ep_gaia.utc.jd

srcs = gm._srcdata

cos_dec  = np.cos(np.radians(srcs['dec'])).values
pm_tot   = np.hypot(srcs['pmdec'], cos_dec*srcs['pmra'])

#which    = np.argsort(pm_tot.values)[-5:]   # top 5
#fast_src = srcs.iloc[top_five]
which  = np.argsort(pm_tot.values)[-2:]   # top 2
movers = srcs.iloc[which]
single = movers.iloc[-1]

# -----------------------------------------------------------------------
# How to put Gaia data into parameter file:
_par_spec = (
        ('ra_deg', 'ra'),
        ('de_deg', 'dec'),
        ('pmra_cosdec_asyr', 'pmra'),
        ('pmde_asyr', 'pmdec'),
        ('parallax_as', 'parallax'),
        )

# How to produce a parameters file:
def make_params(name, epoch, gtarget):
    pars = {pp:gtarget[gg] for pp,gg in _par_spec}
    pars['name'] = name
    pars['epoch_jdutc'] = epoch
    return pars

def make_pars_text(name, epoch, pars, stream=sys.stderr):
    stream.write("""{
     'name'             : '%s',
     'ra_deg'           : %10.5f,
     'de_deg'           : %10.5f,
   'pmra_cosdec_asyr'   :   0.136,      # pmRA * cos(dec)
   'pmde_asyr'          :   0.317,
   'parallax_as'        :   0.070,
   'epoch_jdutc'        :   2455499.35,

}""" % (name, pars['ra_deg'], pars['de_deg']) )
    return

all_params = []
for ii,boom in movers.iterrows():
    sys.stderr.write("ii: %d\n" % ii) 
    sys.stderr.write("boom: %s\n" % boom) 

#[(pp,single[gg]) for pp,gg in _par_spec]
#pars_dict = {pp:single[gg] for pp,gg in _par_spec}
#pars_dict['name']

g125a_pars = make_params('G125A', gaia_epoch_jdutc, movers.iloc[0])
g125b_pars = make_params('G125B', gaia_epoch_jdutc, movers.iloc[1])


#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Load a vizquery result from the CADC back into python for inspection.
#
# Rob Siverd
# Created:       2023-05-23
# Last modified: 2023-05-23
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Astropy might be required ...
import astropy.table as apt  # just in case

## Data are pickled:
import pickle

## Load the file:
cadc_hits_file = 'cadc_hits.pickle'
with open(cadc_hits_file, 'rb') as chp:
    hits = pickle.load(chp)

## NOTE: this has 100 MB of hits ....

## What kind of producIDs do we have??
prodid_last_char = [x[-1] for x in hits['productID']]
unique_prod_char = list(set(prodid_last_char))
# Out[13]: ['p', 'y', 'm', 's', 'w', 'a', 'o', 'g', 'x']





import pandas as pd
import os, sys, time

pdata = pd.read_csv('/home/acolclas/final_wcs_params.csv')

median_vals = []
for thing in ['CD11', 'CD12', 'CD21', 'CD22']:
    medval = np.median(pdata[thing])
    sys.stderr.write("median %s = %15e\n" % (thing, medval))
    median_vals.append(medval)

med_cdmat = np.array(median_vals).reshape(2, 2)


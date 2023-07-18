#!/usr/bin/env python3

# Measure median CD matrix from Abby's reported solutions.

import os, sys, time
import numpy as np
import pandas as pd
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

##--------------------------------------------------------------------------##
## Quick ASCII I/O:
data_file = '20230629--final_wcs_params.csv'
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
all_data = pd.read_csv(data_file, **pdkwargs)

cd11 = all_data.CD11
cd12 = all_data.CD12
cd21 = all_data.CD21
cd22 = all_data.CD22

med_11 = np.median(cd11)
med_12 = np.median(cd12)
med_21 = np.median(cd21)
med_22 = np.median(cd22)

sys.stderr.write("median CD1_1: %e\n" % med_11)
sys.stderr.write("median CD1_2: %e\n" % med_12)
sys.stderr.write("median CD2_1: %e\n" % med_21)
sys.stderr.write("median CD2_2: %e\n" % med_22)


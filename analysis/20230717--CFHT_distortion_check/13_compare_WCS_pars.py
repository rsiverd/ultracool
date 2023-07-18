#!/usr/bin/env python3
#
# How different are the original and best-fit WCS parameters??

import os, sys, time
import numpy as np
import pandas as pd
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Load the fitted parameters:
data_file = '20230629--final_wcs_params.csv'
pdkwargs  = {'skipinitialspace':True, 'low_memory':False}
fit_data  = pd.read_csv(data_file, **pdkwargs)
f_ibase   = [os.path.basename(x) for x in fit_data['Image Name']]
f_itags   = [x.split('_')[-1].split('.')[0] for x in f_ibase]
fit_data['ibase'] = f_ibase
fit_data['itags'] = f_itags

#cd11 = all_data.CD11
#cd12 = all_data.CD12
#cd21 = all_data.CD21
#cd22 = all_data.CD22

## Load the original parameters:
data_file = 'orig_wcs_pars.csv'
pdkwargs  = {'skipinitialspace':True, 'low_memory':False}
wcs_data  = pd.read_csv(data_file, **pdkwargs)
w_ibase   = [os.path.basename(x) for x in wcs_data['FILENAME']]
w_itags   = [x.split('.')[0] for x in w_ibase]
wcs_data['ibase'] = w_ibase
wcs_data['itags'] = w_itags

## ----------------------------------------------------------------------- ##

## Which images are common to both sets?
common_tags = sorted(list(set(w_itags).intersection(set(f_itags))))

## Extract matching / ordered from each DataFrame:
hits_fit = fit_data.set_index('itags').loc[common_tags].reset_index()
hits_wcs = wcs_data.set_index('itags').loc[common_tags].reset_index()



#!/bin/bash

# list of extended catalogs:
ls /home/rsiverd/ucd_project/ucd_fcat/SPITZER_I*fcat.fits > fcat_list.txt

# time/coordinate header info:
imhget -E2 -l fcat_list.txt --progress -N --delim=',' \
   DATE_OBS EXPTIME RA_RQST DEC_RQST -o fcat_hdrs.csv


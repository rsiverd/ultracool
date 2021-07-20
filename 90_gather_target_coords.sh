#!/bin/bash
#
# This file contains commands to scrape name, RA, Dec from target .par files
# in order to produce targets_deg.txt.
#
# To run script, do (adjust file names as needed):
# ./90_gather_target_coords.sh --START   # inspect manually
# ./90_gather_target_coords.sh --START > targets_deg.txt

if [ "$1" != "--START" ]; then
   echo "Syntax: $0 --START"
   echo
   echo "If results look OK after visual inspection, do:"
   echo "$0 --START > targets_deg.txt"
   echo
   exit 1
fi

par_files=( `ls targets/*.par` )

#grep   name ${par_files[*]}
paste <(grep   name ${par_files[*]} | cut -d: -f3 | cut -d\' -f2) \
      <(grep ra_deg ${par_files[*]} | cut -d: -f3 | cut -d,  -f1) \
      <(grep de_deg ${par_files[*]} | cut -d: -f3 | cut -d,  -f1)




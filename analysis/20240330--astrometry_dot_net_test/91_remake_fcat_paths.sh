#!/bin/bash
#
# Remake this file list using _eph version of fcats:

dir_list=( $(cat fcat_paths.txt | xargs dirname | sort -u) )

tmpfile="scratch.$$.txt"
rm $tmpfile 2>/dev/null
for dd in ${dir_list[*]}; do
   echo "dd: $dd"
   ls $dd/wircam_*_eph.fits.fz.fcat >> $tmpfile

done

save_file="fcat_paths_v3.txt"
mv -vf $tmpfile $save_file


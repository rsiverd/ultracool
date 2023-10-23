#!/bin/bash

cfh_root="/home/rsiverd/ucd_project/ucd_cfh_data"
runids_dir="${cfh_root}/for_abby/calib1_p_NE/by_runid"

local_rdir="by_runid"
mkdir -p $local_rdir
#ls $runids_dir

runid_list=( `ls $runids_dir` )
echo "runid_list: ${runid_list[*]}"
rms_csv="run_match_subset.csv"

for runid in ${runid_list[*]}; do
   run_dir="${local_rdir}/${runid}"
   mkdir -p $run_dir
   #ls $runids_dir/$runid/run_*.csv
   cp -f $runids_dir/$runid/run_coeffs_?.csv $run_dir/.
   src_match_csv="${runids_dir}/${runid}/${rms_csv}"
   dst_match_csv="${run_dir}/${rms_csv}"
   if [[ -f $src_match_csv ]]; then
      echo "found: $src_match_csv"
      rm $dst_match_csv 2>/dev/null
      ln -s $src_match_csv $dst_match_csv
   fi
done



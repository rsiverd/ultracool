this_dir=$(pwd)
that_dir="/home/rsiverd/ucd_project/ucd_cfh_data/for_abby"

cmde "cd $that_dir"

keys="CD1_1 CD1_2 CD2_1 CD2_2 CRVAL1 CRVAL2"
cmde "imhget -N -d',' --progress -l image_list.txt $keys -o orig_wcs_pars.csv"
cmde "mv -f orig_wcs_pars.csv $this_dir/."

cmde "cd $this_dir"


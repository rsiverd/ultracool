
# First, get a list of all the images:
imlist="image_paths.txt"
cfh_dir="/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid"
cmde "find $cfh_dir -name 'wircam*.fz' | sort > $imlist"

# Next, gather header data from all of them:
hdrsave="image_headers.txt"
want_keys="DATE-OBS UTC-OBS QRUNID"
want_keys+=" TEMPERAT RELHUMID PRESSURE"  # weather items
cmde "imhget -N -l $imlist $want_keys --progress -o $hdrsave"


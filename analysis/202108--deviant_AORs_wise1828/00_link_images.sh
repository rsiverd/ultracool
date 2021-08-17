
imgs_dir="/home/rsiverd/ucd_project/ucd_targets/wise1828"
find $imgs_dir -type f -name "SPITZ*fits" > flist.tmp

grep -f <(awk '{print $1}' trouble.txt) flist.tmp | tee look_at_me.txt

first=$(head -1 look_at_me.txt)
first="/home/rsiverd/ucd_project/ucd_targets/wise1828/r46656768/SPITZER_I2_46656768_0004_0000_2_clean.fits"

ropts="--rdeg=0.0004 --Jmax=20 --sfrac=2.0"
cmde "make-region-overlay $ropts $first -o nearby.reg"


flztfs `cat look_at_me.txt`




data_root="/home/rsiverd/ucd_project/ucd_targets/wise1828"

find $data_root -type f -name "SPITZER_I2_*clean.fits.pcat" | sort > files_pcat.txt
find $data_root -type f -name "SPITZER_I2_*clean.fits.fcat" | sort > files_fcat.txt

# Gather headers:
img_hkeys="PA AORKEY"
cat_hkeys="GMATCHES GRADELTA GRASIGMA GDEDELTA GDESIGMA"
ghopts="-d, --progress -N"
for ctype in fcat pcat; do
   cat_list="files_${ctype}.txt"
   save_ihdrs="ihdrs_${ctype}.csv"
   save_chdrs="chdrs_${ctype}.csv"
   cmde "imhget $ghopts -l $cat_list -E IMGHEADER $img_hkeys -o $save_ihdrs"
   cmde "imhget $ghopts -l $cat_list -E CATALOG   $cat_hkeys -o $save_chdrs"
done



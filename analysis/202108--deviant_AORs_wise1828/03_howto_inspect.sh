
img_list=( `cat look_at_me.txt` )
rtype="sky"
rtype="pix"
nearby="nearby.reg"

for image in ${img_list[*]}; do
   ibase="${image##*/}"
   rfile="reg/${ibase}.${rtype}.reg"
   upath=$(echo $image | sed 's/clean/cbunc/')
   echo "image: $image"
   echo "upath: $upath"
   echo "ibase: $ibase"
   ctext="ztf -r $nearby -r $rfile $image $upath"
   echo "ctext: '$ctext'"
   eval $ctext
   echo
done


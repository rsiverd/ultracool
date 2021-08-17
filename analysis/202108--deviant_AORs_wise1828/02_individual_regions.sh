
mkdir -p reg

dfile="wise1828_dump.txt"

#imglist=( `awk '{print $1}' $dfile` )
exec 10<$dfile
while read iname xx yy ra de <&10; do
   echo "iname: $iname"
   pix_reg="reg/${iname}.pix.reg"
   echo "image; circle(${xx}, ${yy}, 1.0) # color=red" >  $pix_reg
   echo "image; circle(${xx}, ${yy}, 3.0) # color=red" >> $pix_reg
   sky_reg="reg/${iname}.sky.reg"
   echo "fk5; circle(${ra}d, ${de}d, 0.0010d) # color=red" >  $sky_reg
   echo "fk5; circle(${ra}d, ${de}d, 0.0005d) # color=red" >> $sky_reg
done
exec 10>&-




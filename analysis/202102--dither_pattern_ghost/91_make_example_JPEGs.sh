#!/bin/bash
#
# Create JPEGs to illustrate findings.
#
# Rob Siverd
# Created:      2021-02-18
# Last updated: 2021-02-23
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Default options:
debug=0 ; clobber=0 ; force=0 ; timer=0 ; vlevel=0
script_version="0.20"
this_prog="${0##*/}"
#shopt -s nullglob
# Propagate errors through pipelines: set -o pipefail
# Exit if uninitialized variable used (set -u): set -o nounset
# Exit in case of nonzero status (set -e): set -o errexit

## Program options:
#save_file=""
#shuffle=0
#confirmed=0

## Standard scratch files/dirs:
tmp_name="$(date +%Y%m%d.%H%M%S).$$.$(whoami)"
tmp_root="/tmp"
[ -d /dev/shm ] && [ -w /dev/shm ] && tmp_root="/dev/shm"
tmp_dir="$tmp_root"
#tmp_dir="$tmp_root/$tmp_name"
#mkdir -p $tmp_dir
foo="$tmp_dir/foo_$$.txt"
bar="$tmp_dir/bar_$$.txt"
baz="$tmp_dir/baz_$$.txt"
qux="$tmp_dir/qux_$$.txt"
jnk="$foo $bar $baz $qux"  # working copy
def_jnk="$jnk"             # original set
dir_cleanup='(echo -e "\nAutomatic clean up ... " ; cmde "rm -vrf $tmp_dir")'
jnk_cleanup='for X in $jnk ; do [ -f $X ] && cmde "rm -vf $X" ; done'
trap "$jnk_cleanup >&2" EXIT
##trap '[ -d $tmp_dir ] && cmde "rm -vrf $tmp_dir"' EXIT
#trap "[ -d $tmp_dir ] && $dir_cleanup >&2" EXIT
#trap "[ -d $tmp_dir ] && $dir_cleanup >&2; $jnk_cleanup >&2" EXIT
#trap 'oops=$? ; echo ; exit $oops' HUP INT TERM

## Required programs:
declare -a need_exec
need_exec+=( awk cat fpeg FuncDef ims2table montage sed tr )
#need_exec+=( shuf shuffle sort ) # for randomization
for need in ${need_exec[*]}; do
   if ! ( /usr/bin/which $need >& /dev/null ); then
      echo "Error:  can't find '$need' in PATH !!" >&2
      exit 1
   fi
done

## Helper function definitions:
fd_args="--argchk --colors --cmde --echo"
#fd_args+=" --Critical"
fd_args+=" --rowwrite"
#fd_args+=" --timers"
fd_args+=" --warnings"
FuncDef $fd_args >/dev/null || exit $?
eval "$(FuncDef $fd_args)"  || exit $?

## Check for arguments:
usage () { 
   Recho "\nSyntax: $this_prog --START\n\n"
   #Recho "\nSyntax: $this_prog arg1\n\n"
}
if [ "$1" != "--START" ]; then
#if [ -z "$1" ]; then
   usage >&2
   exit 1
fi

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## File/folder config:
img_dir="r17577216"
fix_dir="fix_test"
iflavor="clean"
channel="SPITZER_I1"
jpg_dir="jpegs"
cmde "mkdir -p $jpg_dir" || exit $?

## JPEG config:
ilower="-0.1"
iupper="5.0"
clipping="-B $ilower -A $iupper"
scaling="--ds9log=1000"
enlarge="-e2"
jopts="$clipping $scaling $enlarge"

#cmde "ls ${fix_dir}/SPITZ*_${iflavor}.fits"
raw_images=( `ls ${img_dir}/${channel}*_${iflavor}.fits` ) || exit $?
fix_images=( `ls ${fix_dir}/${channel}*_${iflavor}.fits` ) || exit $?

echo "raw_images: ${raw_images[*]}"
echo "fix_images: ${fix_images[*]}"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##------------------         Create Individual JPEGs        ----------------##
##--------------------------------------------------------------------------##

## Produce JPEGs:
for src_dir in $img_dir $fix_dir ; do
   save_dir="$jpg_dir/$src_dir"
   cmde "mkdir -p $save_dir" || exit $?
   img_list=( `ls ${src_dir}/${channel}*_${iflavor}.fits` )
   total=${#img_list[*]}
   for image in ${img_list[*]}; do
      ibase="${image##*/}"
      save_jpg="${save_dir}/${ibase}.jpg"
      cmde "fpeg -i $image $jopts -nj $save_jpg" || exit $?
   done
done

##--------------------------------------------------------------------------##
##------------------       Side-by-Side JPEG Montages       ----------------##
##--------------------------------------------------------------------------##

## Configuration:
mnt_opts="-tile 2x -geometry +10+0"
mnt_dir="${jpg_dir}/side_by_side"         # choose output folder
cmde "mkdir -p $mnt_dir" || exit $?       # create output folder


## List of base JPEGs:
find $jpg_dir -type f -name "*.jpg" > $foo
jpg_bases=( `sed 's|^.*/||' $foo | sort -u` )
echo "jpg_bases: ${jpg_bases[*]}"

## Create montages (one per base):
for jbase in ${jpg_bases[*]}; do
   jfiles=( `grep $jbase $foo` )
   save_jpg="${mnt_dir}/${jbase}"
   echo "jfiles: ${jfiles[*]}"
   cmde "montage ${jfiles[*]} $mnt_opts $save_jpg" || exit $?
done

## Make an HTML table for display:
htm_file="z.htm"
i2t_opts="--buffer --buffer --Rtoken"
yecho "\nMake HTML table ...\n"
cmde "ls jpegs/side_by_side/SPI*jpg | ims2table $i2t_opts > $htm_file" || exit $?

Gecho "\nView results here:\n"
yecho "--> `readlink -f $htm_file` \n\n"

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (91_make_example_JPEGs.sh):
#---------------------------------------------------------------------
#
#  2021-02-18:
#     -- Increased script_version to 0.10.
#     -- First created 91_make_example_JPEGs.sh.
#

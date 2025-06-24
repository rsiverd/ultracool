#!/bin/bash
#
# Get me RA/DE coordinates of the central corners of the four sensors using
# the ast.net solution.
#
# Rob Siverd
# Created:      2025-05-09
# Last updated: 2025-05-09
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Default options:
debug=0 ; clobber=0 ; force=0 ; timer=0 ; vlevel=0
script_version="0.01"
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
[[ -d /dev/shm ]] && [[ -w /dev/shm ]] && tmp_root="/dev/shm"
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
jnk_cleanup='for X in $jnk ; do [[ -f $X ]] && cmde "rm -vf $X" ; done'
trap "$jnk_cleanup >&2" EXIT
##trap '[[ -d $tmp_dir ]] && cmde "rm -vrf $tmp_dir"' EXIT
#trap "[[ -d $tmp_dir ]] && $dir_cleanup >&2" EXIT
#trap "[[ -d $tmp_dir ]] && $dir_cleanup >&2; $jnk_cleanup >&2" EXIT
#trap 'oops=$? ; echo ; exit $oops' HUP INT TERM

## Required programs:
declare -a need_exec
need_exec+=( awk cat FuncDef sed skypix tr )
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
if [[ "$1" != "--START" ]]; then
#if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

slv_dir="slvd_11BQ02"
save_file="corner_coords.txt"

## Wipe the output:
cmde "rm -f $bar" || exit $?

## NE image:
ne_image=$(ls $slv_dir/solved*NE.fits)
echo "2048 1" > $foo
echo -n "$ne_image " >> $bar
cmde "skypix --xy2sky $ne_image -l $foo | tee -a $bar"

## NW image:
nw_image=$(ls $slv_dir/solved*NW.fits)
echo "1 1" > $foo
echo -n "$nw_image " >> $bar
cmde "skypix --xy2sky $nw_image -l $foo | tee -a $bar"

## SE image:
se_image=$(ls $slv_dir/solved*SE.fits)
echo "2048 2048" > $foo
echo -n "$se_image " >> $bar
cmde "skypix --xy2sky $se_image -l $foo | tee -a $bar"

## SW image:
sw_image=$(ls $slv_dir/solved*SW.fits)
echo "1 2048" > $foo
echo -n "$sw_image " >> $bar
cmde "skypix --xy2sky $sw_image -l $foo | tee -a $bar"

## Save corner coordinates:
cmde "mv -f $bar $save_file" || exit $?

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## RA/DE coordinates of the four corners:
awk '{ printf "%10.6f %10.6f\n", $5,$6 }' $save_file > $baz

## Quad list:
echo NE NW SE SW | tr ' ' '\n' > $foo

## Reverse it ... get me pixel coordinates of corresponding corners:
for img in $ne_image $nw_image $se_image $sw_image ; do
   quad=$(basename $img | cut -d. -f2)
   legend="cpix_${quad}.txt"
   echo "quad: $quad"
   cmde "skypix $img --sky2xy -l $baz -o $qux"
   paste -d' ' $foo $qux | tee $legend
   #cmde "cat $qux"
done

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (02_corner_coords.sh):
#---------------------------------------------------------------------
#
#  2025-05-09:
#     -- Increased script_version to 0.01.
#     -- First created 02_corner_coords.sh.
#

#!/bin/bash
#
# Quick-and-dirty copy of source WCS into output image.
#
# Rob Siverd
# Created:      2019-10-23
# Last updated: 2019-10-23
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
need_exec+=( awk cat FuncDef hdrtool listhead sed tr )
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
   #Recho "\nSyntax: $this_prog --START\n\n"
   Recho "\nSyntax: $this_prog wcs_image target_image\n\n"
}
#if [ "$1" != "--START" ]; then
if [ -z "$2" ]; then
   usage >&2
   exit 1
fi
wcs_image="$1"
dst_image="$2"

[ -f $wcs_image ] || PauseAbort "Can't find file: $wcs_image"
[ -f $dst_image ] || PauseAbort "Can't find file: $dst_image"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

token1="WCSAXES"
token2="RADESYS"
#cmde "listhead $wcs_image"
listhead $wcs_image | BetweenMatches -A $token1 -B $token2 > $foo

#cmde "cat $foo"
#cut -c1-8 $foo > tmp_kword
#cmde "cut -d/ -f2 $foo | "
#cmde "cut -c34- $foo | awk '{ printf "
cut -c34- $foo | sed -e "s/^/--comment='/" -e "s/$/'/" > $bar
#cut -d/ -f1 $foo | cut -d= -f2 | awk '{ print "--value="$1 }' > tmp_value

#cat $foo
#echo
cut -d/ -f1 $foo | tr -d = | awk -v iname="$dst_image" '{
   printf "hdrtool %s -U %8s --value=%s\n", iname, $1, $2
}' > $baz
paste -d' ' $baz $bar > $qux
cat << EOF >> $qux
hdrtool -d $dst_image
hdrtool -FC $dst_image
EOF
cmde "cat $qux"
source $qux

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (04_patch_wcs.sh):
#---------------------------------------------------------------------
#
#  2019-10-23:
#     -- Increased script_version to 0.01.
#     -- First created 04_patch_wcs.sh.
#

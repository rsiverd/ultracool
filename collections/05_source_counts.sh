#!/bin/bash
#
# Report the number of sources in various extended catalogs.
#
# Rob Siverd
# Created:      2021-09-20
# Last updated: 2021-09-20
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Default options:
debug=0 ; clobber=0 ; force=0 ; timer=0 ; vlevel=0
script_version="0.10"
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
baz="$tmp_dir/baz_$$.fits"
qux="$tmp_dir/qux_$$.fits"
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
need_exec+=( awk cat FuncDef imhget sed tr )
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
   #Recho "\nSyntax: $this_prog catalog_list counts_file\n\n"
}
if [ "$1" != "--START" ]; then
#if [ -z "$2" ]; then
   usage >&2
   exit 1
fi
#cats_list="$1"
#save_file="$2"
#[ -f $cats_list ] || PauseAbort "Can't find file: $cats_list"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Make output folder:
save_dir="source_counts"
cmde "mkdir -p $save_dir" || exit $?

#cat_paths=( `cat $cats_list` )
#ncatalogs="${#cat_paths[@]}"
#echo "ncatalogs: $ncatalogs"

##--------------------------------------------------------------------------##
## Count contents:
ghopts="-E CATALOG"
#cmde "imhget -E CATALOG -l $cats_list NAXIS2"

##--------------------------------------------------------------------------##
## Identify collection files:
collections=()
min_lines=4
for cfile in `ls *_SPITZER_I?_nudge_pcat.txt`; do
   ncat=$(cat $cfile | wc -l)
   #echo "cfile: $cfile --> $ncat"
   if [ $ncat -gt $min_lines ]; then
      collections+=( $cfile )
   else
      echo "Ignoring small file: $cfile ($ncat entries)"
   fi
done
#collections=( `ls *_SPITZER_I?_nudge_pcat.txt` )
ncollection=${#collections[*]}
echo "Found $ncollection collection files."

## Look up catalog source counts:
count=0
ghopts="-E CATALOG NAXIS2"
for cfile in ${collections[*]}; do
   save_file="${save_dir}/nsrcs_${cfile}"
   echo "File $(( ++count )) of $ncollection ... "
   echo "cfile: $cfile"
   echo "saving to: $save_file"
   cmde "imhget $ghopts -l $cfile -o $save_file" || exit $?
done

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (05_source_counts.sh):
#---------------------------------------------------------------------
#
#  2021-09-20:
#     -- Increased script_version to 0.10.
#     -- First created 05_source_counts.sh.
#
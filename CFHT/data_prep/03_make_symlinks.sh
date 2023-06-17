#!/bin/bash
#
# Restructure data set by RUNID with symbolic links (for now).
#
# Rob Siverd
# Created:      2023-05-31
# Last updated: 2023-05-31
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
need_exec+=( awk cat FuncDef sed tr )
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

## Retrieve shared config:
cmde "source config.sh" || exit $?

## Create parent subfolder:
cmde "mkdir -p $runid_base" || exit $?

## Gather QRUNIDs and FILTERs from headers:
cmde "imhget --progress QRUNID -l $imlist -o $foo" || exit $?
cmde "imhget --progress FILTER -l $imlist -o $bar" || exit $?
runid_list=( $(awk '{ print $2 }' $foo | sort -u) )

## Create subdirectories and make symlinks:
orig_pwd="$(pwd)"
for runid in ${runid_list[*]}; do
   echo "runid: $runid"
   runid_dir="${runid_base}/${runid}"
   cmde "mkdir -p $runid_dir" || exit $?
   cmde "cd $runid_dir"       || exit $?
   run_ipaths=( $(awk -v runid="$runid" '$2 == runid { print $1 }' $foo) )
   #run_ibases=( $(awk -v runid="$runid" '$2 == runid { print $1 }' $foo \
   #            | sed 's|^.*/||') )
   #echo "run_imgs: ${run_imgs[*]}"
   #echo "run_ipaths: ${run_ipaths[*]}"
   for ipath in ${run_ipaths[*]}; do
      ibase="${ipath##*/}"
      ldest="../../download/${ibase}"
      filter=$(grep -m1 $ibase $bar | awk '{ print $2 }')
      lname="wircam_${filter}_${ibase}"
      if [[ ! -e $lname ]]; then
         cmde "ln -s $ldest ${lname}" || exit $?
      fi
   done
   cmde "cd $orig_pwd"        || exit $?
   #break
done

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
#[ -f $bar ] && rm -f $bar
#[ -f $baz ] && rm -f $baz
#[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (03_make_symlinks.sh):
#---------------------------------------------------------------------
#
#  2023-05-31:
#     -- Increased script_version to 0.01.
#     -- First created 03_make_symlinks.sh.
#

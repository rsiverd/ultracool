#!/bin/bash
#
# Check for target in all catalogs found for specified target.
#
# Rob Siverd
# Created:      2021-03-08
# Last updated: 2021-03-08
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
   #Recho "\nSyntax: $this_prog --START\n\n"
   Recho "\nSyntax: $this_prog target\n\n"
}
#if [ "$1" != "--START" ]; then
if [ -z "$1" ]; then
   usage >&2
   exit 1
fi
targname="$1"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## File/folder config:
cat_dir="../collections"
[ -d $cat_dir ] || PauseAbort "Can't find directory: $cat_dir"
save_dir="results"
checker="./91_check_listed_catalogs.sh"
[ -f $checker ] || PauseAbort "Can't find file: $checker"

## Look for catalog lists containing named target:
cat_lists=( `ls $cat_dir/${targname}_*.txt` ) || exit $?
nclists=${#cat_lists[*]}
if [ $nclists -eq 0 ]; then
   Recho "No catalog lists found for target: '$targname'\n\n" >&2
   exit 1
fi
echo "Found $nclists catalogs:"
echo ${cat_lists[*]} | tr ' ' '\n' | awk '{ printf "--> %s\n", $1 }'

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Create output folder and process lists:
Yecho "\nChecking listed catalogs ...\n"
cmde "mkdir -p $save_dir" || exit $?

## Investigate each catalog set:
for clist in ${cat_lists[*]}; do
   Mecho "\n`RowWrite 75 -`\n"
   echo "clist: $clist"
   cbase="${clist##*/}"
   echo "cbase: $cbase"
   save_file="${save_dir}/nearest_$cbase"
   echo "save_file: $save_file"
   cmde "$checker $targname $clist $save_file" || exit $?
done


##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
#[ -f $baz ] && rm -f $baz
#[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (Z01_recheck_target.sh):
#---------------------------------------------------------------------
#
#  2021-03-08:
#     -- Increased script_version to 0.01.
#     -- First created Z01_recheck_target.sh.
#

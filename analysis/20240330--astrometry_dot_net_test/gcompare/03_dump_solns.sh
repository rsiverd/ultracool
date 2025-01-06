#!/bin/bash
#
# Dump astrometric solutions from Gaia CSV file for listed objects.
#
# Rob Siverd
# Created:      2024-12-16
# Last updated: 2024-12-16
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
if [[ "$1" != "--START" ]]; then
#if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi
gid_list="gid_proc.txt"
csv_file="gaia_calib1_NE.csv"
[[ -f $gid_list ]] || PauseAbort "Can't find file: $gid_list"
[[ -f $csv_file ]] || PauseAbort "Can't find file: $csv_file"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

#head -1 $csv_file
save_file="solutions.txt"

src_list=( $(cat $gid_list) )
total=${#src_list[*]}

rm $bar 2>/dev/null
count=0
for gid in `cat $gid_list`; do
   (( count++ ))
   echo -ne "\rChecking source $count of $total ... "
   #awk -F, -v gid=$gid '$3 == gid {
   grep $gid $csv_file | awk -F, '{
      id = $3
      ra = $6
      de = $8
      prlx = $10
      pmra = $13
      pmde = $15
      printf "%d %15.10f %15.10f %8.3f %8.3f %8.4f\n", id, ra, de, pmra, pmde, prlx
   }' > $foo
   hits=$(cat $foo | wc -l)
   #echo -ne "hits: $hits"
   if [[ $hits -ne 1 ]]; then
      echo "PANIC!!"
      echo "gid == $gid"
      echo
      cat $foo
      echo
      cmde "grep $gid $csv_file"
      exit
   fi
   cat $foo >> $bar
done
echo "done."

## Sort by RA:
cmde "sort -nk2 $bar > $baz" || exit $?
cmde "mv $baz $save_file"    || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (03_dump_solns.sh):
#---------------------------------------------------------------------
#
#  2024-12-16:
#     -- Increased script_version to 0.01.
#     -- First created 03_dump_solns.sh.
#

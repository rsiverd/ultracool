#!/bin/bash
#
# Figure out which QRUNIDs have every image from every quadrant available.
#
# Rob Siverd
# Created:      2025-04-15
# Last updated: 2025-04-15
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
baz="$tmp_dir/baz_$$.fits"
qux="$tmp_dir/qux_$$.fits"
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

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

ready_list="ready_runids.txt"

got_em_all () {
   uniques=$(echo $* | tr ' ' '\n' | sort -u | wc -l)
   if [[ $1 -gt 1 ]] && [[ $uniques -eq 1 ]]; then
      return 0   # yes identical, test passed
   else
      return 1   # not identical, data missing
   fi
}

base_dir="/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc"
#cmde "ls $base_dir"

quad_dirs=( $(ls $base_dir | grep '^calib1_p_..$') )

quad_dirs=( $(ls -d $base_dir/calib1_p_??) )

runid_bases=( $(find $base_dir/calib1* -type d -name by_runid) )

runid_list=( $(ls ${runid_bases[0]}) )
echo "runid_list: ${runid_list[*]}"
echo ${quad_dirs[*]} | tr ' ' '\n'

## Iterate over QRUNID:
rm -f $foo
for runid in ${runid_list[*]}; do
   echo "runid: $runid"
   counts=()
   for dd in ${runid_bases[*]}; do
      #cmde "ls ${dd}/${runid}/*p.fits.fz"
      nimgs=$(ls ${dd}/${runid}/*p.fits.fz 2>/dev/null | wc -l)
      #echo "nimgs: $nimgs"
      counts+=( $nimgs )
   done
   echo "counts: ${counts[*]}"
   if ! ( got_em_all ${counts[*]} ); then
      echo "Not yet done ..."
      continue
   fi
   echo "$runid complete"
   echo $runid >> $foo
done

cmde "mv -f $foo $ready_list" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
#[[ -f $bar ]] && rm -f $bar
#[[ -f $baz ]] && rm -f $baz
#[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_list_finished_qrunids.sh):
#---------------------------------------------------------------------
#
#  2025-04-15:
#     -- Increased script_version to 0.01.
#     -- First created 01_list_finished_qrunids.sh.
#

#!/bin/bash
#
# Produce lists of folders where fcats for analysis can be found. Generate
# per-RUNID folder lists and a master list with everything in it.
#
# Rob Siverd
# Created:      2026-01-22
# Last updated: 2026-01-22
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
   Recho "\nSyntax: $this_prog lists_folder --START\n\n"
   #Recho "\nSyntax: $this_prog arg1\n\n"
}
if [[ "$2" != "--START" ]]; then
#if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi
save_dir="$1"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Where to look for folders:
c1root="/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc"
[[ -d $c1root ]] || PauseAbort "Can't find directory: $c1root"

## Where to save the lists we make:
cmde "mkdir -p $save_dir" || exit $?

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Start with a listing of 'by_runid' subdirs within our root:
find $c1root -type d -name by_runid | sort > $foo

## Get the RUNID-specific names within each:
rm -f $bar
for brdir in `cat $foo`; do
   echo "Checking $brdir ..."
   #cmde "ls $brdir"
   find $brdir -mindepth 1 -maxdepth 1 -name "??????" | sort >> $bar
done

## Extract the unique RUNIDs from these paths:
runid_list=( $(basename -a $(cat $bar) | sort -u) )
echo "Found ${#runid_list[*]} RUNIDs."

## Iterate over RUNIDs, make imdirs list for each, save to specified folder:
have_files=()
for runid in ${runid_list[*]}; do
   echo "runid: $runid"
   grep $runid $bar > $baz
   save_file="${save_dir}/imdirs_${runid}.txt"
   cmde "mv -f $baz $save_file" || exit $?
   have_files+=( $save_file )
done

## Make a master list:
echo "Making master list ..."
save_file="${save_dir}/imdirs_all.txt"
#echo "have_files: ${have_files[*]}"
cat ${have_files[*]} > $baz
cmde "mv -f $baz $save_file" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
#[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (02_list_fcat_dirs.sh):
#---------------------------------------------------------------------
#
#  2026-01-22:
#     -- Increased script_version to 0.01.
#     -- First created 02_list_fcat_dirs.sh.
#

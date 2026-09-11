#!/bin/bash
#
# Run the fine-tuner script on every RUNID.
#
# Rob Siverd
# Created:      2026-03-03
# Last updated: 2026-03-03
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

dirslist="imdirs_c1/imdirs_all.txt"
[[ -f $dirslist ]] || PauseAbort "Can't find file: $dirslist"
ftscript="./23_fine_tune_RUNID_WCS.py"
[[ -f $ftscript ]] || PauseAbort "Can't find file: $ftscript"
jpar_dir="joint_pars"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Get list of unique RUNIDs (randomized):
runid_list=( $(basename -a $(cat $dirslist) | sort -u | sort -R) )
echo "runid_list: ${runid_list[*]}"

## Iterate over RUNIDs and process each (J-only):
for runid in ${runid_list[*]}; do
   # skip if no results files:
   rfiles=( $(ls results/${runid}/wircam_J_*.pickle 2>/dev/null) )
   nresults=${#rfiles[*]}
   if [[ $nresults -lt 3 ]]; then
      echo "Not much to process ... try next."
      continue
   fi

   # skip if output exists
   outfile="${jpar_dir}/jpars_${runid}_J.txt"
   if [[ -f $outfile ]]; then
      echo "Output $outfile already exists ... skipping!"
      continue
   fi

   # otherwise, process
   cmde "$ftscript -R $runid --Jonly" || exit $?
done


##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
#[[ -f $foo ]] && rm -f $foo
#[[ -f $bar ]] && rm -f $bar
#[[ -f $baz ]] && rm -f $baz
#[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (24_every_RUNID_finetune.sh):
#---------------------------------------------------------------------
#
#  2026-03-03:
#     -- Increased script_version to 0.01.
#     -- First created 24_every_RUNID_finetune.sh.
#

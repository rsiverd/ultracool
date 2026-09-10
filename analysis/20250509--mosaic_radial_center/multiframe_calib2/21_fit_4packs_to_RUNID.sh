#!/bin/bash
#
# This driver script executes coordinate fitting for all available images
# in a user-specified RUNID. Briefly, this script:
# 1) accepts a user-provided RUNID on the command line
# 2) extracts corresponding image bases from the master fcat list
# 3) executes a lower-level Python analysis code on identified images 
#
# Results from this script should be suitable for joint analysis to establish
# * whether or not CRPIXn and CD are really constant in a RUNID
# * whether the distortion model as estimated is RUNID-agnostic
#
# Rob Siverd
# Created:      2026-01-27
# Last updated: 2026-01-27
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
   #Recho "\nSyntax: $this_prog --START\n\n"
   Recho "\nSyntax: $this_prog RUNID\n\n"
}
#if [[ "$1" != "--START" ]]; then
if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi
runid="$1"

## Required files:
fcat_table="fcat_paths.csv"
cfh_solver="./12_factor_fit_4pack.py"
[[ -f $fcat_table ]] || PauseAbort "Can't find file: $fcat_table"
[[ -f $cfh_solver ]] || PauseAbort "Can't find file: $cfh_solver"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

# All entries for this RUNID:
#cmde "grep $runid $fcat_table"

# Just the basename column:
#cmde "grep $runid $fcat_table | cut -d, -f1"
#cmde "grep $runid $fcat_table | cut -d, -f1 | sed 's/_fixed.*$//'"

have_bases=( $(grep $runid $fcat_table | cut -d, -f1 | sed 's/_fixed.*$//') )
#echo "have_bases: ${have_bases[*]}"
nbases=${#have_bases[*]}
echo "Found $nbases image bases for RUNID $runid."

## Optionally randomize the image order for quasi-parallel operation:
have_bases=( $(shuffle ${have_bases[*]}) )

## Folder for data products:
save_dir="results/$runid"
cmde "mkdir -p $save_dir" || exit $?

## Iterate over each of the bases:
count=0
ntodo=0
for fbase in ${have_bases[*]}; do
   (( count++ ))
   pickle="${save_dir}/${fbase}.pickle"
   if [[ -f $pickle ]]; then
      echo "Skipping image (output $pickle exists) ..."
      continue
   fi
   cmde "$cfh_solver -L $fcat_table -I $fbase -O $pickle" || exit $?
   if [[ $ntodo -gt 0 ]] && [[ $count -ge $ntodo ]]; then
      echo "Stopping after $count iterations."
      break
   fi
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
# CHANGELOG (21_fit_4packs_to_RUNID.sh):
#---------------------------------------------------------------------
#
#  2026-01-27:
#     -- Increased script_version to 0.01.
#     -- First created 21_fit_4packs_to_RUNID.sh.
#

#!/bin/bash
#
# List the fcat files from the folders I have identified. Organize by image.
#
# Rob Siverd
# Created:      2026-01-13
# Last updated: 2026-01-13
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
   Recho "\nSyntax: $this_prog folder_list\n\n"
}
#if [[ "$1" != "--START" ]]; then
if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi
folder_list="$1"
[[ -f $folder_list ]] || PauseAbort "Can't find file: $folder_list"

fpath_table="fcat_paths.csv"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## List the 'fixed' fcat catalogs with tuned-up WCS:
#find $(cat $folder_list) -type f | grep fcat | sort
find $(cat $folder_list) -type f | grep 'fixed.fits.fz.fcat$' | sort > $foo
#cat $foo

## Extract unique basenames:
uniq_fbase=( $(basename -a $(cat $foo) | sort -u) )
nfbase=${#uniq_fbase[*]}
echo "Found $nfbase unique fcat bases."

## Make sure each base has all four catalogs:
echo "fbase,ne_fpath,nw_fpath,se_fpath,sw_fpath" > $bar
for fbase in ${uniq_fbase[*]}; do
   echo "fbase: $fbase"
   grep $fbase $foo
   fpaths=( $(grep $fbase $foo) )
   hits=${#fpaths[*]}
   echo "$fbase :: found $hits fixed fcat files"
   if [[ $hits -ne 4 ]]; then
      echo "UNEXPECTED!!"
      exit 1
   fi
   # add an entry:
   echo "$fbase ${fpaths[*]}" | tr ' ' ',' >> $bar
   echo
done

## Save the result:
cmde "mv -f $bar $fpath_table" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
#[[ -f $baz ]] && rm -f $baz
#[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (02_list_fcat_files.sh):
#---------------------------------------------------------------------
#
#  2026-01-13:
#     -- Increased script_version to 0.01.
#     -- First created 02_list_fcat_files.sh.
#

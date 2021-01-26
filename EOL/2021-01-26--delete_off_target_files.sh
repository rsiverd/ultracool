#!/bin/bash
#
# This script identifies Spitzer images that do not contain a target of
# interest so those files can be removed. 
#
# Previous work retrieved data from the SHA using a too-large search radius 
# of 0.1 degree (IRAC FOV is only ~0.087 degrees in diameter). This retrieved
# many images that do not contain the target of interest (often simultaneous
# data of the adjacent star field from a second channel). These files will be
# deleted to streamline future work.
#
# Rob Siverd
# Created:      2021-01-26
# Last updated: 2021-01-26
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

## Storage location:
sha_img_root="../ucd_targets"

## File lists:
fbases=( `ls zfov_05 | grep manifest | sort -u` )
imdir_10="zfov_10"
targets=( "2m0415" "J0212" "J1832" "wise1828" )
wide_dir="zfov_10"
slim_dir="zfov_05"

## Iterate over lists:
for targ in ${targets[*]}; do
   mecho "\n`RowWrite 75 -`\n"


   echo "targ: $targ"
   mfile="manifest_${targ}.txt"
   echo "mfile: $mfile"
   imgdir="${sha_img_root}/${targ}"
   echo "imgdir: $imgdir"
   [ -d $imgdir ] || PauseAbort "Can't find directory: $imgdir"

   wide_list="${wide_dir}/$mfile"
   slim_list="${slim_dir}/$mfile"
   #cmde "wc -l $wide_list $slim_list"

   # produce sorted working copies:
   cmde "sort -u $wide_list > $foo" || exit $?
   cmde "sort -u $slim_list > $bar" || exit $?

   # removal lists get files unique to wide list:
   cmde "comm -23 $foo $bar > $baz"             || exit $?
   cmde "wc -l $foo $bar $baz"                  || exit $?
   cmde "cut -d_ -f1-5 $baz | sort -u > $qux"   || exit $?

   # save lists of files for removal:
   #cmde "mv -f $baz toss_${targ}.full.txt"
   #cmde "mv -f $qux toss_${targ}.base.txt"

   # list files in image folder:
   find $imgdir -type f -name "SPIT*fits" > $foo
   cmde "wc -l $foo"
   cmde "grep -f $qux $foo > $bar"
   ntoss=$(cat $bar | wc -l)
   nbase=$(cat $qux | wc -l)
   iquot=$(( ntoss / nbase ))
   echo "nbase: $nbase / ntoss: $ntoss"
   echo "iquot: $iquot"
   #cmde "less $bar"
   for item in `cat $bar`; do
      [ -f $item ] && cmde "rm $item"
   done

   # cleanup:
   #vcmde "rm $foo $bar"
   echo
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
# CHANGELOG (delete_off_target_files.sh):
#---------------------------------------------------------------------
#
#  2021-01-26:
#     -- Increased script_version to 0.10.
#     -- First created delete_off_target_files.sh.
#

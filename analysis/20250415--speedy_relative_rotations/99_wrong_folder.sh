#!/bin/bash
#
# Look for aligned images that are somehow in the wrong folder. This is
# probably yet another consequence of those dueling scripts ...
#
# Rob Siverd
# Created:      2025-05-05
# Last updated: 2025-05-05
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

#csv_file="shifts.csv"
#[[ -f $csv_file ]] || PauseAbort "Can't find file: $csv_file"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

for targdir in aligned_p{1,2}; do

   # List stuff:
   #targdir="aligned_p1"
   yecho "List $targdir images ... "
   find $targdir -type f -name "interp*.fits" | sort > $foo
   yecho "folders ... "
   find $targdir -type d -name "??????_??"    | sort > $bar
   gecho "done.\n"
   #head $bar
   #exit
   
   yecho "Extract QRUNID ...\n"
   cmde "imhget -l $foo QRUNID --progress -o $baz" || exit $?
   #cmde "mv -f $bar derp.txt"
   #cmde "cat $foo"
   rm -f $qux  # folders to wipe
   exec 10<$baz
   while read ipath runid <&10; do
      #echo "ipath: $ipath, runid: $runid"
      if !( echo $ipath | grep $runid >/dev/null ); then
         echo "DUBIOUS: $ipath ($runid)"
         runfld=$(echo $ipath | cut -d/ -f2 | cut -d_ -f1)
         echo $runid >> $qux
         echo $runfld >> $qux
      fi
   done
   exec 10>&-

   njunk=$(cat $qux 2>/dev/null | wc -l)
   if [[ $njunk -eq 0 ]]; then
      echo "Nothing discrepant to delete!"
      continue
      #rm -f $foo $bar $baz $qux
      #exit 0
   fi

   Mecho "\n\n`RowWrite 75 =`\n\n"

   # Get a list of folders to wipe:
   wipe_these=( $(sort -u $qux) )
   echo "wipe_these: ${wipe_these[*]}"

   # Actually wipe all variants of those folders (recall from list):
   for bad in ${wipe_these[*]}; do
      for jnkdir in $(grep $bad $bar); do
         echo "jnkdir: $jnkdir"
         cmde "rm -rf $jnkdir"
      done
   done

done  # end of loop over aligned_p{1,2}

#head $csv_file

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (99_wrong_folder.sh):
#---------------------------------------------------------------------
#
#  2025-05-05:
#     -- Increased script_version to 0.01.
#     -- First created 99_wrong_folder.sh.
#

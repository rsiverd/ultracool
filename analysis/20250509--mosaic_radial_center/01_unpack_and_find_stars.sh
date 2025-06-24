#!/bin/bash
#
# Unpack listed images (4 sensors from 1 frame) into a directory as plain
# FITS images. Find stars with SExtractor.
#
# Rob Siverd
# Created:      2025-05-09
# Last updated: 2025-05-09
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
need_exec+=( awk cat FuncDef runsex sed tr )
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
   Recho "\nSyntax: $this_prog runid_and_image list --START\n\n"
   #Recho "\nSyntax: $this_prog arg1\n\n"
}
if [[ "$2" != "--START" ]]; then
#if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi
ri_list="$1"
[[ -f $ri_list ]] || PauseAbort "Can't find file: $ri_list"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Config:
c1_proc="/home/rsiverd/ucd_project/ucd_cfh_data/calib1_proc"


##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

rsopts="-T5 -F"
exec 10<$ri_list
while read qrunid imname <&10; do
   echo "qrunid: $qrunid"
   echo "imname: $imname"
   imbase="${imname%.fz}"
   imbase="${imbase%.fits}"
   echo "imbase: $imbase"
   #save_dir="${qrunid}_${imbase}"
   save_dir="data_${qrunid}"
   cmde "mkdir -p $save_dir" || exit $?
   for qq in NE NW SE SW; do
      ls ${c1_proc}/calib1_p_${qq}/by_runid/${qrunid}/${imbase}.fits.fz
      img_path="${c1_proc}/calib1_p_${qq}/by_runid/${qrunid}/${imbase}.fits.fz"
      save_file="${save_dir}/${imbase}.${qq}.fits"
      echo "save_file: $save_file"
      if [[ ! -f $save_file ]]; then
         echo "Making $save_file ..."
         cmde "funpack -S $img_path > $baz"  || exit $?
         cmde "mv -f $baz $save_file"        || exit $?
      fi

      # Run SExtractor:
      save_objs="${save_file}.cat"
      if [[ ! -f $save_objs ]]; then
         cmde "runsex $rsopts $save_file -o $baz" || exit $?
         cmde "mv -f $baz $save_objs"
      fi
   done
   echo
done
exec 10>&-

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
#[[ -f $foo ]] && rm -f $foo
#[[ -f $bar ]] && rm -f $bar
#[[ -f $baz ]] && rm -f $baz
#[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_unpack_and_find_stars.sh):
#---------------------------------------------------------------------
#
#  2025-05-09:
#     -- Increased script_version to 0.01.
#     -- First created 01_unpack_and_find_stars.sh.
#

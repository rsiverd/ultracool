#!/bin/bash
#
# Build a script to retrieve SST ephemerides for all targets with Spitzer
# data.
#
# Rob Siverd
# Created:      2021-11-16
# Last updated: 2021-11-16
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

cmds_list_fwd="z3_get_ephems_fwd.sh"
cmds_list_rev="z3_get_ephems_rev.sh"
cmds_list_rdm="z3_get_ephems_rdm.sh"

storage_path="~/ucd_project/ucd_targets/everything"
storage_path="/net/krang.accre.vanderbilt.edu/fs0/rjs_data/ucd_project/ucd_targets/everything"
#cmde "ls $storage_path"
#[ -d $storage_path ] || PauseAbort "Can't find directory: $storage_path"

short_names=( $(ls $storage_path | grep -v z_metadata) )
ntargets=${#short_names[*]}
echo "ntargets: $ntargets"

rm $foo 2>/dev/null
for sname in ${short_names[*]}; do
   echo "sname: $sname"
   eph_file="sst_eph_${sname}.csv"
   targ_dir="${storage_path}/$sname"
   fetch_cmd="./01_get_SST_ephemeris.py -W -I ${targ_dir}/"
   fetch_cmd+=" -o ${targ_dir}/${eph_file}"
   echo "$fetch_cmd" >> $foo
done

## Save commands list:
cmde "tac     $foo > $cmds_list_rev" || exit $?
cmde "sort -R $foo > $cmds_list_rdm" || exit $?
cmde "mv -f   $foo   $cmds_list_fwd" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
#[ -f $foo ] && rm -f $foo
#[ -f $bar ] && rm -f $bar
#[ -f $baz ] && rm -f $baz
#[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (03_ephemeris_commands.sh):
#---------------------------------------------------------------------
#
#  2021-11-16:
#     -- Increased script_version to 0.10.
#     -- First created 03_ephemeris_commands.sh.
#

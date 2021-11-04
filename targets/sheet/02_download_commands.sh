#!/bin/bash
#
# This script produces a set of commands that should download Spitzer data
# for all the 'everything' objects into a custom directory.
#
# Rob Siverd
# Created:      2021-11-04
# Last updated: 2021-11-04
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

cmds_list_fwd="get_everything_fwd.sh"
cmds_list_rev="get_everything_rev.sh"
cmds_list_rdm="get_everything_rdm.sh"
targets_dir="everything"
[ -d $targets_dir ] || PauseAbort "Can't find directory: $targets_dir"

#targets_path=$(readlink -f $targets_dir)

targets_path="~/ucd_project/ultracool/targets/sheet/everything"
storage_path="~/ucd_project/ucd_targets/everything"
cmde "mkdir -p $storage_path" || exit $?


## Make commands list:
rm $foo 2>/dev/null
short_names=$(ls $targets_dir | sed 's/\.txt$//')
for sname in ${short_names[*]}; do
   echo "sname: $sname"
   fetch_cmd="./fetch_sha_data.py -t ${targets_path}/${sname}.txt"
   fetch_cmd+=" -o ${storage_path}/${sname}"
   #echo "echo $fetch_cmd" >> $foo
   echo $fetch_cmd >> $foo
done

## Save commands list:
cmde "tac     $foo > $cmds_list_rev" || exit $?
cmde "sort -R $foo > $cmds_list_rdm" || exit $?
cmde "mv -f   $foo   $cmds_list_fwd" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (02_download_commands.sh):
#---------------------------------------------------------------------
#
#  2021-11-04:
#     -- Increased script_version to 0.10.
#     -- First created 02_download_commands.sh.
#

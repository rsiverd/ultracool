#!/bin/bash
#
# Fix a bunch of wrongedy-wrong file names before rerunning Gaia matching.
#
# Rob Siverd
# Created:      2024-10-28
# Last updated: 2024-10-28
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

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Folder with solutions needs to exist ...
sol_dir="solutions"
[[ -d $sol_dir ]] || PauseAbort "Can't find directory: $sol_dir"

## First, make the rename commands:
ls $sol_dir | cut -d. -f1 | sort -u > newprefix
cat newprefix | cut -d_ -f3 > oldprefix
paste oldprefix newprefix \
   | awk '{printf "rename %s %s %s.*\n", $1, $2, $1 }' > rename_cmds.txt
rm oldprefix newprefix



##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

rcmds="$(pwd)/rename_cmds.txt"
flist="fldr_list.txt"
[[ -f $flist ]] || PauseAbort "Can't find file: $flist"
[[ -f $rcmds ]] || PauseAbort "Can't find file: $rcmds"

for fff in `cat $flist`; do
   cmde "cd $fff"
   cmde "ls"
   ls *.fits.fz | grep -v '^wirc' | cut -d. -f1 | sort -u > $foo
   for ppp in `cat $foo`; do
      cmde "grep $ppp $rcmds > $bar"
      cmde "source $bar"
   done
   #cmde "grep -f $foo $rcmds > $bar"
   #cmde "source $bar"
   #break
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
# CHANGELOG (10_rename_for_v2.sh):
#---------------------------------------------------------------------
#
#  2024-10-28:
#     -- Increased script_version to 0.01.
#     -- First created 10_rename_for_v2.sh.
#

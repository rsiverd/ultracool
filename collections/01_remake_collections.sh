#!/bin/bash
#
# Create catalog collections per target and processing method for
# further analysis.
#
# Rob Siverd
# Created:      2021-02-23
# Last updated: 2021-03-18
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Default options:
debug=0 ; clobber=0 ; force=0 ; timer=0 ; vlevel=0
script_version="0.25"
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
   Recho "\nSyntax: $this_prog target_dir --START\n\n"
   #Recho "\nSyntax: $this_prog arg1\n\n"
}
if [ "$2" != "--START" ]; then
#if [ -z "$1" ]; then
   usage >&2
   exit 1
fi
targ_dir="$1"
[ -d $targ_dir ] || PauseAbort "Can't find directory: $targ_dir"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Permutation config:
cat_suffixes=( pcat ) #fcat pcat mcat )
image_flavors=( nudge ) #clean hcfix )
irac_channels=( 1 2 )

targname=`basename $targ_dir`
echo "targname: $targname"

for suff in ${cat_suffixes[*]}; do
   find $targ_dir/r* -type f -name "*${suff}*" | sort > $foo
   #cat $foo
   #exit
   for ppalg in ${image_flavors[*]}; do
      grep "${ppalg}.fits.${suff}\$" $foo > $bar
      for ichan in ${irac_channels[*]}; do
         echo -------
         chtag="SPITZER_I${ichan}"
         csave="${targname}_${chtag}_${ppalg}_${suff}.txt"
         echo "csave: '$csave'"
         grep "$chtag" $bar | head
         grep "$chtag" $bar > $csave
      done
   done
done

## Make an 'everything' file:
everything="all_files_${targname}.txt"
cat ${targname}_*.txt | sort -u > $everything
cmde "wc -l $everything"

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_remake_collections.sh):
#---------------------------------------------------------------------
#
#  2021-03-04:
#     -- Increased script_version to 0.15.
#     -- Now also identify pcat catalogs.
#
#  2021-02-23:
#     -- Increased script_version to 0.10.
#     -- First created 01_remake_collections.sh.
#

#!/bin/bash
#
# Extract offsets from the aligned images.
#
# Rob Siverd
# Created:      2025-04-15
# Last updated: 2025-04-15
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

## Output files:
imgs_list="zalign.txt"
save_file="shifts.csv"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

#srcdir="aligned_p1"

## List images:
#find aligned_p1 -type f -name "interp*fits" | sort  > $foo
#find aligned_p2 -type f -name "interp*fits" | sort >> $foo

## This routine looks at a file listing and produces a list of base image
## names that correspond to complete quads (we have NE, NW, SE, SW). The
## results can be used to pick 'safe' interp images for mining. This splits
## on "_", keeps the last element as the base name, and counts occurrences.
## Bases that appear 4 times are printed at the end.
select_quads () {
   awk '{
   nelem = split($1, pieces, "_")
   wname = pieces[nelem]
   count[wname]++
} END { 
   for (im in count) 
      if (count[im] == 4) print im;
}' $1 | sort
}

## Completed quads are identified separately for each polynomial order.
rm -f $baz
for pp in 1 2 ; do
   echo "Checking p$pp images ..."
   srcdir="aligned_p$pp"
   find $srcdir -type f -name "interp*fits" | sort > $foo  # all interps
   select_quads $foo > $bar                                # 4-peat bases
   grep -f $bar $foo | sort >> $baz                        # images that match
done

## Save the image list:
cmde "mv -f $baz $imgs_list" || exit $?

## Mine header keywords:
keys="FILTER OFFSET_X OFFSET_Y OFFSET_R REF_SRCS INT_SRCS INT_REF QSOGRADE"
args="-N -d, --progress"
cmde "imhget $args -l $imgs_list $keys -o $foo"    || exit $?
cmde "mv -f $foo $save_file"                       || exit $?


## Warn in case of obviously bad data:
nbad=$(grep -c ___ $save_file)
if [[ $nbad -gt 0 ]]; then
   echo -e "\n\nTrouble ... delete these and rerun ..."
   grep ___ $save_file | cut -d, -f1 | awk '{printf "rm %s\n", $1}'
fi

## Also warn in case of dubious data:
nsrcmin=500
awk -F, -v nsrcmin=$nsrcmin 'NR>1 { if ($7 < nsrcmin) { print }}' $save_file

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (04_gather_offsets.sh):
#---------------------------------------------------------------------
#
#  2025-04-15:
#     -- Increased script_version to 0.01.
#     -- First created 04_gather_offsets.sh.
#

#!/bin/bash
#
# Gather paths to J-band calib1/calib2 data from the same 2012-04-04 UTC
# observing night (QRUNID 12AQ05) for further inspection.
#
# NOTES: 
# * calib1 images of interest are 1538227p.fits.fz - 1538247p.fits.fz
# * calib2 images of interest are 1537826p.fits.fz - 1537849p.fits.fz
#
# Rob Siverd
# Created:      2026-10-13
# Last updated: 2026-10-13
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
need_exec+=( awk BetweenMatches cat FuncDef sed tr )
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

## Input/output bits:
outdir="imlists"
cmde "mkdir -p $outdir" || exit $?

## Key data points:
qrunid="12AQ05"
c1img1="1538227p.fits.fz"
c1img2="1538247p.fits.fz"
c2img1="1537826p.fits.fz"
c2img2="1537849p.fits.fz"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##------------------         calib1 image selection         ----------------##
##--------------------------------------------------------------------------##

## Output base:
obase="c1_${qrunid}"

## List all calib1_proc image files for QRUNID 12AQ05:
locate $qrunid | grep calib1_proc | grep 'fits.fz$' \
   | grep wircam_J | sort > $foo

## Iterate over sensors:
for qq in NE NW SE SW; do
   token="p_${qq}/by_runid"
   ofile="${outdir}/${obase}_${qq}.txt"
   grep $token $foo \
      | BetweenMatches -A $c1img1 -B $c1img2 > $ofile
done

##--------------------------------------------------------------------------##
##------------------         calib2 image selection         ----------------##
##--------------------------------------------------------------------------##

## Output base:
obase="c2_${qrunid}"

## List all calib1_proc image files for QRUNID 12AQ05:
locate $qrunid | grep calib2_proc | grep 'fits.fz$' \
   | grep wircam_J | sort > $foo

## Iterate over sensors:
for qq in NE NW SE SW; do
   token="p_${qq}/by_runid"
   ofile="${outdir}/${obase}_${qq}.txt"
   grep $token $foo \
      | BetweenMatches -A $c2img1 -B $c2img2 > $ofile
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
# CHANGELOG (04_list_samenight_data.sh):
#---------------------------------------------------------------------
#
#  2026-10-13:
#     -- Increased script_version to 0.01.
#     -- First created 04_list_samenight_data.sh.
#

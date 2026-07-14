#!/bin/bash
#
# Extract an average PSF from the test image using 3 methods.
#
# Rob Siverd
# Created:      2026-07-07
# Last updated: 2026-07-07
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
need_exec+=( awk cat estimate-image-PSF FuncDef runsex sed tr )
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

sample_image="./wircam_J_1325839p.fits.fz"
[[ -f $sample_image ]] || PauseAbort "Can't find file: $sample_image"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

save_dir="PSF"
cmde "mkdir -p $save_dir" || exit $?

## Nearest-neighbor:
Mecho "\n`RowWrite 75 -`\n"
echo "Nearest-neighbor method ..."
save_psf="psf_nearestn.fits"
save_cat="Z${save_psf%.fits}.cat"
cmde "estimate-image-PSF $sample_image -N  -o $baz"   || exit $?
cmde "kimstat -S $baz"                                || exit $?
cmde "runsex $baz -o $foo"                            || exit $?
cmde "mv -f $baz ${save_dir}/${save_psf}"             || exit $?
cmde "mv -f $foo ${save_dir}/${save_cat}"             || exit $?

## Lanczos(2):
Mecho "\n`RowWrite 75 -`\n"
echo "Lanczos(2) method ..."
save_psf="psf_lanczos2.fits"
save_cat="Z${save_psf%.fits}.cat"
cmde "estimate-image-PSF $sample_image -L2 -o $baz"   || exit $?
cmde "kimstat -S $baz"                                || exit $?
cmde "runsex $baz -o $foo"                            || exit $?
cmde "mv -f $baz ${save_dir}/${save_psf}"             || exit $?
cmde "mv -f $foo ${save_dir}/${save_cat}"             || exit $?

## Lanczos(3):
Mecho "\n`RowWrite 75 -`\n"
echo "Lanczos(3) method ..."
save_psf="psf_lanczos3.fits"
save_cat="Z${save_psf%.fits}.cat"
cmde "estimate-image-PSF $sample_image -L3 -o $baz"   || exit $?
cmde "kimstat -S $baz"                                || exit $?
cmde "runsex $baz -o $foo"                            || exit $?
cmde "mv -f $baz ${save_dir}/${save_psf}"             || exit $?
cmde "mv -f $foo ${save_dir}/${save_cat}"             || exit $?

## Lanczos(4):
Mecho "\n`RowWrite 75 -`\n"
echo "Lanczos(4) method ..."
save_psf="psf_lanczos4.fits"
save_cat="Z${save_psf%.fits}.cat"
cmde "estimate-image-PSF $sample_image -L4 -o $baz"   || exit $?
cmde "kimstat -S $baz"                                || exit $?
cmde "runsex $baz -o $foo"                            || exit $?
cmde "mv -f $baz ${save_dir}/${save_psf}"             || exit $?
cmde "mv -f $foo ${save_dir}/${save_cat}"             || exit $?


##--------------------------------------------------------------------------##
## Find centroids with SourceExtractor:


##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (03_make_wircam_PSFs.sh):
#---------------------------------------------------------------------
#
#  2026-07-07:
#     -- Increased script_version to 0.10.
#     -- First created 03_make_wircam_PSFs.sh.
#

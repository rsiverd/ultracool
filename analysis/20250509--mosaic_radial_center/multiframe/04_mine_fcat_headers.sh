#!/bin/bash
#
# Extract important metadata from the headers of listed catalogs to
# assist in downstream analysis of aberration, refraction, and seasonal
# effects seen in the coordinate solutions.
#
# Rob Siverd
# Created:      2026-05-12
# Last updated: 2026-05-12
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
   #Recho "\nSyntax: $this_prog --START\n\n"
   Recho "\nSyntax: $this_prog fcat_listing.csv output_file.csv\n\n"
}
#if [[ "$1" != "--START" ]]; then
if [[ -z "$2" ]]; then
   usage >&2
   exit 1
fi
fcat_csv="$1"
save_csv="$2"
[[ -f $fcat_csv ]] || PauseAbort "Can't find file: $fcat_csv"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

echo "fcat_csv: $fcat_csv"
echo "save_csv: $save_csv"

## Start by selecting the columns that I want from the fcat_paths table and
## converting to SSV for easier reading. Current columns are:
## --> fbase,ne_fpath,nw_fpath,se_fpath,sw_fpath,qsograde,qrunid
## I want to keep fbase, ne_fpath, and qrunid
## NOTES:
## * temporary file foo still has a header
## * temporary file bar has no header and only full fcat paths
## UPDATE:
## * just grab the 2nd column directly for the full paths
#sed '1 d' $fcat_csv | cut -d, -f1,2,7 | tr ',' ' ' > $foo
sed '1 d' $fcat_csv | cut -d, -f2 > $foo
#cut -d, -f1,2,7 $fcat_csv | tr ',' ' ' > $foo
#sed '1 d' $foo | awk '{print $2}' > $bar

## Extract desired keywords from all imgheader extensions, save to file:
imgkeys="AIRMASS TELALT TELAZ MCTR_RA MCTR_DEC"
imgargs="-E IMGHEADER -d, -N"
cmde "imhget $imgargs -l $foo $imgkeys -o $bar" || exit $?
head $bar

## Extract desired keywords from all catalog extensions, save to file:
catkeys="JDTDB OBS_X OBS_Y OBS_Z OBS_VX OBS_VY OBS_VZ"
catargs="-E CATALOG -d, -N"
cmde "imhget $catargs -l $foo -d, -N $catkeys -o $baz" || exit $?
head $baz

##--------------------------------------------------------------------------##
## Build output file by side-by-side appending:
## 1) fbase,qrunid from fcat_paths.csv (now including header)
## 2) imgheader keywords ($baz) with FILENAME snipped
## 3) catalog   keywords ($qux) with FILENAME snipped
cut -d, -f1,7 $fcat_csv > $foo
#head $foo
#cmde "wc -l $foo $bar $baz"
#head $baz | cut -d, -f2-

paste -d, $foo <(cut -d, -f2- $bar) <(cut -d, -f2- $baz) > $qux
#cmde "head $qux"
#cmde "enumerate_columns -d, $qux"
cmde "mv -f $qux $save_csv" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (04_mine_fcat_headers.sh):
#---------------------------------------------------------------------
#
#  2026-05-12:
#     -- Increased script_version to 0.10.
#     -- First created 04_mine_fcat_headers.sh.
#

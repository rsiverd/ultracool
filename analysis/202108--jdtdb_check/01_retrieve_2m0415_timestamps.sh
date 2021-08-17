#!/bin/bash
#
# Get time stamps from 2m0415 images that were used in the analysis. These
# can be cross-referenced to the data set that emerges for fitting.
#
# Rob Siverd
# Created:      2021-08-16
# Last updated: 2021-08-16
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
need_exec+=( awk cat FuncDef imhget sed tr )
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

data_dir="/home/rsiverd/ucd_project/ucd_targets/2m0415"
save_all="mjds_2m0415_all.txt"
save_ch1="mjds_2m0415_ch1.txt"
save_ch2="mjds_2m0415_ch2.txt"

## All image and catalog files of interest:
find $data_dir -type f -name "SPITZER*clean.fits*" | sort > $foo
cmde "wc -l $foo"

## Select pcat versions:
grep 'pcat$' $foo > $bar

## Get corresponding images of pcat catalogs:
sed -i 's/\.pcat$//' $bar
#ls `cat $baz`
keys="MJD_OBS"
cmde "imhget -l $bar -o $baz $keys" || exit $?

## Strip out folder paths from filename:
sed -i 's|^.*/SPI|SPI|' $baz
sort -nk2 $baz > $qux
mv -f $qux $baz

## Convert to JD:
awk '{ printf "%s %15.6f\n", $1, 2400000.5+$2 }' $baz > $qux

## Save for inspection:
cmde "mv -f $qux $save_all" || exit $?

## Snag/save ch1 files:
yecho "Snagging ch1 ... "
grep SPITZER_I1 $save_all > $baz
vcmde "mv -f $baz $save_ch1" || exit $?
gecho "done.\n"

## Snag/save ch2 files:
yecho "Snagging ch2 ... "
grep SPITZER_I2 $save_all > $baz
vcmde "mv -f $baz $save_ch2" || exit $?
gecho "done.\n"


##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_retrieve_2m0415_timestamps.sh):
#---------------------------------------------------------------------
#
#  2021-08-16:
#     -- Increased script_version to 0.01.
#     -- First created 01_retrieve_2m0415_timestamps.sh.
#

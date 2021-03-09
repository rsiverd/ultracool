#!/bin/bash
#
# Check catalog contents.
#
# Rob Siverd
# Created:      2021-03-04
# Last updated: 2021-03-04
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
need_exec+=( awk cat FuncDef sed tablist tr )
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
   Recho "\nSyntax: $this_prog targname catalog\n\n"
}
#if [ "$1" != "--START" ]; then
if [ -z "$2" ]; then
   usage >&2
   exit 1
fi
tgt_name="$1"
cat_file="$2"
[ -f $cat_file ] || PauseAbort "Can't find file: $cat_file"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Column selection:
get_cols="[col wdra;wdde]"
get_cols="[col dra;dde]"

## Target coordinates (WISE1828):
case $tgt_name in
   2m0415)
      targ_ra="63.83208333"
      targ_de="-9.58497222"
      ;;
   wise1828)
      targ_ra="277.12595833"
      targ_de="26.84355556"
      ;;
   *)
      PauseAbort "Unrecognized target: '$tgt_name'"
      ;;
esac

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Get coordinates from catalog:
cmde "tablist '${cat_file}${get_cols}' > $foo" || exit $?
sed '1,2 d' $foo | awk '{print $2,$3}' > $bar
mv -f $bar $foo
#cmde "head $bar"

## Look for target(s):
yecho "Angular separation check ...\n" >&2
awk -v targ_ra=$targ_ra -v targ_de=$targ_de -v pi=3.14159265358979323844 '
function acos(x) { return atan2(sqrt(1.0 - x*x), x) }
function  d2r(x) { return (pi * x / 180.0) }
function  r2d(x) { return (180.0 * x / pi) }
function degsep(ra1, de1, ra2, de2) {
   # Convert to Cartesian:
   X1 = cos(pi * de1 / 180.0) * cos(pi * ra1 / 180.0)
   X2 = cos(pi * de2 / 180.0) * cos(pi * ra2 / 180.0)
   Y1 = cos(pi * de1 / 180.0) * sin(pi * ra1 / 180.0)
   Y2 = cos(pi * de2 / 180.0) * sin(pi * ra2 / 180.0)
   Z1 = sin(pi * de1 / 180.0)
   Z2 = sin(pi * de2 / 180.0)
   dotp = X1*X2 + Y1*Y2 + Z1*Z2
   asep = (180.0 / pi) * acos(dotp)
   return asep
}
{
   dra = $1
   dde = $2
   sep_asec = 3600.0 * degsep(targ_ra, targ_de, dra, dde)
   printf "%s --> %8.4f\n", $0, sep_asec
   #printf "%s %9.5f %9.5f %9.5f %9.5f\n", $1,
}' $foo > $bar

#cmde "cat $bar"
#cmde "sort -rnk4 $bar | tail -n3"
vcmde "sort -rnk4 $bar | tail -n1"

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_cat_check.sh):
#---------------------------------------------------------------------
#
#  2021-03-08:
#     -- Increased script_version to 0.10.
#     -- Changed script name to 01_cat_check.sh.
#
#  2021-03-04:
#     -- Increased script_version to 0.01.
#     -- First created ccheck.sh.
#

#!/bin/bash
#
# Snip out desired quadrant from calib1 data.
#
# Rob Siverd
# Created:      2023-05-25
# Last updated: 2023-05-25
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

## Retrieve shared config:
cmde "source config.sh" || exit $?

## Which quadrant to work with:
#use_quad="SW"

## Folders:
data_dir="/home/rsiverd/ucd_project/ucd_cfh_data/calib1/download"
[ -d $data_dir ] || PauseAbort "Can't find directory: $data_dir"
save_dir="./calib1_p_${use_quad}/download"
cmde "mkdir -p $save_dir" || exit $?

## Sensor orientation and ID:
NW_quad="77"
NW_qext="HAWAII-2RG-#77"
SW_quad="52"
SW_qext="HAWAII-2RG-#52"
NW_quad="54"
NW_qext="HAWAII-2RG-#54"
NE_quad="60"
NE_qext="HAWAII-2RG-#60"

## Which quadrant to use:
#use_qext="$ne_qext"

echo "use_quad: $use_quad"
quad_var="${use_quad}_qext"
echo "quad_var: $quad_var"
use_qext="${!quad_var}"
echo "use_qext: $use_qext"

##--------------------------------------------------------------------------##
## List files:
cmde "ls $data_dir/*p.fits.fz > $foo" || exit $?
total=$(cat $foo | wc -l)

## fitsarith options:
fopts="--bitpix -q -H"

## Snip files:
ntodo=42
count=0
nproc=0
exec 10<$foo
while read ipath <&10; do
   (( count++ ))
   #echo "ipath: $ipath"
   ibase="${ipath##*/}"
   #echo "ibase: $ibase"
   isave="${save_dir}/${ibase}"
   #echo "isave: $isave"
   yecho "Making $isave ... "

   # skip if already done:
   if [[ -f $isave ]]; then
      gecho "exists!\n"
      continue
   fi
   #recho "not found!\n"
   echo
   (( nproc++ ))

   # snip quadrant:
   cmde "fitsarith $fopts -i '${ipath}[${use_qext}]' -o '!$baz'"   || exit $?

   # re-compress:
   #cmde "fpack -F -q0 -1 $baz"                              || exit $?
   cmde "fpack -F -Y $baz"    || exit $?

   # save:
   cmde "mv -f $baz $isave"   || exit $?
   echo

   # stop early if requested:
   [[ $ntodo -gt 0 ]] && [[ $nproc -ge $ntodo ]] && break
done
exec 10>&-

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_snip_calib1.sh):
#---------------------------------------------------------------------
#
#  2023-05-25:
#     -- Increased script_version to 0.01.
#     -- First created 01_snip_calib1.sh.
#

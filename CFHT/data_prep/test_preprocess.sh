#!/bin/bash
#
# Snip out desired quadrant from CFHT/WIRCam data.
#
# Rob Siverd
# Created:      2023-05-25
# Last updated: 2024-02-29
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Default options:
debug=0 ; clobber=0 ; force=0 ; timer=0 ; vlevel=0
script_version="0.02"
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
#data_dir="/home/rsiverd/ucd_project/ucd_cfh_data/calib1/download"
#data_dir="${HOME}/ucd_project/ucd_cfh_data/calib1/download"
[ -d $data_dir ] || PauseAbort "Can't find directory: $data_dir"
save_dir="${targ_dir%/}/download"
save_base="${runid_base}"
#save_dir="./calib1_p_${use_quad}/download"
cmde "mkdir -p $save_base" || exit $?

## Sensor orientation and ID:
NW_quad="77"
NW_qext="HAWAII-2RG-#77"
SW_quad="52"
SW_qext="HAWAII-2RG-#52"
SE_quad="54"
SE_qext="HAWAII-2RG-#54"
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
cmde "mv -f $foo $imlist"             || exit $?

## Extract header data for each image (if needed):
if [[ ! -f $key_data ]]; then
   ghargs="--progress -E $use_qext -N -d,"
   get_keys="MJD-OBS DATE-OBS UTIME EXPTIME FILTER RUNID QRUNID CRUNID"
   get_keys+=" MCTR_RA MCTR_DEC"
   #get_keys+=" MCTR_RA MCTR_DEC IMAGESWV CONSWV PCISWV"
   cmde "imhget $ghargs -l $imlist $get_keys -o $foo"             || exit $?
   #cmde "mv -f $foo $key_data" || exit $?
   
   ## Snip out the extension name, delete spaces, save:
   cmde "sed 's/\[$use_qext\]//' $foo | tr -d '[:blank:]' > $bar" || exit $?
   #head $bar | tr -d' '
   cmde "mv -f $bar $key_data"                                    || exit $?
fi

## Ensure bad pixel masks exist:
#cmde "./make-wircam-pixmasks.py -i $key_data -o $pixmask_dir"  || exit $?
#cmde "./make-wircam-pixmasks.py -i $key_data -o $targ_dir"     || exit $?
cmde "./make-wircam-pixmasks.py --$use_quad -i $key_data -o $runid_base" || exit $?
#cmde "./make-wircam-pixmasks.py -i $vers_csv -o $runid_base"   || exit $?

## fitsarith options:
#fopts="--bitpix -q -H"

## Snip files:
ntodo=0
count=0
nproc=0
exec 10<$imlist
while read ipath <&10; do
   (( count++ ))
   #echo "ipath: $ipath"
   ibase="${ipath##*/}"
   #echo "ibase: $ibase"
   # raw image path:
   #rpath="$(echo $

   # Get QRUNID from image header:
   #cmde "imhget QRUNID $ipath"
   qrunid=$(imhget QRUNID $ipath | awk '{print $1}') || exit $?
   echo "Got qrunid: '$qrunid'"

   # Get filter from image header:
   filter=$(imhget FILTER $ipath | awk '{print $1}') || exit $?
   echo "Got filter: '$filter'"

   # Output folder based on QRUNID:
   save_dir="${save_base}/${qrunid}"
   echo "save_dir: $save_dir"
   #cmde "ls $save_dir"
   pix_mask="${save_dir}/badpix.fits"
   if [[ ! -f $pix_mask ]]; then
      Recho "\nPixel mask missing: ${pix_mask}\n"
      continue
   fi
   isave="${save_dir}/wircam_${filter}_${ibase}"
   echo "isave: $isave"
   #exit

   # Corresponding raw:
   rpath=$(echo $ipath | sed 's/p\.fits.fz$/o.fits.fz/')
   yecho "Making $isave ... "

   # abort if either of o/p images missing:
   [[ ! -f $ipath ]] && PauseAbort "Image $ipath disappeared ..."
   if [[ ! -f $rpath ]]; then
      recho "Raw not found: '$rpath' ...\n"
      continue
   fi

   # skip if already done:
   if [[ -f $isave ]]; then
      gecho "exists!\n"
      continue
   fi
   #recho "not found!\n"
   echo
   (( nproc++ ))

   # snip quadrant:
   #cmde "fitsarith $fopts -i '${ipath}[${use_qext}]' -o '!$baz'"   || exit $?
   echo "ipath: $ipath"
   args="--$use_quad --raw $rpath --proc $ipath --mask $pix_mask"
   cmde "./preprocess-wircam.py $args -o $baz" || exit $?

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

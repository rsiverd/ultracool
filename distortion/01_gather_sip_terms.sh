#!/bin/bash
#
# Gather SIP distortion terms from FITS header of specified images.
#
# Rob Siverd
# Created:      2019-11-06
# Last updated: 2019-11-06
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
   #Recho "\nSyntax: $this_prog --START\n\n"
   Recho "\nSyntax: $this_prog image_list.txt output_file\n\n"
}
#if [ "$1" != "--START" ]; then
if [ -z "$2" ]; then
   usage >&2
   exit 1
fi
imlist="$1"
save_txt="$2"
[ -f $imlist ] || PauseAbort "Can't find file: $imlist"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## General keywords:
cat << EOF > $foo
DATE_OBS
EOF

## General WCS keywords:
cat << EOF >> $foo
CRPIX1
CRPIX2
CRVAL1
CRVAL2
CRDER1
CRDER2
EOF

## Distortion keywords:
cat << EOF >> $foo
A_ORDER
A_0_2  
A_0_3  
A_0_4  
A_0_5  
A_1_1  
A_1_2  
A_1_3  
A_1_4  
A_2_0  
A_2_1  
A_2_2  
A_2_3  
A_3_0  
A_3_1  
A_3_2  
A_4_0  
A_4_1  
A_5_0  
A_DMAX 
B_ORDER
B_0_2  
B_0_3  
B_0_4  
B_0_5  
B_1_1  
B_1_2  
B_1_3  
B_1_4  
B_2_0  
B_2_1  
B_2_2  
B_2_3  
B_3_0  
B_3_1  
B_3_2  
B_4_0  
B_4_1  
B_5_0  
B_DMAX 
EOF

#cmde "cat $foo"

##--------------------------------------------------------------------------##
## Retrieve keywords:
cmde "imhget --progress -N -k $foo -l $imlist -o $bar" || exit $?
cmde "mv -f $bar $save_txt" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
#[ -f $baz ] && rm -f $baz
#[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_gather_sip_terms.sh):
#---------------------------------------------------------------------
#
#  2019-11-06:
#     -- Increased script_version to 0.10.
#     -- First created 01_gather_sip_terms.sh.
#

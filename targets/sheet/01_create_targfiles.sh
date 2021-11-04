#!/bin/bash
#
# Create rudimentary target files from UltracoolSheet contents (CSV).
#
# Rob Siverd
# Created:      2021-11-04
# Last updated: 2021-11-04
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
need_exec+=( awk cat coords FuncDef sed tr )
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

## Data file:
sheet_csv="main_tab.csv"
[ -f $sheet_csv ] || PauseAbort "Can't find file: $sheet_csv"

## Output folder:
save_dir="everything"
cmde "mkdir -p $save_dir" || exit $?

## Columns of interest:
name_col=1
dra_col=11
dde_col=12

## Select desired columns and remove header:
sed '1 d' $sheet_csv | cut -d, -f${name_col},${dra_col},${dde_col} > $foo
cmde "wc -l $foo"
echo

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Make shortish names, check for uniqueness:
cut -d, -f1 $foo | tr ' ' '_' | tr '+' 'n' | tr '-' 's' | tr -d '.' \
   | tr -d \( | tr -d \) | tr -d \' | tr -d \" \
   | tr -d \[ | tr -d \] | tr -d \* > $bar
cmde "wc -l $bar"
total=$(cat $bar | wc -l)
echo

n_unique=$(sort -u $bar | wc -l)
if [ $n_unique -ne $total ]; then
   PauseAbort "Names fail uniqueness test!"
fi

## Append to data for further processing:
paste -d, $foo $bar > $baz
#head $baz
#exit

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Converter function:
skycoord_fmt () {
   dra=$1
   dde=$2
   vcmde "coords --sex_only -D $dra $dde > $qux"
   shra=$(grep '^R.A.(hrs)' $qux | awk '{print $2}')
   sdde=$(grep '^Dec.(deg)' $qux | awk '{print $2}')
   #cat $qux
   #echo "shra: $shra"
   #echo "sdde: $sdde"
   # reformat RA:
   ra_hh="${shra:1:2}"
   ra_mm="${shra:4:2}"
   ra_ss="${shra:7:6}"
   skyc_ra="${ra_hh}h${ra_mm}m${ra_ss}s"
   # reformat DE:
   de_dd="${sdde:0:3}"
   de_mm="${sdde:4:2}"
   de_ss="${sdde:7:6}"
   skyc_de="${de_dd}d${de_mm}m${de_ss}s"
   echo $skyc_ra $skyc_de
}

## Convert decimal degrees to sexagesimal:
exec 10<$baz
#head -1 $foo | cut -d, -f2-3 | tr ',' ' '
while read line <&10; do
   radec=( $(echo $line | cut -d, -f2-3 | tr ',' ' ') )
   oname=$(echo $line | cut -d, -f4)
   #echo "line:  $line"
   #echo "radec: ${radec[*]}"
   #cmde "coords -D ${radec[*]}"
   skycoords=$(skycoord_fmt ${radec[*]})
   #echo "skycoords: $skycoords"
   #echo "$skycoords      #  $oname"
   save_file="${save_dir}/${oname}.txt"
   #echo "save_file: $save_file"
   echo "Creating $save_file ..."
   echo "$skycoords      #  $oname" > $save_file
   #break
done
exec 10>&-

echo "Target files created, script complete!"

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (01_create_targfiles.sh):
#---------------------------------------------------------------------
#
#  2021-11-04:
#     -- Increased script_version to 0.10.
#     -- First created 01_create_targfiles.sh.
#

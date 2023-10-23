#!/bin/bash
#
# Create region files from the 'gmatch' data for diagnostic purposes.
#
# Rob Siverd
# Created:      2023-09-28
# Last updated: 2023-09-28
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
   Recho "\nSyntax: $this_prog gmatch_file\n\n"
}
#if [ "$1" != "--START" ]; then
if [ -z "$1" ]; then
   usage >&2
   exit 1
fi
gmatch_file="$1"
[ -f $gmatch_file ] || PauseAbort "Can't find file: $gmatch_file"

## Save files:
pix_reg_file="${gmatch_file}.pix.reg"
rel_reg_file="${gmatch_file}.rel.reg"
sky_reg_file="${gmatch_file}.sky.reg"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Strip header, save data into temporary file:
sed '1 d' $gmatch_file > $foo

## Count entries in file:
n_matches=$(cat $foo | wc -l)
echo "Found $n_matches matches."

## Color map for matches:
use_colors=( "red" "blue" "green" "magenta" "yellow" "cyan" )
n_colors=${#use_colors[*]}

## Make a cycling color map:
rm $bar 2>/dev/null
for x in `seq $n_matches`; do
   cidx=$(( x % $n_colors ))
   this_color="${use_colors[$cidx]}"
   #echo "x: $x ($this_color)"
   echo "$this_color" >> $bar
done

## Append colors to existing data:
paste -d, $foo $bar > $baz
mv -f $baz $foo ; rm $bar


##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Make pixel region file:
echo "Making pixel region file ... "
awk -F, -v radius=3.0 '{
   xpix = $1 ; ypix = $2
   xrel = $3 ; yrel = $4
   g_ra = $5 ; g_de = $6
   color = $7
   printf "image; circle(%.3f, %.3f, %.2f) # color=%s\n", xpix, ypix, radius, color
}' $foo > $qux
cmde "mv -f $qux $pix_reg_file"

## Make relative pixel region file:
awk -F, -v radius=2.0 '{
   xpix = $1 ; ypix = $2
   xrel = $3 ; yrel = $4
   g_ra = $5 ; g_de = $6
   color = $7
   printf "image; circle(%.3f, %.3f, %.2f) # color=%s\n", xrel, yrel, radius, color
}' $foo > $qux
cmde "mv -f $qux $rel_reg_file"

## Make ra/dec region file:
awk -F, -v radius=0.0002 '{
   xpix = $1 ; ypix = $2
   xrel = $3 ; yrel = $4
   g_ra = $5 ; g_de = $6
   color = $7
   printf "fk5; circle(%.5fd, %.5fd, %.5fd) # color=%s\n", g_ra, g_de, radius, color
}' $foo > $qux
cmde "mv -f $qux $sky_reg_file"

##--------------------------------------------------------------------------##
## G

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (14_regify_gmatch.sh):
#---------------------------------------------------------------------
#
#  2023-09-28:
#     -- Increased script_version to 0.01.
#     -- First created 14_regify_gmatch.sh.
#

#!/bin/bash
#
# Estimate a central RA/DE coordinate for this field from MCTR keywords.
#
# Rob Siverd
# Created:      2023-06-02
# Last updated: 2023-06-02
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

## Check for file:
cmde "ls $key_data"     || exit $?

## Determine header columns:
delim=" "
hdr_cols=( $(head -1 $key_data) )
echo "${hdr_cols[*]}"

## Column lookup:
get_colnum () {
   head -1 $key_data | awk -v name="$1" '{
      for (i = 1; i <= NF; i++) {
         if ($i == name) { print i; }
      }
   }'
}

## Find MCTR_RA column:
#head -1 $key_data | tr "$delim" '\n' | nl | grep MCTR_RA | awk '{print $1}'
#ra_colnum=$(get_colnum MCTR_RA)
#de_colnum=$(get_colnum MCTR_DEC)
#echo "ra_colnum: $ra_colnum"
#echo "de_colnum: $de_colnum"

##--------------------------------------------------------------------------##
## Get average RA/DE position with awk:
awk -v ra_cname="MCTR_RA" -v de_cname="MCTR_DEC" '
BEGIN { 
   # start sums and counters:
   ra_sum = 0
   de_sum = 0
   ncoord = 0
}
NR == 1 {
   # make lookup table of columns
   for (i = 1; i <= NF; i++) {
      ix[$i] = i
   }
}
NR  > 1 {
   # sum RA/DE positions:
   this_ra = $ix[ra_cname]
   this_de = $ix[de_cname]
   if ((this_ra == "___") || (this_de == "___")) next;
   #printf "%s %s\n", $ix[ra_cname], $ix[de_cname]
   ra_sum += this_ra
   de_sum += this_de
   ncoord += 1
}
END {
   ra_avg = ra_sum / ncoord
   de_avg = de_sum / ncoord
   printf "RA avg: %10.3f\n", ra_avg
   printf "DE avg: %10.3f\n", de_avg
}' $key_data



##--------------------------------------------------------------------------##
## Make working data copy without header:
#sed '1 d' $key_data | grep -v ___ > $foo
#head $foo
#outliers --stats -c $ra_colnum $foo | tail -n4


##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
#[ -f $baz ] && rm -f $baz
#[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (04_central_radec.sh):
#---------------------------------------------------------------------
#
#  2023-06-02:
#     -- Increased script_version to 0.01.
#     -- First created 04_central_radec.sh.
#

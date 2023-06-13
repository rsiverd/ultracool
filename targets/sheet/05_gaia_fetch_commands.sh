#!/bin/bash
#
# Build a script to perform preprocessing cleanup on all downloaded images
# for UltracoolSheet targets.
#
# Rob Siverd
# Created:      2021-12-16
# Last updated: 2021-12-16
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

## Coordinate sanitizer:
sanitize_coord () {
   echo $1 | sed 's/s$//' | tr 'hmd' ':'
}

## Decimal degree coordinates from a target file:
get_deg_coords () {
   radec=( $(head -1 $1 | awk '{ print $1,$2 }') )
   #echo "radec: ${radec[*]}"
   deg_ra=$(sanitize_coord ${radec[0]})
   deg_de=$(sanitize_coord ${radec[1]})
   #echo "deg_ra: $deg_ra"
   #echo "deg_de: $deg_de"
   coords -H $deg_ra $deg_de > $qux; status=$?
   if [[ $status != 0 ]]; then
      echo "Failure on: ${radec[*]}"
      return 1
   fi
   deg_coo=( $(sed -n '2,3 p' $qux | awk '{ print $2 }') )
   echo ${deg_coo[*]}
}

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

cmds_list_fwd="z5_fetch_gaia_fwd.sh"
#cmds_list_rev="z5_fetch_gaia_rev.sh"
#cmds_list_rdm="z5_fetch_gaia_rdm.sh"

storage_path="~/ucd_project/ucd_targets/everything"
storage_path="/net/krang.accre.vanderbilt.edu/fs0/rjs_data/ucd_project/ucd_targets/everything"

## Target files with coordinates:
targfiles_dir=$(readlink -f everything)
[ -d $targfiles_dir ] || PauseAbort "Can't find directory: $targfiles_dir"

## Make maste list of folders in storage (attempted downloads):
short_names=( $(ls $storage_path | grep -v z_metadata | grep -v z_missing) )
#short_names=( $(find $storage_path -type d -maxdepth 1 \
#   | sort | grep -v z_metadata) )
ntargets=${#short_names[*]}
echo "ntargets: $ntargets"

## Restrict coordinate retrieval to targets with SST ephemeides (have data):
rm $foo 2>/dev/null
search_radius=0.3
for sname in ${short_names[*]}; do
   #echo "sname: $sname"
   eph_file="sst_eph_${sname}.csv"
   targ_dir="${storage_path}/$sname"
   eph_path="${targ_dir}/${eph_file}"
   target_file="${targfiles_dir}/${sname}.txt"
   [[ -f $target_file ]] || PauseAbort "Can't find file: $target_file"
   if [[ ! -f $eph_path ]]; then
      recho "Missing $eph_path \n"
      #echo "sname: $sname"
      continue
   fi
   deg_coo=( $(get_deg_coords $target_file) ); status=$?
   if [[ $status != 0 ]]; then
      echo BARF BARF BARF
      echo "Choked on this:"
      cat $target_file
      exit 1
      continue
   fi
   #cat $target_file
   #get_deg_coords $target_file
   #echo "deg_coo: ${deg_coo[*]}"
   #exit
   #continue
   gaia_file="gaia_${sname}.csv"
   gaia_path="${targ_dir}/${gaia_file}"
   fetch_cmd="./fetch_gaia_nearby.py -R $search_radius"
   fetch_cmd+=" ${deg_coo[*]}"
   fetch_cmd+=" -o $gaia_path"
   #echo "eph_path: $eph_path"
   #cmde "ls $eph_path >> $foo"
   if [[ -f $gaia_path ]]; then
      echo "CSV exists: $gaia_path"
   else
      echo "$fetch_cmd" >> $foo
   fi
done

if [[ ! -f $foo ]]; then
   echo "Nothing left to retrieve!"
   exit 0
fi
cmde "wc -l $foo"
echo "ntargets: $ntargets"

#rm $foo 2>/dev/null
#for sname in ${short_names[*]}; do
#   echo "sname: $sname"
#   eph_file="sst_eph_${sname}.csv"
#   targ_dir="${storage_path}/$sname"
#   eph_path="${targ_dir}/${eph_file}"
#   target_file="${targfiles_dir}/${sname}.txt"
#   [ -f $target_file ] || PauseAbort "Can't find file: $target_file"
#   fetch_cmd="./02_clean_all_spitzer.py -W -I ${targ_dir}/"
#   fetch_cmd+=" -t $target_file --ignore_off_target -E ${eph_path}"
#   echo "$fetch_cmd" >> $foo
#done

## Save commands list:
#cmde "tac     $foo > $cmds_list_rev" || exit $?
#cmde "sort -R $foo > $cmds_list_rdm" || exit $?
cmde "mv -f   $foo   $cmds_list_fwd" || exit $?


##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (05_gaia_fetch_commands.sh):
#---------------------------------------------------------------------
#
#  2021-11-17:
#     -- Increased script_version to 0.10.
#     -- First created 05_gaia_fetch_commands.sh.
#

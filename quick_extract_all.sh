#!/bin/bash
#
# Extract stars from all project images and save extended catalogs for
# further analysis.
#
# Rob Siverd
# Created:      2019-10-15
# Last updated: 2019-10-15
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

## Suffix adjuster:
suffix_swap () {
   local filename="$1"
   echo $filename | sed "s|$2|$3|"
}

## Extractor script:
exscript="./extract_and_match_gaia.py"
[ -f $exscript ] || PauseAbort "Can't find file: $exscript"

## Activate virtualenv:
venv_start="$HOME/venv/astrom/bin/activate"
[ -f $venv_start ] || PauseAbort "Can't find file: $venv_start"
cmde "source $venv_start" || exit $?

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

src_dir="ucd_data"
cat_dir="ucd_fcat"

yecho "Listing CBCD images ... "
vcmde "ls $src_dir/SPITZER_*_cbcd.fits | sed 's|^.*/||' > $foo" || exit $?
image_list=( `cat $foo` )
total=${#image_list[*]}
gecho "done. Found $total to process.\n"
echo

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Iterate over file list and process everything:
ntodo=100
count=0
nproc=0
for cbcd_ibase in ${image_list[*]}; do
   (( count++ ))
   yecho "\rImage $count of $total $cbcd_ibase ... "
   cbcd_ipath="$src_dir/$cbcd_ibase"
   cbun_ipath="$src_dir/$(suffix_swap $cbcd_ibase cbcd.fits cbunc.fits)"
   fcat_ipath="$cat_dir/$(suffix_swap $cbcd_ibase cbcd.fits  fcat.fits)"
   if [ -f $fcat_ipath ]; then
      gecho "already processed!   "
      continue
   fi
   recho "needs extraction ...\n"

   # ensure input files exist:
   if [ ! -f $cbcd_ipath ] || [ ! -f $cbun_ipath ]; then
      Recho "Input file(s) not found!!\n"
      continue
   fi

   (( nproc++ ))
   args="-i $cbcd_ipath -u $cbun_ipath -o $fcat_ipath"
   cmde "$exscript $args" || exit $?

   [ $ntodo -gt 0 ] && [ $nproc -ge $ntodo ] && break
done


##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (quick_extract_all.sh):
#---------------------------------------------------------------------
#
#  2019-10-15:
#     -- Increased script_version to 0.10.
#     -- First created quick_extract_all.sh.
#

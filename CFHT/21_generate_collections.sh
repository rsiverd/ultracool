#!/bin/bash
#
# Generate collection files for CFHT/Wircam test run.
#
# Rob Siverd
# Created:      2023-07-26
# Last updated: 2024-06-03
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

## Colors:
ENDC="\033[0m"
BRED="\033[1;31m"         # Bold red
BGREEN="\033[1;32m"       # Bold green
BYELLOW="\033[1;33m"      # Bold yellow
BBLUE="\033[1;34m"        # Bold blue
BMAGENTA="\033[1;35m"     # Bold magenta
BCYAN="\033[1;36m"        # Bold cyan
BWHITE="\033[1;37m"       # Bold white
NRED="\033[0;31m"         # Normal red
NGREEN="\033[0;32m"       # Normal green
NYELLOW="\033[0;33m"      # Normal yellow
NBLUE="\033[0;34m"        # Normal blue
NMAGENTA="\033[0;35m"     # Normal magenta
NCYAN="\033[0;36m"        # Normal cyan
NWHITE="\033[0;37m"       # Normal white

##--------------------------------------------------------------------------##
# Verbosity >= 0:
cmde () {
   echo -e "$NCYAN""$1""$ENDC"
   eval $1
}

Cmde () {
   echo -e "$BCYAN""$1""$ENDC"
   eval $1
}

# red:
 recho () { echo -ne "$NRED""$1""$ENDC" ; }
vrecho () {
   if [[ $VERBOSE -gt 0 ]] || [[ $verbose -gt 0 ]] || [[ $vlevel -ge 1 ]]; then
      echo -ne "$NRED""$1""$ENDC"
   fi
}
 Recho () { echo -ne "$BRED""$1""$ENDC" ; }
vRecho () {
   if [[ $VERBOSE -gt 0 ]] || [[ $verbose -gt 0 ]] || [[ $vlevel -ge 1 ]]; then
      echo -ne "$BRED""$1""$ENDC"
   fi
}

##--------------------------------------------------------------------------##

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
need_exec+=( awk cat sed tr ) #FuncDef
#need_exec+=( shuf shuffle sort ) # for randomization
for need in ${need_exec[*]}; do
   if ! ( /usr/bin/which $need >& /dev/null ); then
      echo "Error:  can't find '$need' in PATH !!" >&2
      exit 1
   fi
done

### Helper function definitions:
#fd_args="--argchk --colors --cmde --echo"
##fd_args+=" --Critical"
##fd_args+=" --rowwrite"
##fd_args+=" --timers"
##fd_args+=" --warnings"
#FuncDef $fd_args >/dev/null || exit $?
#eval "$(FuncDef $fd_args)"  || exit $?

## Check for arguments:
usage () { 
   Recho "\nSyntax: $this_prog target_name /path/to/data/by_runid --START\n\n"
   #Recho "\nSyntax: $this_prog arg1\n\n"
}
if [ "$3" != "--START" ]; then
#if [ -z "$1" ]; then
   usage >&2
   exit 1
fi

## Collect args, check for required things:
targ_name="$1"
runid_dir="$2"
colls_dir="collections"
echo "targ_name: '$targ_name'"
echo "runid_dir: '$runid_dir'"
if [[ ! -d $runid_dir ]]; then
   echo "Can't find directory: '$runid_dir'" >&2
   exit 1
fi
cmde "mkdir -p $colls_dir" || exit $?

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Sort by obsid:
obsid_sorted () {
   paste <(sed 's|^.*/||g' $1 | cut -d_ -f3 | cut -d. -f1) $1 \
      | sort -k1 | awk '{ print $2 }'
}

### List catalogs, sorted by obsid:
### Syntax: list_catalogs_sorted /path/to/runid_dir
#list_catalogs_sorted () {
#   data_dir="$1"
#
#}

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## For now, just use fixed fcat files:
wircam_filters=( J H2 )
cat_suffixes=( fcat ) #fcat pcat mcat )
cat_flavors=( fixed ) #eph fixed )
#cflav="fixed"
#ctype="fcat"
#find $runid_dir -type f -name "*${cflav}*${ctype}" > $foo
#cmde "wc -l $foo"

## Exclude anything listed here from collections:
exclude_file="exclusions.txt"
nexclude=$(cat $exclude_file 2>/dev/null | wc -l)

## Breakdown by filter:
for suff in ${cat_suffixes[*]}; do
   echo "Catalog type: $suff"
   #find $runid_dir -type f -name "*${ctype}" > $foo
   find $runid_dir -type f -name "*${suff}" > $foo
   if [[ $nexclude -gt 0 ]]; then
      n1=$(cat $foo | wc -l)
      cat $foo | grep -v -f $exclude_file > $bar
      mv -f $bar $foo
      n2=$(cat $foo | wc -l)
      echo "Files before/after exclusion: $n1 / $n2"
   fi
   #cmde "wc -l $foo"
   ncat=$(cat $foo | wc -l)
   echo "Found $ncat $suff file(s)."
   if [[ $ncat -lt 1 ]]; then
      echo "Nothing to see here!"
      continue
   fi

   # Re-order files by obsid number:
   obsid_sorted $foo > $bar
   mv -f $bar $foo

   # Create collections:
   for ppalg in ${cat_flavors[*]}; do
      grep "${ppalg}.fits" $foo > $bar
      for ifilt in ${wircam_filters[*]}; do
         echo "-------"
         chtag="wircam_${ifilt}"
         csave="${colls_dir}/${targ_name}_${chtag}_${ppalg}_${suff}.txt"
         echo "csave: $csave"
         grep "$chtag" $bar > $baz
         #head $baz
         cmde "sort $baz > $qux"  || exit $?
         cmde "mv -f $qux  $baz"  || exit $?
         cmde "mv -f $baz $csave" || exit $?
      done
   done
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
# CHANGELOG (21_generate_collections.sh):
#---------------------------------------------------------------------
#
#  2023-07-26:
#     -- Increased script_version to 0.01.
#     -- First created 21_generate_collections.sh.
#

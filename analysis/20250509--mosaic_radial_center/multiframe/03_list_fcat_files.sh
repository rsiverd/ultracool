#!/bin/bash
#
# List the fcat files from the folders I have identified. Organize by image.
#
# Rob Siverd
# Created:      2026-01-13
# Last updated: 2026-01-27
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
[[ -d /dev/shm ]] && [[ -w /dev/shm ]] && tmp_root="/dev/shm"
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
jnk_cleanup='for X in $jnk ; do [[ -f $X ]] && cmde "rm -vf $X" ; done'
trap "$jnk_cleanup >&2" EXIT
##trap '[[ -d $tmp_dir ]] && cmde "rm -vrf $tmp_dir"' EXIT
#trap "[[ -d $tmp_dir ]] && $dir_cleanup >&2" EXIT
#trap "[[ -d $tmp_dir ]] && $dir_cleanup >&2; $jnk_cleanup >&2" EXIT
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
   Recho "\nSyntax: $this_prog folder_list\n\n"
}
#if [[ "$1" != "--START" ]]; then
if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi
folder_list="$1"
[[ -f $folder_list ]] || PauseAbort "Can't find file: $folder_list"

grade_table="fcat_grades.txt"  # contains: fcat, qsograde (all catalogs)
highq_table="fbase_good.txt"   # contains: unique fbases (QSOGRADE<=cutoff)
fpath_table="fcat_paths.csv"   # contains: fbase, 4x fcat, qsograde (all)

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## List the 'fixed' fcat catalogs with tuned-up WCS:
#find $(cat $folder_list) -type f | grep fcat | sort
find $(cat $folder_list) -type f | grep 'fixed.fits.fz.fcat$' | sort > $foo
nstart=$(cat $foo | wc -l)
echo "Start with $nstart sensor-frames."

## Trim it to a few for testing ...
#sort -R $foo | head -300 > $bar; mv -f $bar $foo
#cat $foo
#head $foo > $bar

## Extract QSOGRADE value from stored header and filter:
maxgrade=3
echo "Analyze QSOGRADE (max=$maxgrade) ..."
cmde "imhget --progress -nE IMGHEADER QSOGRADE -l $foo -o $bar" || exit $?
#head $bar

## Build and save a table of QSOGRADE:
paste $foo $bar | awk '{ print $1, $3 }' > $baz
cmde "mv -f $baz $grade_table" || exit $?
#exit
#paste $foo $bar | awk -v gmax=$maxgrade '$3 <= gmax { print $1,$3 }' > $qux
#paste $foo $bar | awk -v gmax=$maxgrade            '{ print $1,$3 }' > $baz
#cmde "mv -f $baz $foo" || exit $?
#cmde "mv -f $baz $foo" || exit $?
#nfinal=$(cat $baz | wc -l)
#echo "Found $nfinal sensor-frames with QSOGRADE<=$maxgrade."

## Make high-quality image list.
## 1) Select images by QSOGRADE
## 2) Extract unique basenames of catalogs
## 3) Sort unique basenames by image number
## 4) Save result for later
bstart=$(basename -a $(awk '{print $1}' $grade_table) | sort -u | wc -l)
echo "Start with $bstart unique catalog bases."
awk -v gmax=$maxgrade '$2 <= gmax { print $1 }' $grade_table \
   | sed 's|^.*/||' | sort -u > $baz
bfinal=$(cat $baz | wc -l)
echo "Found $bfinal bases with QSOGRADE<=$maxgrade."
#head $baz
paste $baz <(cut -d_ -f3 $baz) | sort -k2 | awk '{print $1}' > $qux
cmde "mv -f $qux $highq_table" || exit $?

## Extract unique basenames (preserve order):
basename -a $(cat $foo) | sort -u > $baz
paste $baz <(cut -d_ -f3 $baz) | sort -k2 | awk '{print $1}' > $qux
#uniq_fbase=( $(basename -a $(cat $foo) | sort -u) )
uniq_fbase=( $(cat $qux) )
nfbase=${#uniq_fbase[*]}
echo "Found $nfbase unique fcat bases."

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

#perform_diags=yes
perform_diags=

if [[ -n "$perform_diags" ]]; then
   # DIAGNOSTICS: are we missing quadrants for any images?
   mecho "`RowWrite 75 -`\n"
   yecho "Check for sensor completeness ...\n"
   total=${#uniq_fbase[*]}
   for fbase in ${uniq_fbase[*]}; do
      #grep $fbase $foo | wc -l
      nhits=$(grep $fbase $foo | wc -l)
      if [[ $nhits -ne 4 ]]; then
         echo "ODDITY: $fbase ..."
         grep $fbase $foo
         echo
      fi
   done
   echo

   # DIAGNOSTICS: is QSOGRADE universial or can it vary among sensors?
   mecho "`RowWrite 75 -`\n"
   yecho "Verify that QSOGRADE matches among sensors ...\n"
   anybad=0
   total=${#uniq_fbase[*]}
   count=0
   for fbase in ${uniq_fbase[*]}; do
      (( count++ ))
      echo -ne "\rChecking image $count of $total ... "
      ngrades=$(grep $fbase $grade_table | awk '{print $2}' | sort -u | wc -l)
      #echo "ngrades: $ngrades"
      #nhits=$(grep $fbase $foo | wc -l)
      if [[ $ngrades -ne 1 ]]; then
         echo -e "QSOGRADE mismatch: $fbase ..."
         grep $fbase $baz
         (( anybad++ ))
      fi
   done
   echo "done."
   echo "Found $anybad image(s) with differing QSOGRADE among sensors."
fi
#exit

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##
#
## Make sure each base has all four catalogs:
echo "fbase,ne_fpath,nw_fpath,se_fpath,sw_fpath,qsograde" > $bar
for fbase in ${uniq_fbase[*]}; do
   echo "fbase: $fbase"
   #grep $fbase $baz
   #grep -m1 $fbase $baz
   #grep -m1 $fbase $baz | awk '{print $2}'
   #qsograde=$(grep -m1 $fbase $baz | awk '{print $2}')
   qsograde=$(grep -m1 $fbase $grade_table | awk '{print $2}')
   #echo "qsograde: $qsograde"
   #exit
   #grep $fbase $foo
   fpaths=( $(grep $fbase $foo | sort) )
   hits=${#fpaths[*]}
   echo "$fbase :: found $hits fixed fcat files"
   #exit
   if [[ $hits -ne 4 ]]; then
      echo "Catalog base $fbase does not have all four sensors ..."
      grep $fbase $foo
      echo "SKIPPING!"
      continue
      #echo "UNEXPECTED!!"
      #exit 1
   fi
   # add an entry:
   echo "$fbase ${fpaths[*]} $qsograde" | tr ' ' ',' >> $bar
   echo
done

## Save the result:
cmde "mv -f $bar $fpath_table" || exit $?

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (02_list_fcat_files.sh):
#---------------------------------------------------------------------
#
#  2026-01-13:
#     -- Increased script_version to 0.01.
#     -- First created 02_list_fcat_files.sh.
#

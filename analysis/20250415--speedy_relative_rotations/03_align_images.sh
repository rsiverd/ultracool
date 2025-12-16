#!/bin/bash
#
# Align images that have been unpacked. Each folder can be handled
# independently so this script can be very simple.
#
# Rob Siverd
# Created:      2025-04-15
# Last updated: 2025-04-15
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
   Recho "\nSyntax: $this_prog --START\n\n"
   #Recho "\nSyntax: $this_prog arg1\n\n"
}
if [[ "$1" != "--START" ]]; then
#if [[ -z "$1" ]]; then
   usage >&2
   exit 1
fi

src_dir="unpacked"
[[ -d $src_dir ]] || PauseAbort "Can't find directory: $src_dir"

#dst_dir1="aligned1"
#dst_dir2="aligned2"
#cmde "mkdir -p $dst_dir1 $dst_dir2" || exit $?

## Use a process-specific scratch folder to prevent collisions between
## multiple instances of this script. It also keeps the CWD clean.
scratch="./working$$"
cmde "mkdir -p $scratch" || exit $?
jnk+=" $scratch"
#trap "$jnk_cleanup >&2; rm -rf $scratch" EXIT

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Iterate over unpacked folders. QRUNIDs where _SW exists are probably done.
## Start by making a list of QRUNIDs we should focus on.
ls $src_dir | grep _SW | cut -d_ -f1 > $foo
#use_runids=( $(ls $src_dir | grep _SW | cut -d_ -f1) )

ls -d $src_dir/* | grep -f $foo > $bar
img_dirs=( $(cat $bar | sort -R) )
#img_dirs=( $(cat $bar | sort ) )
#img_dirs=( $(cat $bar | grep 13BQ06) )
#img_dirs=( $(ls -d $src_dir/* | grep -f $foo) )
#img_dirs=( $(shuffle ${img_dirs[*]}) )

medra=294.591229
medde=35.118382



rm -f $foo $bar
cmde "cd $scratch"
for imdir in ${img_dirs[*]}; do 
   imrel="../$imdir"
   echo "imdir: $imdir"
   # make sure we have J-band images:
   #n_jband=$(ls $imdir/wircam_J*fits 2>/dev/null | wc -l)
   yecho "Listing J-band images ... \n"
   n_jband=$(ls $imrel/wircam_J*fits 2>/dev/null | tee $foo | wc -l)
   if [[ $n_jband -lt 3 ]]; then
      echo "Only have $n_jband J-band images ... skip"
      continue
   fi
   runquad=$(basename $imdir)
   echo "runquad: $runquad"
   #first=$(ls $imdir/wircam_J*fits | head -1)

   # Improvement: ignore images with QSOGRADE >= 3. Higher than this
   # is CERTAINLY not suitable for the reference image and is possibly
   # not worth aligning at all (ignored downstream).
   #cmde "ls $imrel/wircam_J*fits"
   yecho "Checking QSOGRADE ...\n"
   cmde "imhget -l $foo QSOGRADE -o $bar" || exit $?
   cmde "cat $bar"

   # Ensure we have SOME that are decent:
   ndecent=$(awk '$2 < 3 { print $1 }' $bar | tee $baz | wc -l)
   echo "ndecent: $ndecent"
   if [[ $ndecent -lt 3 ]]; then
      echo "Only $ndecent decent images ... skip this RUNID"
      continue
   fi

   cmde "imhget -l $baz MCTR_RA MCTR_DEC -o $qux"
   #cmde "cat $qux"
   first=$(awk -v medra=$medra -v medde=$medde '
   function abs(value) { return ( value < 0 ? -value : value ); }
   {
      radiff = abs($2 - medra)
      dediff = abs($3 - medde)
      totsep = sqrt(radiff*radiff + dediff*dediff)
      printf "%s %.4f dra=%.4f, dde=%.4f\n", $1, totsep, radiff, dediff
   }' $qux | sort -nk2 | head -1 | awk '{print $1}')
   #echo "first: $first"
   #exit
   # Ensure we have ONE top-notch:
   #ngood=$(awk '$2 == 1 { print $1 }' $bar | tee $qux | wc -l)
   #echo "ngood: $ngood"
   #if [[ $ngood -lt 2 ]]; then
   #   PauseAbort "TROUBLE!"
   #fi
   #echo "jnk: $jnk"
   #exit

   #first=$(head -1 $baz)
   #first=$(ls $imrel/wircam_J*fits | head -1)
   echo "first: $first"

   for poly in 1 2 ; do
      save_dir="../aligned_p${poly}/${runquad}"
      cmde "mkdir -p $save_dir" || exit $?
      #cmde "sexterp -D$save_dir -p$poly -T5 -r$first ${imdir}/wircam_J*.fits"
      cmde "sexterp -D$save_dir -p$poly -T5 -r$first ${imrel}/wircam_J*.fits"
      #save_dir="${dst_dir1}"
   done

   #cmde "rm log_interp* log_xy.* coeff.txt"
   #cmde "rm isis.terp.*.params dates-sexterp-*"
   #break
done
cmde "cd .."
cmde "rm -rf $scratch"

##--------------------------------------------------------------------------##
## Clean up:
#[[ -d $tmp_dir ]] && [[ -O $tmp_dir ]] && rm -rf $tmp_dir
[[ -f $foo ]] && rm -f $foo
[[ -f $bar ]] && rm -f $bar
[[ -f $baz ]] && rm -f $baz
[[ -f $qux ]] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (03_align_images.sh):
#---------------------------------------------------------------------
#
#  2025-04-15:
#     -- Increased script_version to 0.01.
#     -- First created 03_align_images.sh.
#

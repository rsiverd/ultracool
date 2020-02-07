#!/bin/bash
#
# Identify sources in Spitzer images using standard SExtractor for comparison.
#
# Rob Siverd
# Created:      2019-11-11
# Last updated: 2019-11-11
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
ntodo=0
shuffle=0
weighted=0
confirmed=0
image_type=""
src_folder=""
dst_folder=""

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
need_exec+=( awk cat FuncDef getopt runsex sed sex tr )
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

##--------------------------------------------------------------------------##
## Syntax / help menu:
usage () {
   cat << EOH

Usage: $this_prog [options] --START <flavor> <folders>
Extract all Spitzer images using standard SExtractor.
Version: $script_version

Program options:
    -c, --clobber       allow overwrite of output files
    -n, --ntodo=NUM     stop after processing NUM images [def: all]
    -r, --random        randomize processing order (for parallel operation)

Input/output folders (REQUIRED):
    -I, --srcdir=SRC    look for input data in SRC
    -O, --dstdir=DST    save output catalogs in DST

Data flavor (REQUIRED):  
        --original      use original images
        --cosclean      use 'clean' images with cosmics removed

Processing options:
        --rmswei        use uncertainty image as RMS map if available

Other options:
        --deps          return list of required programs (for parent script)
    -h, --help          display this page
    -q, --quiet         less on-screen info and progress reporting
    -t, --timer         measure and report script execution time
    -v, --verbose       more on-screen info and progress reporting

EOH
#        --debug         extra verbosity to assist bug hunting
#    -f, --force         force redo of whatever this script does
#    -f, --force         allow clobbering of output file
#    -o, --output=FILE   send output to FILE
}

##--------------------------------------------------------------------------##
## Parse command line with getopt (reorders and stores CL args):
s_opts="cn:rI:O:rhqtv" # f
l_opts="START,clobber,ntodo:,random,srcdir:,dstdir:,original,cosclean"
l_opts+=",rmswei,debug,deps,help,quiet,timer,verbose" # force
args=`getopt -n $this_prog -o $s_opts -l $l_opts -- "$@"` ; failed=$?

## Check for parsing errors:
if [ $failed -ne 0 ]; then 
   echo "Try \`$this_prog --help' for more information." >&2
   exit 1
fi

## Change the arguments in the environment (e.g., $1, $2, ... $N):
eval set -- "$args"

## Loop through arguments (shift removes them from environment):
while true ; do
   case $1 in
      #-------------------------------------------------------------------
      --START)
         confirmed=1
         shift
         ;;
      #-------------------------------------------------------------------
      -c|--clobber)
         [ $vlevel -ge 0 ] && yecho "Enabled output clobber!\n" >&2
         clobber=1
         shift
         ;;
      -r|--random)
         [ $vlevel -ge 0 ] && yecho "Randomizing order!\n" >&2
         shuffle=1
         shift
         ;;
      --rmswei)
         [ $vlevel -ge 0 ] && yecho "Using weight-images!\n" >&2
         weighted=1
         shift
         ;;
      #-------------------------------------------------------------------
      # Data flavors:
      --cosclean|--original)
         image_type="${1#--}"
         [ $vlevel -ge 0 ] && yecho "Selected image type: $image_type \n" >&2
         shift
         ;;
      #-------------------------------------------------------------------
      -n|--ntodo)
         case $2 in
            -*)
               msg="Option -n|--ntodo needs a positive integer argument!"
               Recho "\n${msg}\n" >&2
               usage >&2
               exit 1
               ;;
            *)
      ##       if !( num_check_pass $2 ); then
      ##       if !( num_check_pass $2 ) || (is_negative $2); then
               if !( int_check_pass $2 ) || [ $2 -lt 0 ]; then
                  Recho "Invalid ntodo: " >&2 ; Yecho "$2 \n\n" >&2
                  exit 1
               fi
               ntodo=$2
               ;;
         esac
         [ $vlevel -ge 0 ] && yecho "Stopping after $ntodo images.\n" >&2
         shift 2
         ;;
      #-------------------------------------------------------------------
      # Input folder with images:
      -I|--srcdir)
         case $2 in
            -*)
               Recho "\nOption -I|--srcdir requires an argument!\n" >&2
               usage >&2
               exit 1
               ;;
            *)
               src_folder="$2"
               # check value here ...
               #if [ $clobber -eq 0 ] && [ -f $save_file ]; then
               #   Recho "\nFile already exists: " >&2
               #   Yecho "$save_file \n\n" >&2
               #   exit 1
               #fi
               ;;
         esac
         [ $vlevel -ge 0 ] && yecho "Images from folder: '$src_folder'\n" >&2
         shift 2
         ;;
      #-------------------------------------------------------------------
      # Output folder for catalogs:
      -O|--dstdir)
         case $2 in
            -*)
               Recho "\nOption -o|--outdir requires an argument!\n" >&2
               usage >&2
               exit 1
               ;;
            *)
               dst_folder="$2"
               # check value here ...
               #if [ $clobber -eq 0 ] && [ -f $save_file ]; then
               #   Recho "\nFile already exists: " >&2
               #   Yecho "$save_file \n\n" >&2
               #   exit 1
               #fi
               ;;
         esac
         [ $vlevel -ge 0 ] && yecho "Output to: '$dst_folder' \n" >&2
         shift 2
         ;;
      #-------------------------------------------------------------------
      # Additional options (output control etc.):
      --debug)
         yecho "Debugging mode enabled!\n"
         debug=1
         shift
         ;;
      --deps)
         echo ${need_exec[*]}
         exit 0
         ;;
     #-f|--force)
     #   [ $vlevel -ge 0 ] && yecho "Output clobbering enabled!\n" >&2
     #   [ $vlevel -ge 0 ] && yecho "Forcing <WHAT THIS SCRIPT DOES>!\n" >&2
     #   clobber=1
     #   force=1
     #   shift
     #   ;;
      -h|--help)
         usage
         exit 0
         ;;
      -q|--quiet)
         (( vlevel-- ))
         shift
         ;;
      -t|--timer)
         [ $vlevel -ge 1 ] && yecho "Timing script execution!\n" >&2
         timer=1
         shift
         ;;
      -v|--verbose)
         (( vlevel++ ))
         shift
         ;;
      #-------------------------------------------------------------------
      --)
         shift
         break 
         ;;
      *)
         echo -e "\n$this_prog error: unhandled option: $1 \n" >&2
         exit 1 
         ;;
      #-------------------------------------------------------------------
   esac
done

## Check for an appropriate number of arguments:
if [ $confirmed -ne 1 ]; then
   usage >&2
   exit 1
fi

## Image type (clean / original) is required:
if [ -z "$image_type" ]; then
   Recho "\nError: no image type selected!\n" >&2
   usage >&2
   exit 1
fi

## Input folder is required:
if [ -z "$src_folder" ]; then
   Recho "\nError: no input folder specified!\n" >&2
   usage >&2
   exit 1
fi

## Output folder is required:
if [ -z "$dst_folder" ]; then
   Recho "\nError: no output folder specified!\n" >&2
   usage >&2
   exit 1
fi

[ $debug -eq 1 ] && vlevel=3
[ $vlevel -gt 1 ] && echo "Verbosity: $vlevel" >&2

## Input folder must exist:
[ -d $src_folder ] || PauseAbort "Can't find directory: $src_folder"

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##

## Files to process:
yecho "Listing input images ... "
img_list=()
case $image_type in
   original)
      imsuff="cbcd"
      #vcmde "ls $src_folder/SPIT*_cbcd.fits > $foo" || exit $?
      ;;
   cosclean)
      imsuff="clean"
      #vcmde "ls $src_folder/SPIT*_clean.fits > $foo" || exit $?
      ;;
   *) PauseAbort "Unhandled image_type: '$image_type'" ;;
esac
vcmde "ls $src_folder/SPIT*_${imsuff}.fits > $foo" || exit $?
total=$(cat $foo | wc -l)
gecho "done. Found $total images.\n"

## Randomize order if requested:
if [ $shuffle -eq 1 ]; then
   img_list=( `sort -R $foo` )
else
   img_list=( `cat $foo` )
fi

##--------------------------------------------------------------------------##
##------------------         Extraction Parameters          ----------------##
##--------------------------------------------------------------------------##

rs_opts="-q -F -T3 -g 1.0"
rs_opts+=" -p X_IMAGE -p Y_IMAGE -p X2_IMAGE -p Y2_IMAGE -p XY_IMAGE"
rs_opts+=" -p ERRX2_IMAGE -p ERRY2_IMAGE -p ERRXY_IMAGE"
rs_opts+=" -p X2WIN_IMAGE -p Y2WIN_IMAGE -p XYWIN_IMAGE"
rs_opts+=" -p ERRX2WIN_IMAGE -p ERRY2WIN_IMAGE -p ERRXYWIN_IMAGE"

##--------------------------------------------------------------------------##
##------------------        Extract Object Catalogs         ----------------##
##--------------------------------------------------------------------------##

count=0
nproc=0
#ntodo=0
#ntodo=5
for image in ${img_list[*]}; do
   ibase="${image##*/}"
   cpath="${dst_folder}/${ibase}.cat"
   yecho "\rChecking $ibase ($((++count)) of $total) ... "
   if [ $clobber -eq 0 ] && [ -f $cpath ]; then
      gecho "already exists!   "
      continue
   fi
   recho "needs work ... "
   (( nproc++ ))

   # locate uncertainty image (if needed):
   use_opts="$rs_opts"
   if [ $weighted -eq 1 ]; then
      wpath="${image/$imsuff/cbunc}"
      if [ -f "$wpath" ]; then
         gecho "(error-image found) "
         #echo "Found error-image: '$wpath'"
         #cmde "ls $wpath"
         use_opts+=" --rmswei=$wpath"
      else
         recho "(error-image missing!) "
      fi
   fi
   echo
   #exit 0

   # extract sources:
   xfr_file="${cpath}.tmp$$"
   cmde "runsex $use_opts -o $baz $image" || exit $?
   cmde "mv -f $baz $xfr_file"            || exit $?
   cmde "mv -f $xfr_file $cpath"          || exit $?
   becho "`RowWrite 75 -`\n"

   # stop early if requested:
   [ $ntodo -gt 0 ] && [ $nproc -ge $ntodo ] && break
done
echo

Gecho "Images processed, script complete!\n"

##--------------------------------------------------------------------------##
## Clean up:
#[ -d $tmp_dir ] && [ -O $tmp_dir ] && rm -rf $tmp_dir
[ -f $foo ] && rm -f $foo
[ -f $bar ] && rm -f $bar
[ -f $baz ] && rm -f $baz
[ -f $qux ] && rm -f $qux
exit 0

######################################################################
# CHANGELOG (06_sextract_spitzer.sh):
#---------------------------------------------------------------------
#
#  2019-11-11:
#     -- Increased script_version to 0.10.
#     -- First created 06_sextract_spitzer.sh.
#

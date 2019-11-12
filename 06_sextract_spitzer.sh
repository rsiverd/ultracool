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

Usage: $this_prog [options] FILE(S)
SCRIPT DESCRIPTION GOES HERE.
Version: $script_version

Program options:
    -c, --clobber       allow overwrite of output file
    -o, --output=FILE   send output to FILE
    -r, --random        randomize processing order (for parallel operation)

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
}

##--------------------------------------------------------------------------##
## Parse command line with getopt (reorders and stores CL args):
s_opts="co:rhqtv" # f
l_opts="START,clobber,output:,random"
l_opts+=",debug,deps,help,quiet,timer,verbose" # force
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
     #--START)
     #   confirmed=1
     #   shift
     #   ;;
      #-------------------------------------------------------------------
      -c|--clobber)
         [ $vlevel -ge 0 ] && yecho "Enabled output clobber!\n" >&2
         clobber=1
         shift
         ;;
      #-------------------------------------------------------------------
     #-n|--number)
     #   case $2 in
     #      -*)
     #         msg="Option -n|--number requires an argument!"
     #         #msg="Option -n|--number needs a positive integer argument!"
     #         #msg="Option -n|--number needs a positive numerical argument!"
     #         Recho "\n${msg}\n" >&2
     #         usage >&2
     #         exit 1
     #         ;;
     #      *)
     ###       if !( num_check_pass $2 ); then
     ###       if !( num_check_pass $2 ) || (is_negative $2); then
     ###       if !( int_check_pass $2 ) || [ $2 -lt 0 ]; then
     #            Recho "Invalid value: " >&2 ; Yecho "$2 \n\n" >&2
     #            exit 1
     #         fi
     #         num_val=$2
     #         ;;
     #   esac
     #   [ $vlevel -ge 0 ] && yecho "Using value: ${num_val}\n" >&2
     #   shift 2
     #   ;;
      -o|--output)
         case $2 in
            -*)
               Recho "\nOption -o|--output requires an argument!\n" >&2
               usage >&2
               exit 1
               ;;
            *)
               save_file=$2
               # check value here ...
               if [ $clobber -eq 0 ] && [ -f $save_file ]; then
                  Recho "\nFile already exists: " >&2
                  Yecho "$save_file \n\n" >&2
                  exit 1
               fi
               ;;
         esac
         [ $vlevel -ge 0 ] && yecho "Output to: $save_file \n" >&2
         shift 2
         ;;
      -r|--random)
         [ $vlevel -ge 0 ] && yecho "Randomizing order!\n" >&2
         shuffle=1
         shift
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
#if [ -z "$1" ]; then
   usage >&2
   exit 1
fi

[ $debug -eq 1 ] && vlevel=3
[ $vlevel -gt 1 ] && echo "Verbosity: $vlevel" >&2

##**************************************************************************##
##==========================================================================##
##--------------------------------------------------------------------------##


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

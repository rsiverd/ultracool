#!/bin/bash

ENDC="\033[0m"
NCYAN="\033[0;36m"        # Normal cyan

cmde () {
   echo -e "$NCYAN""$1""$ENDC"
   eval $1
}

symlink_path="../../illustrate.py"
ill_script="illustrate.py"

#ls */gaia_*
#rm `find . -type l -name "illustrate.py"`
pickle_files=( `find . -type f -name "GSE_tuple.pickle"` )

start_dir=`pwd`
for item in ${pickle_files[*]}; do
   echo "---------------------------------------------------"
   echo "item: $item"
   pdir=`dirname $item`
   echo "pdir: $pdir"
   cmde "cd $pdir"      || exit $?

   #cmde "ls $ill_script"
   cmde "rm $ill_script 2>/dev/null"
   cmde "ln -s ../../$ill_script"
   cmde "cd $start_dir" || exit $?
   echo
done


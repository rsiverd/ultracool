# Update PYTHONPATH to include 'modules' project subfolder.

here_base=$(basename $(pwd))
#echo "here_base: $here_base"
if [ "$here_base" != "ultracool" ]; then
   echo "Please source this inside the 'ultracool' project folder!" >&2
   return 1
fi
if [ ! -d modules ]; then
   echo "Cannot find 'modules' folder!  Something is very wrong ..." >&2
   return 1
fi
module_dir="`pwd`/modules"
echo "module_dir: $module_dir"
#echo $PYTHONPATH
is_dupe=`echo $PYTHONPATH | tr ':' '\n' | grep "^${module_dir}$" | wc -l`
if [ $is_dupe -gt 0 ]; then
   echo "PYTHONPATH already configured!"
   return 0
fi
new_pypath=$(echo "${module_dir}:${PYTHONPATH}" | sed 's/:$//')
#echo "new_pypath: '$new_pypath'"
export PYTHONPATH="$new_pypath"
echo "PYTHONPATH updated."

export NUMEXPR_MAX_THREADS=8
echo "Using NUMEXPR_MAX_THREADS = $NUMEXPR_MAX_THREADS"


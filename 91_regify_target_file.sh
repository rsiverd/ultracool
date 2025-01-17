#!/bin/bash
#
# This file contains the commands needed to make a useful region file from
# targets_deg.txt.
#
# To run script, do (adjust file names as needed):
# ./91_regify_target_file.sh targets_deg.txt    # inspect manually
# ./91_regify_target_file.sh targets_deg.txt > targets.reg

if [ -z "$1" ]; then
   echo "Syntax: $0 targets_deg.txt"
   exit 1
fi

cat $1 | awk '{
   name = $1
   dra  = $2
   dde  = $3
   printf "fk5; annulus(%12.8fd, %12.8fd, 0.0015d, 0.0005d)", dra, dde
   printf " # color=red text={%s}", name
   printf "\n"
}'



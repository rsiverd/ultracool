#!/bin/bash

csv_file="$1"
reg_file="$2"
#inner_as="0.00025"
#outer_as="0.00075"
if [ $# -lt 2 ]; then
   echo "NOPE!"
   echo "Syntax: ./$0 csv_file region_file"
   exit 1
fi

awk -F, '
NR > 1 {
   ra = $7 ; dec = $9
   printf "annulus(%.7fd, %.7fd, 0.00025d, 0.00075d)\n", ra, dec
}' $csv_file > $reg_file


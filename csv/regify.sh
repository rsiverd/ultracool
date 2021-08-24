#!/bin/bash

csv_file="$1"
reg_file="$2"
#inner_as="0.00025"
#outer_as="0.00075"
if [ $# -lt 2 ]; then
   echo "NOPE!"
   echo "Syntax: $0 csv_file region_file"
   exit 1
fi

# SPITZER IRAC is ~1.2" / pixel
# Useful column names are: ra, dec, phot_g_mean_mag
awk -F, '
BEGIN {
   # define columns of interest:
   rcname = "ra"
   dcname = "dec"
   mcname = "phot_g_mean_mag"

   # magnitude limit for region file:
   max_mag = 22.0
   
   # region file annulus size:
   pxscale = 1.2                 # arcsec  per pixel (IRAC)
   deg_pix = pxscale / 3600.0    # degrees per pixel (IRAC)
   r_inner_pix = 1.0
   r_outer_pix = 3.0
   r_inner_deg = r_inner_pix * deg_pix
   r_outer_deg = r_outer_pix * deg_pix
   #r_inner_deg = 0.0005
   #r_outer_deg = 0.0015
   printf "Inner radius (deg): %9.5f\n", r_inner_deg > "/dev/stderr"
   printf "Outer radius (deg): %9.5f\n", r_outer_deg > "/dev/stderr"
}
NR == 1 {
   # get columns
   for (i=1; i<=NF; i++) {
      #printf "col %3d: %s\n", i, $i
      if (rcname == $i) { rcnum = i }
      if (dcname == $i) { dcnum = i }
      if (mcname == $i) { mcnum = i }
   }
   printf "%s is column %d\n", rcname, rcnum > "/dev/stderr"
   printf "%s is column %d\n", dcname, dcnum > "/dev/stderr"
   printf "%s is column %d\n", mcname, mcnum > "/dev/stderr"
}
NR > 1 {
   this_ra = $rcnum
   this_de = $dcnum
   this_gg = $mcnum
   if (this_gg < max_mag) {
      printf "annulus(%11.7fd, %11.7fd, %7.5fd, %7.5fd)\n", \
                  this_ra, this_de, r_inner_deg, r_outer_deg
   }
}' $csv_file > $reg_file


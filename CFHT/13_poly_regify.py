
import os, sys, time
import glob

#hh_fcat_path = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid/11AQ15/wircam_H2_1319402p.fits.fz.fcat'
#jj_fcat_path = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid/11AQ15/wircam_J_1319388p.fits.fz.fcat'

import extended_catalog
ec = extended_catalog
ccc = ec.ExtendedCatalog()

if len(sys.argv) < 3:
    sys.stderr.write("\nSyntax: %s input_fcat output_region_file\n" % sys.argv[0])
    sys.exit(1)
load_cat = sys.argv[1]
save_reg = sys.argv[2]

# Ensure input file exists:
if not os.path.isfile(load_cat):
    sys.stderr.write("\nError: file not found: %s\n" % load_cat)
    sys.exit(1)

# Load catalog:
ccc.load_from_fits(load_cat)
stars = ccc.get_catalog()

# Write to region file:
radec_r1_sec = 1.0
radec_r2_sec = 2.0
radec_r1_deg = radec_r1_sec / 3600.0
radec_r2_deg = radec_r2_sec / 3600.0
image_r1_pix = 3.0
image_r2_pix = 6.0
with open(save_reg, 'w') as sr:
    #for cra,cde in zip(stars['dra'], stars['dde']):
    #    sr.write("fk5; annulus(%.6f, %.6f, %.5fd, %.5fd)\n"
    #            % (cra, cde, radec_r1_deg, radec_r2_deg))
    # original x,y:
    for cxx,cyy in zip(stars['x'], stars['y']):
        sr.write("image; annulus(%.3f, %.3f, %.1f, %.1f) # color=red\n"
                % (cxx, cyy, image_r1_pix, image_r2_pix))
    # dewarped x,y:
    for cxx,cyy in zip(stars['xdw'], stars['ydw']):
        sr.write("image; annulus(%.3f, %.3f, %.1f, %.1f) # color=green\n"
                % (cxx, cyy, image_r1_pix, image_r2_pix))


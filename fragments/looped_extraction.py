import glob, os, sys, time

src_dir = "ucd_data"
cat_dir = "ucd_fcat"

cbcd_files = sorted(glob.glob('%s/SPITZER_*_cbcd.fits' % src_dir))

nproc = 0
for ii,cbcd_ipath in enumerate(cbcd_files, 1):
    sys.stderr.write("-----------------------------------------\n")
    cbun_ipath = cbcd_ipath.replace('cbcd', 'cbunc')
    cbcd_ibase = os.path.basename(cbcd_ipath)
    fcat_ibase = cbcd_ibase.replace('cbcd', 'fcat')
    fcat_ipath = os.path.join(cat_dir, fcat_ibase)
    sys.stderr.write("making %s ...\n" % fcat_ipath)
    if not os.path.isfile(fcat_ipath):
        nproc += 1
        %run ./extract_and_match_gaia.py -i $cbcd_ipath -u $cbun_ipath -o $fcat_ipath
    if (nproc >= 500):
        break

# -----------------------------------------------------------------------

with open('derp.reg', 'w') as ff: 
    for coo in zip(ccd_xx, ccd_yy): 
        ff.write("image; circle(%8.3f, %8.3f, 2)\n" % coo) 
                                                                                       



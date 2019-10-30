
%run ./extract_and_match_gaia.py -g ultracool/csv/gaia_wise1828.csv \ 
   -i ucd_data/SPITZER_I2_61246976_0004_0000_1_cbcd.fits            \ 
   -u ucd_data/SPITZER_I2_61246976_0004_0000_1_cbunc.fits 

from lacosmic import lacosmic

_mask_NaNs = True
#_mask_NaNs = False

lakw = {'contrast': 12.0,
        'cr_threshold':6.0,
        'neighbor_threshold':4.0,
        }
if _have_err_image:
    lakw['error'] = udata
if _mask_NaNs:
    lakw['mask'] = np.isnan(idata)

sys.stderr.write("Running LACOSMIC ... ")
tik = time.time()
clean_data, cr_mask = lacosmic(idata, **lakw)
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))


tag = '%d.%d' % (lakw['contrast'], lakw['cr_threshold'])
cln_name = 'clean_image.%s.fits' % tag
msk_name = 'cosmic_mask.%s.fits' % tag
qsave(cln_name, clean_data)
qsave(msk_name, cr_mask.astype('uint8'))


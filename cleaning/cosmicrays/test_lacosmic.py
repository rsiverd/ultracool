
import astropy.io.fits as pf
from lacosmic import lacosmic

# load images:
ipath = 'SPITZER_I2_61246976_0004_0000_1_cbcd.fits'
upath = 'SPITZER_I2_61246976_0004_0000_1_cbunc.fits'
idata, ihdrs = pf.getdata(ipath, header=True)
udata, uhdrs = pf.getdata(upath, header=True)

_mask_NaNs = True
#_mask_NaNs = False

# settings:
lakw = {'contrast': 12.0,
        'cr_threshold':6.0,
        'neighbor_threshold':4.0,
        }
if _have_err_image:
    lakw['error'] = udata
if _mask_NaNs:
    lakw['mask'] = np.isnan(idata)

# trial run:
sys.stderr.write("Running LACOSMIC ... ")
tik = time.time()
clean_data, cr_mask = lacosmic(idata, **lakw)
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))

# save results:
tag = '%d.%d' % (lakw['contrast'], lakw['cr_threshold'])
cln_name = 'clean_image.%s.fits' % tag
msk_name = 'cosmic_mask.%s.fits' % tag
qsave(cln_name, clean_data)
qsave(msk_name, cr_mask.astype('uint8'))


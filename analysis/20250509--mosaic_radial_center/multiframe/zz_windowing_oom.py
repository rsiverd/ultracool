# From inside script 41 ...

qtag = {'NE':1, 'NW':2, 'SE':3, 'SW':4}
m_xdif, m_ydif = [], []
m_fwhm, m_tsnr = [], []

for qq,ss in stars.items():
    xdif = ss['x'] - ss['wx']
    ydif = ss['x'] - ss['wx']
    fwhm = 2.0 * np.sqrt(np.log(2.) * (ss['a']**2 + ss['b']**2))
    tsnr = np.sqrt(ss['flux'])
    m_xdif.extend(xdif)
    m_ydif.extend(ydif)
    m_fwhm.extend(fwhm)
    m_tsnr.extend(tsnr)
    pass

#cols = ['x', 'y', 'wx', 'wy', 'flux', '
bigdf = pd.concat(stars.values()).copy().reset_index()
bigdf = bigdf[bigdf.npix < 1000]
xdif = bigdf['x'] - bigdf['wx']
ydif = bigdf['y'] - bigdf['wy']
lim = 0.75
rng2d = ((-lim, lim), (-lim, lim))
plt.hist2d(xdif, ydif, range=rng2d, bins=21, density=True)

pxdif = bigdf['x'] - bigdf['xpeak']
pydif = bigdf['y'] - bigdf['ypeak']
nxdif = xdif / bigdf['realerr']
nydif = ydif / bigdf['realerr']

dx_med, dx_sig = rs.calc_ls_med_IQR(xdif)
dy_med, dy_sig = rs.calc_ls_med_IQR(ydif)

sys.stderr.write(" xdif med/iqr: %s\n" % str(rs.calc_ls_med_IQR(xdif)))
sys.stderr.write(" ydif med/iqr: %s\n" % str(rs.calc_ls_med_IQR(ydif)))
sys.stderr.write("nxdif med/iqr: %s\n" % str(rs.calc_ls_med_IQR(nxdif)))
sys.stderr.write("nydif med/iqr: %s\n" % str(rs.calc_ls_med_IQR(nydif)))

lim=2
hist(nxdif, bins=51, range=(-lim, lim), density=True)
hist(nydif, bins=51, range=(-lim, lim), density=True)


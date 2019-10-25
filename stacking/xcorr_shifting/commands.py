
import glob
import fitsio
import os, sys, time
#import scipy.signal as ssig
import numpy as np
import robust_stats as rs

imlist = sorted(glob.glob('SPITZ*fits'))

imdata = [fitsio.read(ff) for ff in imlist]   
ny, nx = imdata[0].shape
padspec = (int(0.5 * ny), int(0.5 * nx))

# patch NaNs:
for frame in imdata:
    which = np.isnan(frame)
    replacement = np.median(frame[~which])
    frame[which] = replacement

# reduce to bright pixel masks:
bpmask = []
thresh = 20.0
for frame in imdata:
    pix_med, pix_iqrn = rs.calc_ls_med_IQR(frame)
    bright = (frame - pix_med >= thresh * pix_iqrn) 
    bpmask.append(bright)

# FFT-assisted cross-correlation:
def qcorr(rowcol1, rowcol2):
    npix = rowcol1.size
    cft1 = np.fft.fft(rowcol1)
    cft2 = np.fft.fft(rowcol2)
    cft2.imag *= -1.0
    corr = np.fft.ifft(cft1 * cft2)
    nshift = corr.argmax()
    if (nshift > 0.5*npix):
        nshift -= npix
    return nshift

## Collapse along rows/columns, add zero padding:
#xsmashed = [rr/rr.max() for rr in [np.sum(im, axis=1) for im in zpdata]]
#ysmashed = [cc/cc.max() for cc in [np.sum(im, axis=0) for im in zpdata]]
#xsmashed = [np.pad(rr, ny) for rr in [np.sum(im, axis=1) for im in imdata]]
#ysmashed = [np.pad(cc, nx) for cc in [np.sum(im, axis=0) for im in imdata]]
xsmashed = [np.pad(rr, ny) for rr in [np.sum(im, axis=1) for im in bpmask]]
ysmashed = [np.pad(cc, nx) for cc in [np.sum(im, axis=0) for im in bpmask]]

## Cross-correlate to find pixel offsets:
ynudges = [qcorr(xsmashed[0], rr) for rr in xsmashed]
xnudges = [qcorr(ysmashed[0], cc) for cc in ysmashed]

# FIXME: should roll a NaN-padded image so stacking works properly ...
for ff,im,dx,dy in zip(imlist, imdata, xnudges, ynudges):
    npdata = np.pad(im, padspec, constant_values=np.nan)
    r2data = np.roll(np.roll(npdata, dx, axis=1), dy, axis=0)
    fitsio.write('r' + ff, r2data, clobber=True)


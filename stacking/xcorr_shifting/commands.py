
import glob
import fitsio
import os, sys, time
#import scipy.signal as ssig
import numpy as np
import robust_stats as rs

imlist = sorted(glob.glob('SPITZ*fits'))

imdata = [fitsio.read(ff) for ff in imlist]   
ny, nx = imdata[0].shape

## How much padding:
ypads = int(0.5 * ny)
xpads = int(0.5 * nx)
yxpads = (ypads, xpads)

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
    ##bpmask.append(bright.astype('float32')*np.sqrt(frame))
    #high_only = bright.astype('float32') * frame
    #bpmask.append(np.sqrt(high_only ))

## square-rooted amplitude, sign preserved (not effective):
#bpmask = [(np.sign(im) * np.sqrt(np.abs(im))) for im in imdata]


# FFT-assisted cross-correlation:
def qcorr(rowcol1, rowcol2):
    npix = rowcol1.size
    cft1 = np.fft.fft(rowcol1)
    cft2 = np.fft.fft(rowcol2)
    cft2.imag *= -1.0
    corr = np.fft.ifft(cft1 * cft2)
    nshift = corr.argmax()
    sys.stderr.write("--------------------------------\n")
    if (nshift > 0):
        sys.stderr.write("corr[%d]: %10.5f\n" % (nshift-1, corr[nshift-1]))
    sys.stderr.write("corr[%d]: %10.5f\n" % (nshift+0, corr[nshift+0]))
    if (nshift < npix - 1):
        sys.stderr.write("corr[%d]: %10.5f\n" % (nshift+1, corr[nshift+1]))
    sys.stderr.write("--------------------------------\n")
    if (nshift > 0.5*npix):
        nshift -= npix
    return nshift

## UNPADDED cross-correlation to find pixel shifts:
## Sum across rows to produce average column, along columns for average row:
xsmashed = [np.sum(im, axis=1) for im in bpmask]    # sum each row
ysmashed = [np.sum(im, axis=0) for im in bpmask]    # sum each col

## Cross-correlate to find pixel shifts:
xnudges = [qcorr(ysmashed[0], rr) for rr in ysmashed]
ynudges = [qcorr(xsmashed[0], cc) for cc in xsmashed]

#sys.exit(0)
# FIXME: should roll a NaN-padded image so stacking works properly ...
for ff,im,dx,dy in zip(imlist, imdata, xnudges, ynudges):
    npdata = np.pad(im, yxpads, constant_values=np.nan)
    r2data = np.roll(np.roll(npdata, dx, axis=1), dy, axis=0)
    fitsio.write('r' + ff, r2data, clobber=True)


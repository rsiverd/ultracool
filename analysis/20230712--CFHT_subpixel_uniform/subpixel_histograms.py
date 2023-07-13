# Here are the commands I used to produce a crude histogram of subpixel
# position. The outputs would benefit from labels (indicating X, Y, and
# the 0.5-pixel offset) plus some cleanup beforehand. Objects that have
# a sub-pixel position of exactly zero (before applying the offset) are
# probably artifacts.
#

data = pd.read_csv('/path/to/final_matches.csv')

# These are the pixel positions of everything:
all_xpix = data['X Pixel']
all_ypix = data['Y Pixel']

# Some of them are exactly centered in X or Y:
exact_center_x = (all_xpix % 1.0 == 0.0)     # exact pixel center in X
exact_center_y = (all_ypix % 1.0 == 0.0)     # exact pixel center in Y

# These are (not X-centered) and (not Y-centered):
useful_sources = ~exact_center_x & ~exact_center_y

# Restrict ourselves to the pixel positions of useful objects:
xpix = all_xpix[useful_sources]
ypix = all_ypix[useful_sources]

# In the FITS convention, sub-pixel position 0.0 is in the *center* of
# the pixel. The arrays below show subpixel position but shifted from
# [-0.5, 0.5] to [0, 1]:
xsubpos = (xpix - 0.5) % 1.0    # shifted so 0.5 is mid-pixel
ysubpos = (ypix - 0.5) % 1.0    # shifted so 0.5 is mid-pixel

# Alternatively, you might want to keep them at their true positions:
xsubpos = xpix % 1.0                # true position, but in [0, 1)
xsubpos[xsubpos >= 0.5] -= 1.0      # true position, now in [-0.5, 0.5)
ysubpos = xpix % 1.0                # true position, but in [0, 1)
ysubpos[ysubpos >= 0.5] -= 1.0      # true position, now in [-0.5, 0.5)

# Plot away:
hist(xsubpos, bins=100)
hist(ysubpos, bins=100)



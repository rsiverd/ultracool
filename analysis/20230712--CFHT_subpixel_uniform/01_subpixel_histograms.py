import pandas as pd
import matplotlib.pyplot as plt

csv_file = '20230629--final_matches.csv'
data = pd.read_csv(csv_file)

#xchunks = data.groupby('x_cell')
#
#for xc,xsubset in xchunks:
#    xychunks = xsubset.groupby('y_cell')
#    for yc,celldata in xychunks:
#        pass

all_xpix = data['X Pixel']
all_ypix = data['Y Pixel']

exact_center_x = (all_xpix % 1.0 == 0.0)     # exact pixel center in X
exact_center_y = (all_ypix % 1.0 == 0.0)     # exact pixel center in Y

# These are centered in one or the other:
exact_centers  = exact_center_x | exact_center_y 

# Restrict to the non-center subset:
xpix = all_xpix[~exact_centers]
ypix = all_ypix[~exact_centers]

xsubpos = (xpix - 0.5) % 1.0    # shifted so 0.5 is mid-pixel
ysubpos = (ypix - 0.5) % 1.0    # shifted so 0.5 is mid-pixel

# Histogram config:
n_bins = 100

# Output files:
sx_plot_name = 'x_subpixel_dist.%03d.png' % n_bins
sy_plot_name = 'y_subpixel_dist.%03d.png' % n_bins

# Make two figures:
fig_dims = (11, 9)
x_fig = plt.figure(1, figsize=fig_dims)
y_fig = plt.figure(2, figsize=fig_dims)
x_fig.clf()
y_fig.clf()

# One axis per figure:
x_ax = x_fig.add_subplot(111)
y_ax = y_fig.add_subplot(111)

# Histogram sub-pixel positions:
x_ax.hist(xsubpos, bins=n_bins)
x_ax.set_xlabel('X sub-pixel position - 0.5')
y_ax.hist(ysubpos, bins=n_bins)
y_ax.set_xlabel('Y sub-pixel position - 0.5')

# Remove whitespace:
x_fig.tight_layout()
y_fig.tight_layout()

# Save figures:
x_fig.savefig(sx_plot_name, bbox_inches='tight')
y_fig.savefig(sy_plot_name, bbox_inches='tight')



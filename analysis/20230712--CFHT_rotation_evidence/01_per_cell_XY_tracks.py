#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

csv_file = '20230629--final_matches.csv'
data = pd.read_csv(csv_file)

# To illustrate time, let's convert the RunID into a decimal year. This
# can then be used to colorize data points by time later.
def runid2year(runid):
    year = float(runid[0:2])
    yadd = 0.5 if runid[2] == 'B' else 0.0
    #sys.stderr.write("sem: %s\n" % semc)
    return year + yadd

# Add a column with the approximate year:
data['year'] = [runid2year(x) for x in data['RunID']]

# Per-cell data point picker:
def get_cell_subset(x_cell, y_cell):
    keep = (data['x_cell'] == x_cell) & (data['y_cell'] == y_cell)
    return data[keep]

# Cells of interest:
xc, yc =  1, 14      # near top left
xc, yc =  1,  1      # near bottom left
xc, yc = 14, 14      # near top right
xc, yc = 14,  1      # near bottom right

# Pick a cell one (all the points):
this_cell = get_cell_subset(xc, yc)

# Optionally keep just a fraction for clarity:
this_cell = get_cell_subset(xc, yc)[::5]    # every 5th element in the cell

# Some scatter plot args:
skw = {'lw':0, 's':2}

# Make a figure with equal-aspect axes:
fig = plt.figure(figsize=(8,7))
fig.clf()
ax1 = fig.add_subplot(111, aspect='equal')
ax1.patch.set_facecolor((0.8, 0.8, 0.8))    # grayed plot background

# A plot of the raw locations:
skw = {'lw':0, 's':2}
#ax1.scatter(this_cell['X Pixel'], this_cell['Y Pixel'], **skw)

# Optionally, colorize by time:
skw = {'lw':0, 's':4}
#skw = {'lw':0, 's':4, 'cmap':'jet'}     # jet uses more colors. better?
skw = {'lw':0, 's':4, 'cmap':'plasma'}   
ax1.scatter(this_cell['X Pixel'], this_cell['Y Pixel'],
        c=this_cell['year'], **skw)



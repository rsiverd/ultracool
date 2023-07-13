import pandas as pd

data = pd.read_csv('/home/acolclas/final_matches.csv')

xchunks = data.groupby('x_cell')

for xc,xsubset in xchunks:
    xychunks = xsubset.groupby('y_cell')
    for yc,celldata in xychunks:
        pass

xpix = data['X Pixel']
ypix = data['Y Pixel']

xsubpos = (xpix - 0.5) % 1.0    # shifted so 0.5 is mid-pixel
ysubpos = (ypix - 0.5) % 1.0    # shifted so 0.5 is mid-pixel


hist(xsubpos, bins=100)

hist(ysubpos, bins=100)

hist(xsubpos, bins=1000)

hist(ysubpos, bins=1000)


xc, yc =  1, 14      # near top left
xc, yc =  1,  1      # near bottom left
xc, yc = 14, 14      # near top right
xc, yc = 14,  1      # near bottom right
skw = {'lw':0, 's':2}

which = (data.x_cell == xc) & (data.y_cell == yc)
subset = data[which][::5]

clf()
scatter(subset['X Pixel'], subset['Y Pixel'], **skw)



import matplotlib.pyplot as plt
import numpy as np

save_plot = "sigma_scatter.png"

blurb = r"$K_n S = \frac{\sigma}{\sqrt{2(n - 1)}}$"

## Shape of curve:
npts = np.arange(150.0) + 5.0
factor = 1.0 / np.sqrt(2.0 * (npts - 1.0))


## Make plot:
fig_dims = (8, 7)
fig = plt.figure(1, figsize=fig_dims)
fig.clf()
ax1 = fig.add_subplot(111)
ax1.grid(True)

#ax1.set_title(blurb)
ax1.plot(npts, factor, label=blurb)

ax1.set_xlabel("Data points")
ax1.set_ylabel("Fractional Scatter in stddev")
ax1.legend(loc="upper right", fontsize='xx-large')

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig.savefig(save_plot)


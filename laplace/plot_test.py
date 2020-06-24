import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np


# make these smaller to increase the resolution
dx1, dx2 = 0.05, 0.05

# generate 2 2d grids for the x & y bounds
x1, x2 = np.mgrid[slice(0, 1 + dx1, dx1), slice(0, 1 + dx2, dx2)]

# z = np.sin(x2) ** 10 + np.cos(10 + x1 * x2) * np.cos(x2)
z = (1 / (np.exp(np.pi) - np.exp(-np.pi))) * np.sin(np.pi*x1) * (np.exp(np.pi * x2) - np.exp(-np.pi * x2))

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
levels = MaxNLocator(nbins=20).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('rainbow')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(ncols=2)

im = ax0.pcolormesh(x1, x2, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
fig.set_figheight(2.5)
ax0.set_aspect('equal')
ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(x1[:-1, :-1] + dx1 / 2.,
                  x2[:-1, :-1] + dx2 / 2., z, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_aspect('equal')
ax1.set_title('contourf with levels')
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
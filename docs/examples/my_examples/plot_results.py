
'''Example of plotting results'''

import pdb
# Library imports
import xarray as xr
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Load the ParticleFile
ds = xr.open_zarr('C:/Users/martino.rial/Escritorio/resultados_parcels/my_example.zarr')

# Arousa limits
lat_min = 42.38
lat_max = 42.71
lon_min = -9.1
lon_max = -8.78

# Settings for the concentration map
bins = (np.linspace(lon_min, lon_max, 100), np.linspace(lat_min, lat_max, 70))
i = -1  # Use final timestep

# Create the figure object
plt.figure(figsize=(14, 5), dpi=200)
gs = gridspec.GridSpec(1, 3, width_ratios=[30, 30, 1], wspace=0.1)
cb_axes_position = [1.025, 0.0, 0.03, 0.95]

# Plot the trajectories
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.plot(ds['lon'].T, ds['lat'].T, transform=ccrs.PlateCarree(), zorder=0, linewidth=0.5)
# ax.add_feature(cfeature.LAND(scale=''), zorder=0, color='grey')
coast = cfeature.GSHHSFeature(scale='full')
ax.add_feature(coast, facecolor="brown", alpha=0.5)


# Define plotted region
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())    # latitude and longitude limits
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color="None")
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator([6, 9, 12, 15, 18, 21, 24])

ax.text(-0.06, 1.015, 'a)', transform=ax.transAxes, size=16)
ax.text(-0.13, 0.45, 'Latitude', transform=ax.transAxes, size=10, rotation=90)
ax.text(0.4, -0.13, 'Longitude', transform=ax.transAxes, size=10)


# Plot the Concentration map
ax = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
cb = ax.hist2d(ds['lon'][:, :i].values.flatten(), ds['lat'][:, :i].values.flatten(), bins=bins, norm=LogNorm(vmin=1, vmax=1000), cmap=plt.cm.magma, transform=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, zorder=0, color='grey')
coast = cfeature.GSHHSFeature(scale='full')
ax.add_feature(coast, facecolor="brown", alpha=0.5)


cbar_ax = ax.inset_axes(cb_axes_position)
cbar = plt.colorbar(cb[3], cax=cbar_ax, extend='both')
cbar.set_label('Number of plastic particles', fontsize=10)
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color="None")
gl.top_labels = False
gl.right_labels = False
gl.left_labels = False
gl.xlocator = mticker.FixedLocator([6, 9, 12, 15, 18, 21, 24])

ax.text(-0.06, 1.015, 'b)', transform=ax.transAxes, size=16)
ax.text(0.4, -0.13, 'Longitude', transform=ax.transAxes, size=10)

plt.show()

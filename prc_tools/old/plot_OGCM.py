# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:50:31 2025

@author: ust21
"""
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import interp2d, griddata

OGCM_u=Dataset('D:/shjo/projects/myROMS/test_data/raw/HYCOM_GLBy0.08_u_2019_121.nc')
OGCM_v=Dataset('D:/shjo/projects/myROMS/test_data/raw/HYCOM_GLBy0.08_v_2019_121.nc')

lonO,latO=OGCM_u['lon'][:],OGCM_u['lat'][:]

uO=OGCM_u['u'][0,0,:,:].data
vO=OGCM_v['v'][0,0,:,:].data

vO[vO<-100]=np.nan
uO[uO<-100]=np.nan


n=4
lat_re,lon_re=latO[::n,::n],lonO[::n,::n]


U_diff_=griddata((lonO.flatten(),latO.flatten()),uO.flatten(),
                (lon_re.flatten(),lat_re.flatten()),
            method='linear',fill_value=np.nan)
V_diff_=griddata((lonO.flatten(),latO.flatten()),vO.flatten(),
                (lon_re.flatten(),lat_re.flatten()),
            method='linear',fill_value=np.nan)
U_re = U_diff_.reshape(lon_re.shape)
V_re = V_diff_.reshape(lon_re.shape)


PC = ccrs.PlateCarree(central_longitude=0.0,globe=None)
Label_size=14
PC180 = ccrs.PlateCarree(central_longitude=200.0,globe=None)
fig, ax = plt.subplots(1, 1, figsize=(6,8),
                    subplot_kw={'projection': PC180},dpi=200,constrained_layout=True)
gl = ax.gridlines(crs=PC, draw_labels=True,y_inline=False,x_inline=False,
                    linewidth=.6, color='k', alpha=0.45, linestyle='-.',\
                        xlocs=range(-180, 180, 2),ylocs=range(30, 90, 2),zorder=200)
gl.xlabels_top,gl.ylabels_right = False,False
gl.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels
n=32
gl.xlabel_style = gl.ylabel_style = {"size" : 14}
ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=1,zorder=110)
ax.add_feature(cf.LAND,color=[.75,.75,.75],zorder=100)
ax.set_title('Surface uv',loc='right',fontdict={'fontsize':Label_size+3,'fontweight':'regular'})
q1 = ax.quiver(lon_re,lat_re,U_re,V_re,
    scale=4.,headwidth=8.,headaxislength=10,headlength=13,color='k',
    minlength=1,edgecolor='k',minshaft=1.3,alpha=1.,transform=ccrs.PlateCarree(),zorder=100,
    pivot='mid',angles='xy')
#qk = ax.quiverkey(q1, 0.23, .85, 1., r'$1 m^2/s $', labelpos='E',color='k',
#                coordinates='figure')
#M=plt.contourf(lonG,latG,v_rho,cmap=GSMCMAP,levels=GSM_Levels,transform=PC,zorder=1)
#plt.contour(lon_new_m,lat_new_m,DATA_rgd_mean,colors='k',zorder=100,transform=PC,linestyles='dashdot')
# M=plt.pcolor(lon_m,lat_m,mydataD,cmap=Mycmap,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
ax.set_extent([127.3, 135, 34.5, 41.5], crs=PC)
#ax.tick_params(axis='both', which='major', labelsize=Label_size)
#divider = make_axes_locatable(ax)
#ax_cb = divider.new_horizontal(size="5%", pad=.1, axes_class=plt.Axes)
#fig.add_axes(ax_cb)
# cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=CTicks)
#cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=Cticks)

if 0:
    plt.savefig(snm+'_eof',bbox_inches='tight')
plt.show()

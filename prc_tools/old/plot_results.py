# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:55:18 2025

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
A=xr.open_dataset('D:/shjo/projects/myROMS/outputs/test_ini.nc')
My_Grd='D:/shjo/projects/myROMS/test_data/Domain.nc' # Grd name
ncG=Dataset(My_Grd)
lonG,latG=ncG['lon_rho'][:],ncG['lat_rho'][:]
angle,topo,mask=ncG['angle'][:],ncG['h'][:],ncG['mask_rho'][:]

d=A.temp[0,-1].values



# =============================================================================
# surface temp
# =============================================================================
d=A.temp[0,-1].values
GSMCMAP=plt.get_cmap('RdYlBu_r')
GSM_Levels=np.arange(6,18,0.5)
Cticks=np.arange(6,18,2)
Label_size=14
PC = ccrs.PlateCarree(central_longitude=0.0,globe=None)

PC180 = ccrs.PlateCarree(central_longitude=200.0,globe=None)
fig, ax = plt.subplots(1, 1, figsize=(6,8),
                    subplot_kw={'projection': PC180},dpi=200,constrained_layout=True)
gl = ax.gridlines(crs=PC, draw_labels=True,y_inline=False,x_inline=False,
                    linewidth=.6, color='k', alpha=0.45, linestyle='-.',\
                        xlocs=range(-180, 180, 2),ylocs=range(30, 90, 2),zorder=200)
gl.xlabels_top,gl.ylabels_right = False,False
gl.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

gl.xlabel_style = gl.ylabel_style = {"size" : 14}
ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=1,zorder=110)
ax.add_feature(cf.LAND,color=[.75,.75,.75],zorder=100)
ax.set_title('Surface temp',loc='right',fontdict={'fontsize':Label_size+3,'fontweight':'regular'})
M=plt.contourf(lonG,latG,d,cmap=GSMCMAP,levels=GSM_Levels,transform=PC,zorder=1)
#plt.contour(lon_new_m,lat_new_m,DATA_rgd_mean,colors='k',zorder=100,transform=PC,linestyles='dashdot')
# M=plt.pcolor(lon_m,lat_m,mydataD,cmap=Mycmap,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
ax.set_extent([127.3, 135, 34.5, 41.5], crs=PC)
ax.tick_params(axis='both', which='major', labelsize=Label_size)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)
# cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=CTicks)
cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=Cticks)

if 0:
    plt.savefig(snm+'_eof',bbox_inches='tight')
plt.show()

# =============================================================================
# surface salt
# =============================================================================
d=A.salt[0,-1].values
GSMCMAP=plt.get_cmap('Blues')
GSM_Levels=np.arange(33.5,34.8,.1)
Cticks=np.arange(33.5,34.9,0.2)
Label_size=14
PC = ccrs.PlateCarree(central_longitude=0.0,globe=None)

PC180 = ccrs.PlateCarree(central_longitude=200.0,globe=None)
fig, ax = plt.subplots(1, 1, figsize=(6,8),
                    subplot_kw={'projection': PC180},dpi=200,constrained_layout=True)
gl = ax.gridlines(crs=PC, draw_labels=True,y_inline=False,x_inline=False,
                    linewidth=.6, color='k', alpha=0.45, linestyle='-.',\
                        xlocs=range(-180, 180, 2),ylocs=range(30, 90, 2),zorder=200)
gl.xlabels_top,gl.ylabels_right = False,False
gl.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

gl.xlabel_style = gl.ylabel_style = {"size" : 14}
ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=1,zorder=110)
ax.add_feature(cf.LAND,color=[.75,.75,.75],zorder=100)
ax.set_title('Surface salinity',loc='right',fontdict={'fontsize':Label_size+3,'fontweight':'regular'})
M=plt.contourf(lonG,latG,d,cmap=GSMCMAP,levels=GSM_Levels,transform=PC,zorder=1)
#plt.contour(lon_new_m,lat_new_m,DATA_rgd_mean,colors='k',zorder=100,transform=PC,linestyles='dashdot')
# M=plt.pcolor(lon_m,lat_m,mydataD,cmap=Mycmap,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
ax.set_extent([127.3, 135, 34.5, 41.5], crs=PC)
ax.tick_params(axis='both', which='major', labelsize=Label_size)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)
# cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=CTicks)
cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=Cticks)

if 0:
    plt.savefig(snm+'_eof',bbox_inches='tight')
plt.show()


# =============================================================================
# surface u
# =============================================================================
su=A.u[0,-1].values
sv=A.v[0,-1].values

su.shape
sv.shape

Mp, L = su.shape
Lp = L + 1
Lm = L - 1
u_rho = np.zeros((Mp, Lp))
u_rho[:, 1:L] = 0.5 * (su[:, :Lm] + su[:, 1:L])
u_rho[:, 0] = u_rho[:, 1]
u_rho[:, Lp - 1] = u_rho[:, L - 1]

M, Lp = sv.shape
Mp = M + 1
Mm = M - 1

v_rho = np.zeros((Mp, Lp))
v_rho[1:M, :] = 0.5 * (sv[:Mm, :] + sv[1:M, :])
v_rho[0, :] = v_rho[1, :]
v_rho[Mp - 1, :] = v_rho[M - 1, :]


u_geo = u_rho * np.cos(angle) - v_rho * np.sin(angle)
v_geo = u_rho * np.sin(angle) + v_rho * np.cos(angle)

GSMCMAP=plt.get_cmap('seismic')
GSM_Levels=np.arange(-0.8,0.8,.05)
Cticks=np.arange(-0.8,0.8,.1)
Label_size=14
PC = ccrs.PlateCarree(central_longitude=0.0,globe=None)

PC180 = ccrs.PlateCarree(central_longitude=200.0,globe=None)
fig, ax = plt.subplots(1, 1, figsize=(6,8),
                    subplot_kw={'projection': PC180},dpi=200,constrained_layout=True)
gl = ax.gridlines(crs=PC, draw_labels=True,y_inline=False,x_inline=False,
                    linewidth=.6, color='k', alpha=0.45, linestyle='-.',\
                        xlocs=range(-180, 180, 2),ylocs=range(30, 90, 2),zorder=200)
gl.xlabels_top,gl.ylabels_right = False,False
gl.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

gl.xlabel_style = gl.ylabel_style = {"size" : 14}
ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=1,zorder=110)
ax.add_feature(cf.LAND,color=[.75,.75,.75],zorder=100)
ax.set_title('Surface u_rho',loc='right',fontdict={'fontsize':Label_size+3,'fontweight':'regular'})
M=plt.contourf(lonG,latG,u_rho,cmap=GSMCMAP,levels=GSM_Levels,transform=PC,zorder=1)
#plt.contour(lon_new_m,lat_new_m,DATA_rgd_mean,colors='k',zorder=100,transform=PC,linestyles='dashdot')
# M=plt.pcolor(lon_m,lat_m,mydataD,cmap=Mycmap,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
ax.set_extent([127.3, 135, 34.5, 41.5], crs=PC)
ax.tick_params(axis='both', which='major', labelsize=Label_size)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)
# cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=CTicks)
cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=Cticks)

if 0:
    plt.savefig(snm+'_eof',bbox_inches='tight')
plt.show()


PC = ccrs.PlateCarree(central_longitude=0.0,globe=None)

PC180 = ccrs.PlateCarree(central_longitude=200.0,globe=None)
fig, ax = plt.subplots(1, 1, figsize=(6,8),
                    subplot_kw={'projection': PC180},dpi=200,constrained_layout=True)
gl = ax.gridlines(crs=PC, draw_labels=True,y_inline=False,x_inline=False,
                    linewidth=.6, color='k', alpha=0.45, linestyle='-.',\
                        xlocs=range(-180, 180, 2),ylocs=range(30, 90, 2),zorder=200)
gl.xlabels_top,gl.ylabels_right = False,False
gl.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

gl.xlabel_style = gl.ylabel_style = {"size" : 14}
ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=1,zorder=110)
ax.add_feature(cf.LAND,color=[.75,.75,.75],zorder=100)
ax.set_title('Surface v_rho',loc='right',fontdict={'fontsize':Label_size+3,'fontweight':'regular'})
M=plt.contourf(lonG,latG,v_rho,cmap=GSMCMAP,levels=GSM_Levels,transform=PC,zorder=1)
#plt.contour(lon_new_m,lat_new_m,DATA_rgd_mean,colors='k',zorder=100,transform=PC,linestyles='dashdot')
# M=plt.pcolor(lon_m,lat_m,mydataD,cmap=Mycmap,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
ax.set_extent([127.3, 135, 34.5, 41.5], crs=PC)
ax.tick_params(axis='both', which='major', labelsize=Label_size)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)
# cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=CTicks)
cb=plt.colorbar(M,extend='both',pad=0.01,cax=ax_cb,ticks=Cticks)

if 0:
    plt.savefig(snm+'_eof',bbox_inches='tight')
plt.show()

### Regrid ==================================================
n=2;Label_size=14

PC = ccrs.PlateCarree(central_longitude=0.0,globe=None)

PC180 = ccrs.PlateCarree(central_longitude=200.0,globe=None)
fig, ax = plt.subplots(1, 1, figsize=(6,8),
                    subplot_kw={'projection': PC180},dpi=200,constrained_layout=True)
gl = ax.gridlines(crs=PC, draw_labels=True,y_inline=False,x_inline=False,
                    linewidth=.6, color='k', alpha=0.45, linestyle='-.',\
                        xlocs=range(-180, 180, 2),ylocs=range(30, 90, 2),zorder=200)
gl.xlabels_top,gl.ylabels_right = False,False
gl.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels
n=1
gl.xlabel_style = gl.ylabel_style = {"size" : 14}
ax.add_feature(cf.COASTLINE.with_scale("110m"), lw=1,zorder=110)
ax.add_feature(cf.LAND,color=[.75,.75,.75],zorder=100)
ax.set_title('Surface uv',loc='right',fontdict={'fontsize':Label_size+3,'fontweight':'regular'})
q1 = ax.quiver(lonG[::n,::n],latG[::n,::n],u_geo[::n,::n],v_geo[::n,::n],
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










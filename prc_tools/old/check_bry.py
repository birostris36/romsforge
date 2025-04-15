import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

#A=xr.open_dataset('D:/shjo/projects/myROMS/outputs/test_ini.nc')
My_Grd='D:/shjo/projects/myROMS/test_data/Domain.nc' # Grd name
ncG=Dataset(My_Grd)
lonG,latG=ncG['lon_rho'][:],ncG['lat_rho'][:]
angle,topo,mask=ncG['angle'][:],ncG['h'][:],ncG['mask_rho'][:]
s_rho=ncG['s_rho'][:]

bry=Dataset('D:/shjo/projects/myROMS/outputs/Bry_test_end.nc')
ini=Dataset('D:/shjo/projects/myROMS/outputs/test_ini_end.nc')

t=1
temp=ini['v'][0,:,:]
north_temp=bry['v_north'][0,:,:]
lon_north=lonG[:,0]

lat_n_m,s_m=np.meshgrid(lon_north,s_rho)


plt.figure(1)
plt.pcolor(lat_n_m,s_m,north_temp,cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([0,10])
plt.colorbar()

plt.figure(2)
plt.pcolor(lat_n_m,s_m,temp[:,:,0],cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([0,10])
plt.colorbar()


diff_temp_north=temp[:,:,0]-north_temp

plt.figure(3)
plt.pcolor(diff_temp_north,cmap=plt.get_cmap('seismic'))
plt.clim([-1.,1.])
plt.colorbar()

# =============================================================================
# south !!!!!!!!!!!
# =============================================================================
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
s_rho=ncG['s_rho'][:]

bry=Dataset('D:/shjo/projects/myROMS/outputs/Bry_test_end.nc')
ini=Dataset('D:/shjo/projects/myROMS/outputs/test_ini_end.nc')
temp=ini['v'][0,:,:]
south_temp=bry['v_south'][0,:,:]
lat_north=latG[:,0]

lat_n_m,s_m=np.meshgrid(lat_north,s_rho)


plt.figure(1)
plt.pcolor(lat_n_m,s_m,south_temp,cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([14,17])
plt.colorbar()

plt.figure(2)
plt.pcolor(lat_n_m,s_m,temp[:,:,-1],cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([14,17])
plt.colorbar()


diff_temp_south=temp[:,:,-1]-south_temp
plt.figure(3)
plt.pcolor(diff_temp_south,cmap=plt.get_cmap('seismic'))
plt.clim([-.1,.1])
plt.colorbar()


# =============================================================================
# East 
# =============================================================================
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
s_rho=ncG['s_rho'][:]

bry=Dataset('D:/shjo/projects/myROMS/outputs/Bry_test.nc')
ini=Dataset('D:/shjo/projects/myROMS/outputs/test_ini.nc')


temp=ini['u'][0,:,:]
east_temp=bry['u_east'][0,:,:]
lat_east=lonG[0,:]

lat_n_m,s_m=np.meshgrid(lat_east,s_rho)


plt.figure(1)
plt.pcolor(lat_n_m,s_m,east_temp,cmap=plt.get_cmap('RdYlBu_r'))
#plt.clim([0,17])
plt.colorbar()

plt.figure(2)
plt.pcolor(lat_n_m,s_m,temp[:,-1,:],cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([0,17])
plt.colorbar()


diff_temp_east=temp[:,-1,:]-east_temp
plt.figure(3)
plt.pcolor(diff_temp_east,cmap=plt.get_cmap('seismic'))
plt.clim([-.1,.1])
plt.colorbar()


# =============================================================================
# west !!!!!!!!!!!
# =============================================================================
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

#A=xr.open_dataset('D:/shjo/projects/myROMS/outputs/test_ini.nc')
My_Grd='D:/shjo/projects/myROMS/test_data/Domain.nc' # Grd name
ncG=Dataset(My_Grd)
lonG,latG=ncG['lon_rho'][:],ncG['lat_rho'][:]
angle,topo,mask=ncG['angle'][:],ncG['h'][:],ncG['mask_rho'][:]
s_rho=ncG['s_rho'][:]

bry=Dataset('D:/shjo/projects/myROMS/outputs/Bry_test.nc')
ini=Dataset('D:/shjo/projects/myROMS/outputs/Ini_test.nc')

temp=ini['temp'][0,:,:]
west_temp=bry['temp_west'][0,:,:]
lat_east=latG[0,:]

lat_n_m,s_m=np.meshgrid(lat_east,s_rho)


plt.figure(1)
plt.pcolor(lat_n_m,s_m,west_temp,cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([0,17])
plt.colorbar()

plt.figure(2)
plt.pcolor(lat_n_m,s_m,temp[:,1,:],cmap=plt.get_cmap('RdYlBu_r'))
plt.clim([0,17])
plt.colorbar()

diff_west_temp=temp[:,1,:]-west_temp

plt.figure(3)
plt.pcolor(diff_west_temp,cmap=plt.get_cmap('seismic'))
plt.clim([-1,1])
plt.colorbar()



OGCM_pth='D:/shjo/projects/myROMS/test_data/raw/HYCOM_GLBy0.08_temp_2019_121.nc'

OGCM=Dataset(OGCM_pth)['temp'][0,0,:,:]
lonO,latO=Dataset(OGCM_pth)['lon'][:],Dataset(OGCM_pth)['lat'][:]

print(np.min(latG))
print(np.min(latO))



print(np.min(lonG))
print(np.min(lonO))

















# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:50:22 2025

@author: ust21
"""



PKG_path = 'D:/shjo/projects/myROMS/prc_tools/' # Location of JNUROMS directory
import sys 
sys.path.append(PKG_path)
import utils.ROMS_utils01 as ru
import utils.ROMS_utils02 as ru2
from utils.ncCreate import create_bry_VVV
import xarray as xr
import numpy as np
from netCDF4 import Dataset,MFDataset,date2num,num2date
import netCDF4 as nc4
import os
from scipy.interpolate import griddata
from copy import deepcopy
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
My_Bry='D:/shjo/projects/myROMS/outputs//Bry_test.nc'
My_Grd='D:/shjo/projects/myROMS/test_data/Domain.nc' # Grd name

ncdir='D:/shjo/projects/myROMS/test_data/raw/'
sshNC=ncdir+'HYCOM_GLBy0.08_ssh_2019'
tempNC=ncdir+'HYCOM_GLBy0.08_temp_2019'
saltNC=ncdir+'HYCOM_GLBy0.08_salt_2019'
uNC=ncdir+'HYCOM_GLBy0.08_u_2019'
vNC=ncdir+'HYCOM_GLBy0.08_v_2019'

Bry_title='test'
# My Variables
MyVar={'Layer_N':50,'Vtransform':2,'Vstretching':4,\
       'Theta_s':7,'Theta_b':.1,'Tcline':450,'hmin':50}
# OGCM Variables name
# OGCM Variables name
OGCMVar={'lon_rho':'lon','lat_rho':'lat','depth':'z','time':'ocean_time',\
         'lon_u':'lon','lat_u':'lon','lon_v':'lon','lat_v':'lat',
         'temp':'temp','salt':'salt','u':'u','v':'v','zeta':'ssh'}
conserv=1

OGCMS=[ncdir+i for i in os.listdir(ncdir) if (i.endswith('.nc') & i.startswith('HYCOM_GLBy0.08_temp') ) ]

# Get My Grid info
ncG=Dataset(My_Grd)
lonG,latG=ncG['lon_rho'][:],ncG['lat_rho'][:]
angle,topo,mask=ncG['angle'][:],ncG['h'][:],ncG['mask_rho'][:]
ncG['mask_u'].shape
ncG['mask_v'].shape

ncG.close()




# Get OGCM Grid info
ncO=Dataset(OGCMS[-1])
lonO,latO=ncO[OGCMVar['lon_rho']][:],ncO[OGCMVar['lat_rho']][:]
depthO=ncO[OGCMVar['depth']][:]
'''
t_rng=['2019-12-01 01:00','2019-12-02 24:00']
My_time_ref='days since 1990-1-1 00:00:00'
OGCM_TIMES=xr.open_mfdataset(OGCMS[0],decode_times=False)[OGCMVar['time']]
TIME_UNIT=OGCM_TIMES.units
OGCM_times=num2date(OGCM_TIMES[:],TIME_UNIT)
Tst=dt.datetime(int(t_rng[0].split('-')[0]), int(t_rng[0].split('-')[1]),int(t_rng[0].split('-')[2][:2]),int(t_rng[0].split('-')[-1].split(' ')[-1][:2]))
Ted=dt.datetime(int(t_rng[1].split('-')[0]), int(t_rng[1].split('-')[1]),int(t_rng[1].split('-')[2][:2]),int(t_rng[1].split('-')[-1].split(' ')[-1][:2]))
TIMES_co=np.where( (OGCM_times>=Tst)&(OGCM_times<=Ted) )[0]
tmp_y,tmp_m,tmp_d,tmp_H=int(t_rng[0].split('-')[0]),int(t_rng[0].split('-')[1][:2]),int(t_rng[0].split('-')[2].split(' ')[0][:2]),int(t_rng[0].split('-')[2].split(' ')[1][0:2])
tmp_dif=date2num(dt.datetime(tmp_y,tmp_m,tmp_d,tmp_H),TIME_UNIT)-date2num(dt.datetime(tmp_y,tmp_m,tmp_d,tmp_H),My_time_ref)
Bry_time_time=num2date(OGCM_TIMES[TIMES_co]-tmp_dif,My_time_ref)
Bry_time_num=OGCM_TIMES[TIMES_co]-tmp_dif
'''
TIMES_co=[0,1]
Bry_time_num=TIMES_co

thO=ncO[OGCMVar['depth']].shape[0]
ncO.close()

atG,onG=lonG.shape
#cosa_,sina_=np.cos(angle)[-2:],np.sin(angle)[-2:] #NORTHERN BRY
#cosa=np.tile( np.tile(cosa_,(thO,1,1)), (len(Bry_time_num),1,1,1) )
#sina=np.tile( np.tile(sina_,(thO,1,1)), (len(Bry_time_num),1,1,1) )

NSEW=[True,True,True,True]
#create_bry_VVV(My_Bry,mask,topo,MyVar,NSEW,Bry_time_num,My_time_ref,Bry_title,ncFormat='NETCDF4')


#-- Get OGCM lon lat coordinates for slicing ----------------------------------
lonO_co01=np.where( (lonO[0,:]>=np.min(lonG)) & (lonO[0,:]<=np.max(lonG)) )[0]
latO_co01=np.where( (latO[:,0]>=np.min(latG)) & (latO[:,0]<=np.max(latG)) )[0]

latO_re=latO[latO_co01,0]
lonO_re=lonO[0,lonO_co01]

lonGmax=np.max(np.abs(np.diff(lonG[0,:])))
latGmax=np.max(np.abs(np.diff(latG[:,0])))

lonOmax=np.max(np.abs(np.diff(lonO_re[:])))
latOmax=np.max(np.abs(np.diff(latO_re[:])))

lonEval=np.max([ np.max(lonO_re) ,np.max(lonG)+lonOmax+0.5])
lonSval=np.min([ np.min(lonO_re), np.min(lonG)-lonOmax-0.5])

latEval=np.max([ np.max(latO_re),np.max(latG)+latOmax+0.5])
latSval=np.min([ np.min(latO_re),np.min(latG)-latOmax-0.5])

lonO_co=np.where((lonO[0,:]>=lonSval)&(lonO[0,:]<=lonEval))[0]
latO_co=np.where((latO[:,0]>=latSval)&(latO[:,0]<=latEval))[0]

latO_s=latO[latO_co,0]
lonO_s=lonO[0,lonO_co]

# lonO_s_m,latO_s_m=np.meshgrid(lonO_s,latO_s)
# lonO_s_m,latO_s_m=np.meshgrid(lonO_s,latO_s)


# =============================================================================
# Western Boundary
# =============================================================================

if NSEW[0]:
    pass

# West (gridbuilder version)
bry_lat_co=latO_co
bry_lon_co=np.where( (lonO[0,:]>=np.min(lonG[:2,:])-lonOmax) & (lonO[0,:]<=np.max(lonG[:2,:])+lonOmax) )[0]

lonO_s_m,latO_s_m=np.meshgrid(lonO[0,bry_lon_co],latO[bry_lat_co,0])


OGCM_Data={}#,OGCM_Mask={}

for i in ['temp','u','v','salt','zeta','ubar','vbar']:
#for i in ['temp']:
    print('!!! Data processing : '+i+' !!!')
    
    if (i=='zeta') or (i=='ubar') or (i=='vbar'):
        data=np.zeros([len(Bry_time_num),2,lonG.shape[-1]])
        if i=='zeta':
            # DATA=np.squeeze(MFDataset(sshNC+'*.nc')[OGCMVar[i]][TIMES_co,bry_lat_co,bry_lon_co])
            #DATA=xr.open_mfdataset(sshNC+'*.nc')[OGCMVar[i]][TIMES_co,bry_lat_co,bry_lon_co].squeeze().values
            DATA=MFDataset(sshNC+'*.nc')[OGCMVar[i]][TIMES_co,bry_lat_co,bry_lon_co]  

            mask = np.isnan(DATA)
            DATA = np.ma.array(DATA, mask=mask)
            tmp_mask_=np.invert(DATA.mask)
            
        elif i=='ubar':
            # tmp_u=np.squeeze(MFDataset(uNC+'*.nc')[OGCMVar['u']][TIMES_co,:,bry_lat_co,bry_lon_co])
            #tmp_u=xr.open_mfdataset(uNC+'*.nc',parallel=True)[OGCMVar['u']][TIMES_co,:,bry_lat_co,bry_lon_co].squeeze().values
            tmp_u=MFDataset(uNC+'*.nc')[OGCMVar['u']][TIMES_co,:,bry_lat_co,bry_lon_co]  

            mask = np.isnan(tmp_u)
            tmp_u = np.ma.array(tmp_u, mask=mask)
            tmp_mask_=np.invert(tmp_u.mask)[:,:,:,:]
            
            tmp_u[tmp_u.mask]=0
            
            du=np.zeros([tmp_u.shape[0],tmp_u.shape[2],tmp_u.shape[3]])
            zu=np.zeros_like(du)
            dz=np.gradient(-depthO)
            for n in range(len(depthO)):
                du=du+dz[n]*tmp_u[:,n,:,:].data
                zu=zu+dz[n]*tmp_mask_[:,n,:,:]
            DATA=du/zu
            # DATA[DATA==0]=np.nan
            tmp_mask_=tmp_mask_[:,0,:,:]
            
        elif i=='vbar':
            # tmp_v=np.squeeze(MFDataset(vNC+'*.nc')[OGCMVar['v']][TIMES_co,:,bry_lat_co,bry_lon_co])
            tmp_v=MFDataset(vNC+'*.nc')[OGCMVar['v']][TIMES_co,:,bry_lat_co,bry_lon_co]  

            mask = np.isnan(tmp_v)
            tmp_v = np.ma.array(tmp_v, mask=mask)
            
            tmp_mask_=np.invert(tmp_v.mask)[:,:,:,:]
            
            tmp_v[tmp_v.mask]=0
            
            dv=np.zeros([tmp_v.shape[0],tmp_v.shape[2],tmp_v.shape[3]])
            zv=np.zeros_like(dv)
            dz=np.gradient(-depthO)
            for n in range(len(depthO)):
                dv=dv+dz[n]*tmp_v[:,n,:,:].data
                zv=zv+dz[n]*tmp_mask_[:,n,:,:]
            DATA=dv/zv
            # DATA[DATA==0]=np.nan
            tmp_mask_=tmp_mask_[:,0,:,:]
            
        for t in tqdm(range(len(Bry_time_num))):
            tmp_mask=tmp_mask_[t]
            data_=griddata((lonO_s_m[tmp_mask].flatten(),latO_s_m[tmp_mask].flatten()),\
                          DATA[t][tmp_mask].flatten(),(lonO_s_m.flatten(),latO_s_m.flatten()),'nearest')
            data_=data_.reshape(latO_s_m.shape)
    
            # Interp 4 Grid
            data_re_=griddata( (lonO_s_m.flatten(),latO_s_m.flatten()), data_.flatten(), (lonG[:2,:].flatten(),latG[:2,:].flatten()) ,'cubic' )
            data[t]=data_re_.reshape(lonG[:2,:].shape)
            
            tmp_var=data[t]

            if np.sum(np.isnan(data[t]))!=0:
                tmp_var[np.isnan(tmp_var)]=np.nanmean(tmp_var)
                data[t]=tmp_var
        OGCM_Data[i]=data
    else:
        
        if i=='u' :
            OGCM_npth=uNC;
        elif i=='v':
            OGCM_npth=vNC;
        elif i=='temp':
            OGCM_npth=tempNC;
        elif i=='salt':
            OGCM_npth=saltNC;
            
        # tmp_data=np.squeeze(MFDataset(OGCM_npth+'*.nc')[OGCMVar[i]][TIMES_co,:,bry_lat_co,bry_lon_co])
        #tmp_data=xr.open_mfdataset(OGCM_npth+'*.nc',parallel=True)[OGCMVar[i]][TIMES_co,:,bry_lat_co,bry_lon_co].squeeze().values
        tmp_data=MFDataset(OGCM_npth+'*.nc')[OGCMVar[i]][TIMES_co,:,bry_lat_co,bry_lon_co]  
        
        mask = np.isnan(tmp_data)
        DATA = np.ma.array(tmp_data, mask=mask)
        
        tmp_mask_=np.invert(DATA.mask)    
        data=np.zeros([len(Bry_time_num),len(depthO),2,lonG.shape[-1]])
    
        for t in tqdm(range(len(Bry_time_num))):
            for d in range(len(depthO)):
                # Interp mask
                tmp_mask=tmp_mask_[t,d]
                data_=griddata((lonO_s_m[tmp_mask].flatten(),latO_s_m[tmp_mask].flatten()),\
                              DATA[t,d][tmp_mask].flatten(),(lonO_s_m.flatten(),latO_s_m.flatten()),'nearest')
                data_=data_.reshape(latO_s_m.shape)
                
                # Interp 4 Grid
                data_re_=griddata( (lonO_s_m.flatten(),latO_s_m.flatten()), data_.flatten(), (lonG[:2,:],latG[:2,:]) ,'cubic' )
                data[t,d]=data_re_.reshape(lonG[:2,:].shape) #.reshape(-1)
                
                
                tmp_var=data[t,d]
               
                if np.sum(~np.isnan(data[t,d]))==0:
                    data[t,d]=data[t,d-1]

                if np.sum(np.isnan(data[t,d]))!=0:
                    tmp_var[np.isnan(tmp_var)]=np.nanmean(tmp_var)
                    data[t,d]=tmp_var
                    

        OGCM_Data[i]=data
     
cosa_,sina_=np.cos(angle)[:2,:],np.sin(angle)[:2,:] #WESTERN BRY
cosa=np.tile( np.tile(cosa_,(thO,1,1)), (len(Bry_time_num),1,1,1) )
sina=np.tile( np.tile(sina_,(thO,1,1)), (len(Bry_time_num),1,1,1) )

#Process 2D vectors
#modi
ubar_west= ru2.rho2u_2d(OGCM_Data['ubar'][:,-1,:]*cosa[:,0,-1,:]+OGCM_Data['vbar'][:,-1,:]*sina[:,0,-1,:])
vbar_west= ru2.rho2v_3d(OGCM_Data['vbar'][:,:2,:]*cosa[:,0,:2,:]+OGCM_Data['ubar'][:,:2,:]*sina[:,0,:2,:]).squeeze()

#ori
#ubar_north= ru2.rho2u_3d(OGCM_Data['ubar'][:,:,:2]*cosa[:,0,:,:2]+OGCM_Data['vbar'][:,:,:2]*sina[:,0,:,:2]).squeeze()
#vbar_north= ru2.rho2u_2d(OGCM_Data['vbar'][:,:,-1]*cosa[:,0,:,-1]+OGCM_Data['ubar'][:,:,-1]*sina[:,0,:,-1])

#Process 3D vectors
u=ru2.rho2u_3d(OGCM_Data['u'][:,:,-1,:]*cosa[:,:,-1,:]+OGCM_Data['v'][:,:,-1,:]*sina[:,:,-1,:])
v=ru2.rho2v_4d(OGCM_Data['v'][:,:,:2,:]*cosa[:,:,:2,:]-OGCM_Data['u'][:,:,:2,:]*sina[:,:,:2,:]).squeeze()


# Process ROMS Vertical grid
Z=np.zeros(len(depthO)+2)
Z[0]=100;Z[1:-1]=-depthO;Z[-1]=-100000

Rzeta=OGCM_Data['zeta'][:,-1,:] # -1 for northern BRY


zr_=np.zeros([OGCM_Data['zeta'].shape[0],MyVar['Layer_N'],OGCM_Data['zeta'].shape[1],OGCM_Data['zeta'].shape[-1]])
zw=np.zeros([OGCM_Data['zeta'].shape[0],MyVar['Layer_N']+1,OGCM_Data['zeta'].shape[1],OGCM_Data['zeta'].shape[-1]])

for i,n in zip(OGCM_Data['zeta'],range(Rzeta.shape[0])):
    zr_[n,:,:,:]=ru.zlevs(MyVar['Vtransform'],MyVar['Vstretching'],MyVar['Theta_s'],\
             MyVar['Theta_b'],MyVar['Tcline'],MyVar['Layer_N'],\
                 1,topo[:2,:],i);  # -2: ???    
    zw[n,:,:,:]=ru.zlevs(MyVar['Vtransform'],MyVar['Vstretching'],MyVar['Theta_s'],\
             MyVar['Theta_b'],MyVar['Tcline'],MyVar['Layer_N'],\
                 5,topo[:2,:],i) # -2: ???
zu=ru2.rho2u_4d(zr_)[:,:,-1,:]
zv=ru2.rho2v_4d(zr_).squeeze()
dzr=zw[:,1:,:,:]-zw[:,:-1,:,:] # [t,depth,lat,lon]
dzu=ru2.rho2u_4d(dzr)[:,:,-1,:];
dzv=ru2.rho2v_4d(dzr).squeeze();
zr=zr_[:,:,-1,:]

# Add a level on top and bottom with no-gradient
temp,salt=OGCM_Data['temp'][:,:,-1,:],OGCM_Data['salt'][:,:,-1,:]

u1=np.hstack((np.expand_dims(u[:,0,:],axis=1)\
              ,u,np.expand_dims(u[:,-1,:],axis=1)))
v1=np.hstack((np.expand_dims(v[:,0,:],axis=1)\
              ,v,np.expand_dims(v[:,-1,:],axis=1)))
temp=np.hstack((np.expand_dims(temp[:,0,:],axis=1)\
              ,temp,np.expand_dims(temp[:,-1,:],axis=1)))
salt=np.hstack((np.expand_dims(salt[:,0,:],axis=1)\
              ,salt,np.expand_dims(salt[:,-1,:],axis=1)))

print('!!! ztosigma_1d !!!')
u_c_,v_c_=np.zeros_like(zu),np.zeros_like(zv)
temp_c,salt_c=np.zeros_like(zr),np.zeros_like(zr)
for i,j,k,l,n in zip(u1,v1,temp,salt,range(zu.shape[0])): 
    u_c_[n]   =ru.ztosigma_1d(np.flip(i,axis=0),zu[n],np.flipud(Z));
    v_c_[n]   =ru.ztosigma_1d(np.flip(j,axis=0),zv[n],np.flipud(Z));
    temp_c[n]=ru.ztosigma_1d(np.flip(k,axis=0),zr[n],np.flipud(Z));
    salt_c[n]=ru.ztosigma_1d(np.flip(l,axis=0),zr[n],np.flipud(Z));

# =============================================================================

# Conservation
if conserv==1:
    u_c= deepcopy(u_c_)  
    v_c= deepcopy(v_c_) 

    tmpu=np.sum(u_c_*dzu,axis=1)/np.sum(dzu,axis=1)
    tmpv=np.sum(v_c_*dzv,axis=1)/np.sum(dzv,axis=1)

    for i in range(dzu.shape[1]):
        u_c[:,i]=u_c[:,i,:] - tmpu +ubar_west
        v_c[:,i]=v_c[:,i,:] - tmpv +vbar_west

# Barotropic velocities2
ubar_west_=np.sum(u_c*dzu,axis=1)/np.sum(dzu,axis=1)
vbar_west_=np.sum(v_c*dzv,axis=1)/np.sum(dzv,axis=1)

ncI=Dataset(My_Bry,mode='a')
ncI['zeta_west'][:]=Rzeta
# ncI['SSH'][:]=Rzeta
ncI['temp_west'][:]=temp_c
ncI['salt_west'][:]=salt_c
ncI['u_west'][:]=u_c
ncI['v_west'][:]=v_c
ncI['ubar_west'][:]=ubar_west_
ncI['vbar_west'][:]=vbar_west_
ncI.close()


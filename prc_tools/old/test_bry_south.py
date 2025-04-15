# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:16:45 2023

@author: birostris
@email : birostris36@gmail.com

Name : 
Reference :
Description :
"""
PKG_path = 'C:/Users/shjo/Desktop/JNUROMS/'
import sys 
sys.path.append(PKG_path)
import Tools.JNUROMS as jr
from Tools.JNU_create import create_bry2
import numpy as np
from netCDF4 import Dataset,MFDataset,date2num,num2date
import os
from scipy.interpolate import griddata
import datetime as dt
from tqdm import tqdm
from copy import deepcopy
My_Bry='J:/swan/test/test_bry.nc'
My_Grd='J:/swan/test/Grd_4_test_bry.nc'
OGCM_PATH='D:/SODA/'

Bry_title='test_bry'
# My Variables
MyVar={'Layer_N':50,'Vtransform':2,'Vstretching':4,\
       'Theta_s':7,'Theta_b':1,'Tcline':450,'hmin':50}

# OGCM Variables
OGCMVar={'lon_rho':'xt_ocean','lat_rho':'yt_ocean','depth':'st_ocean','time':'time',\
         'lon_u':'xu_ocean','lat_u':'yu_ocean','lon_v':'xu_ocean','lat_v':'yu_ocean',
         'temp':'temp','salt':'salt','u':'u','v':'v','zeta':'ssh'}
conserv=1
OGCMS=[OGCM_PATH+'/'+i for i in os.listdir(OGCM_PATH) if i.endswith('.nc')]

# Get My Grid info
ncG=Dataset(My_Grd)
lonG,latG=ncG['lon_rho'][:],ncG['lat_rho'][:]
angle,topo,mask=ncG['angle'][:],ncG['h'][:],ncG['mask_rho'][:]
ncG.close()

# Get OGCM Grid info
ncO=Dataset(OGCMS[0])
lonO,latO=ncO[OGCMVar['lon_rho']][:],ncO[OGCMVar['lat_rho']][:]
depthO=ncO[OGCMVar['depth']][:]
ncO.close()

# Get OGCM lon lat coordinates for slicing
# =============================================================================
# Process Times
tmp_time_var='time'
t_rng=['2000-01','2000-12']
My_time_ref='days since 1970-1-1 00:00:00'
OGCM_TIMES=MFDataset(OGCM_PATH+'*.nc')[tmp_time_var]
TIME_UNIT=OGCM_TIMES.units
OGCM_times=num2date(OGCM_TIMES[:],TIME_UNIT)
Tst=dt.datetime(int(t_rng[0].split('-')[0]), int(t_rng[0].split('-')[1]),1)
Ted=dt.datetime(int(t_rng[1].split('-')[0]), int(t_rng[1].split('-')[1]),28)
TIMES_co=np.where( (OGCM_times>=Tst)&(OGCM_times<=Ted) )[0]
# =============================================================================
tmp_y,tmp_m=int(t_rng[0].split('-')[0]),int(t_rng[0].split('-')[-1])
tmp_dif=date2num(dt.datetime(tmp_y,tmp_m,1),TIME_UNIT)-date2num(dt.datetime(tmp_y,tmp_m,1),My_time_ref)
Bry_time_time=num2date(OGCM_TIMES[TIMES_co]-tmp_dif,My_time_ref)
Bry_time_num=OGCM_TIMES[TIMES_co]-tmp_dif

thO=ncO[OGCMVar['depth']].shape[0]
atG,onG=lonG.shape
cosa_,sina_=np.cos(angle)[-2:],np.sin(angle)[-2:] #NORTHERN BRY
cosa=np.tile( np.tile(cosa_,(thO,1,1)), (len(Bry_time_num),1,1,1) )
sina=np.tile( np.tile(sina_,(thO,1,1)), (len(Bry_time_num),1,1,1) )

# Make an Inifile
create_bry2(My_Bry,mask,topo,MyVar,Bry_time_num,My_time_ref,Bry_title,ncFormat='NETCDF4')

dlG,dlO=np.diff(latG[:,0]),np.diff(latO)

if np.max(dlG)>=np.max(dlO):
    pass
else:
    tmp_co=np.where(latO>=latG[-1,0])[0][0]
    bry_lat_co=np.array([tmp_co-1,tmp_co])

lonO_s_m,latO_s_m=np.meshgrid(lonO,latO[bry_lat_co])

# Read_Data
OGCM_Data={}#,OGCM_Mask={}

for i in ['u','v','temp','salt','zeta','ubar','vbar']:
    print('!!! Data processing : '+i+' !!!')
    
    if (i=='zeta') or (i=='ubar') or (i=='vbar'):
        data=np.zeros([len(Bry_time_num),2,lonG.shape[-1]])
        if i=='zeta':
            DATA=np.squeeze(MFDataset(OGCMS)[OGCMVar[i]][TIMES_co,bry_lat_co,:])
            tmp_mask_=np.invert(DATA.mask)
            
        elif i=='ubar':
            tmp_u=np.squeeze(MFDataset(OGCMS)[OGCMVar['u']][TIMES_co,:,bry_lat_co,:])
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
            tmp_v=np.squeeze(MFDataset(OGCMS)[OGCMVar['v']][TIMES_co,:,bry_lat_co,:])
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
            data_re_=griddata( (lonO_s_m.flatten(),latO_s_m.flatten()), data_.flatten(), (lonG[-2:,:].flatten(),latG[-2:,:].flatten()) ,'cubic' )
            data[t]=data_re_.reshape(lonG[-2:,:].shape)
        OGCM_Data[i]=data
        
    else:
        data=np.zeros([len(Bry_time_num),len(depthO),2,lonG.shape[-1]])
        DATA=np.squeeze(MFDataset(OGCMS)[OGCMVar[i]][TIMES_co,:,bry_lat_co,:])
        tmp_mask_=np.invert(DATA.mask)
    
        for t in tqdm(range(len(Bry_time_num))):
            for d in range(len(depthO)):
                # Interp mask
                tmp_mask=tmp_mask_[t,d]
                data_=griddata((lonO_s_m[tmp_mask].flatten(),latO_s_m[tmp_mask].flatten()),\
                              DATA[t,d][tmp_mask].flatten(),(lonO_s_m.flatten(),latO_s_m.flatten()),'nearest')
                data_=data_.reshape(latO_s_m.shape)
        
                # Interp 4 Grid
                data_re_=griddata( (lonO_s_m.flatten(),latO_s_m.flatten()), data_.flatten(), (lonG[-2:,:],latG[-2:,:]) ,'cubic' )
                data[t,d]=data_re_.reshape(lonG[-2:,:].shape) #.reshape(-1)
        OGCM_Data[i]=data

# =============================================================================
def rho2u_2d(var):
    N,Lp=var.shape
    L=Lp-1
    var_u=0.5*(var[:,:L]+var[:,1:Lp])
    return var_u
def rho2u_3d(var):
    N,Mp,Lp=var.shape
    L=Lp-1
    var_u=0.5*(var[:,:,:L]+var[:,:,1:Lp])
    return var_u
def rho2u_4d(var):
    T,N,Mp,Lp=var.shape
    L=Lp-1
    var_u=0.5*(var[:,:,:,:L]+var[:,:,:,1:Lp])
    return var_u
def rho2v_3d(var):
    T,Mp,Lp=var.shape
    M=Mp-1
    var_v=0.5*(var[:,:M,:]+var[:,1:Mp,:])
    return var_v
def rho2v_4d(var):
    T,N,Mp,Lp=var.shape
    M=Mp-1
    var_v=0.5*(var[:,:,:M,:]+var[:,:,1:Mp,:])
    return var_v

# 4D angle
cosa_,sina_=np.cos(angle)[-2:],np.sin(angle)[-2:] #NORTHERN BRY
cosa=np.tile( np.tile(cosa_,(thO,1,1)), (len(Bry_time_num),1,1,1) )
sina=np.tile( np.tile(sina_,(thO,1,1)), (len(Bry_time_num),1,1,1) )

#Process 2D vectors
ubar_north= rho2u_2d(OGCM_Data['ubar'][:,-1,:]*cosa[:,0,-1,:]+OGCM_Data['vbar'][:,-1,:]*sina[:,0,-1,:])
vbar_north= rho2v_3d(OGCM_Data['vbar'][:,-2:,:]*cosa[:,0,-2:,:]+OGCM_Data['ubar'][:,-2:,:]*sina[:,0,-2:,:]).squeeze()

#Process 3D vectors
u=rho2u_3d(OGCM_Data['u'][:,:,-1,:]*cosa[:,:,-1,:]+OGCM_Data['v'][:,:,-1,:]*sina[:,:,-1,:])
v=rho2v_4d(OGCM_Data['v'][:,:,-2:,:]*cosa[:,:,-2:,:]-OGCM_Data['u'][:,:,-2:,:]*sina[:,:,-2:,:]).squeeze()

# =============================================================================

# Process ROMS Vertical grid
Z=np.zeros(len(depthO)+2)
Z[0]=100;Z[1:-1]=-depthO;Z[-1]=-100000

Rzeta=OGCM_Data['zeta'][:,-2,:] # -1 for northern BRY

zr_=np.zeros([OGCM_Data['zeta'].shape[0],MyVar['Layer_N'],OGCM_Data['zeta'].shape[1],OGCM_Data['zeta'].shape[-1]])
zw=np.zeros([OGCM_Data['zeta'].shape[0],MyVar['Layer_N']+1,OGCM_Data['zeta'].shape[1],OGCM_Data['zeta'].shape[-1]])

for i,n in zip(OGCM_Data['zeta'],range(Rzeta.shape[0])):
    zr_[n,:,:,:]=jr.zlevs(MyVar['Vtransform'],MyVar['Vstretching'],MyVar['Theta_s'],\
             MyVar['Theta_b'],MyVar['Tcline'],MyVar['Layer_N'],\
                 1,topo[-2:],i);        
    zw[n,:,:,:]=jr.zlevs(MyVar['Vtransform'],MyVar['Vstretching'],MyVar['Theta_s'],\
             MyVar['Theta_b'],MyVar['Tcline'],MyVar['Layer_N'],\
                 5,topo[-2:],i)
zu=rho2u_4d(zr_)[:,:,-1,:]
zv=rho2v_4d(zr_).squeeze()
dzr=zw[:,1:,:,:]-zw[:,:-1,:,:] # [t,depth,lat,lon]
dzu=rho2u_4d(dzr)[:,:,-1,:];
dzv=rho2v_4d(dzr).squeeze();
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
    u_c_[n]   =jr.ztosigma_1d(np.flip(i,axis=0),zu[0],np.flipud(Z));
    v_c_[n]   =jr.ztosigma_1d(np.flip(j,axis=0),zv[0],np.flipud(Z));
    temp_c[n]=jr.ztosigma_1d(np.flip(k,axis=0),zr[0],np.flipud(Z));
    salt_c[n]=jr.ztosigma_1d(np.flip(l,axis=0),zr[0],np.flipud(Z));

# =============================================================================
# =============================================================================

# Conservation
if conserv==1:
    u_c= deepcopy(u_c_)  
    v_c= deepcopy(v_c_) 

    tmpu=np.sum(u_c_*dzu,axis=1)/np.sum(dzu,axis=1)
    tmpv=np.sum(v_c_*dzv,axis=1)/np.sum(dzv,axis=1)

    for i in range(dzu.shape[1]):
        u_c[:,i]=u_c[:,i,:] - tmpu +ubar_north
        v_c[:,i]=v_c[:,i,:] - tmpv +vbar_north

# Barotropic velocities2
ubar_north_=np.sum(u_c*dzu,axis=1)/np.sum(dzu,axis=1)
vbar_north_=np.sum(v_c*dzv,axis=1)/np.sum(dzv,axis=1)

ncI=Dataset(My_Bry,mode='a')
ncI['zeta_north'][:]=Rzeta
# ncI['SSH'][:]=Rzeta
ncI['temp_north'][:]=temp_c
ncI['salt_north'][:]=salt_c
ncI['u_north'][:]=u_c
ncI['v_north'][:]=v_c
ncI['ubar_north'][:]=ubar_north_
ncI['vbar_north'][:]=vbar_north_
ncI.close()





















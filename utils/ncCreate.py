# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 16:29:08 2023

@author: birostris
@email : birostris36@gmail.com

Name : 
Reference :
Description :
"""

import numpy as np
from netCDF4 import Dataset,stringtochar
PKG_path = 'D:/OneDrive/JNUpack/JNUROMS/'
import sys 
sys.path.append(PKG_path)
import utils.ROMS_utils01 as ru

def create_ini(My_Ini,mask,topo,MyVar,Ini_time,Title,ncFormat='NETCDF3_CLASSIC'):
    hmin_=np.min(topo[mask==1]);
    if MyVar['Vtransform']==1 and MyVar['Tcline']>hmin_:
        raise
        
    Mp,Lp=topo.shape
    L,M,Np=Lp-1,Mp-1,MyVar['Layer_N']+1
    
    My_type='INITIAL file'
    history='ROMS'
    
    ncfile = Dataset(My_Ini,mode='w',format=ncFormat)

    ncfile.createDimension('xi_u',L)
    ncfile.createDimension('xi_v',Lp)
    ncfile.createDimension('xi_rho',Lp)
    ncfile.createDimension('eta_u',Mp)
    ncfile.createDimension('eta_v',M)
    ncfile.createDimension('eta_rho',Mp)
    ncfile.createDimension('s_rho',MyVar['Layer_N'])
    ncfile.createDimension('s_w',Np)
    ncfile.createDimension('tracer',2)
    ncfile.createDimension('one',1)
    ncfile.createDimension('ocean_time',1)

    ncfile.createVariable('spherical', 'S1', ('one'))
    ncfile.createVariable('Vtransform', 'f4', ('one'))
    ncfile.createVariable('Vstretching', 'f4', ('one'))
    ncfile.createVariable('tstart', 'f4', ('one'))
    ncfile.createVariable('tend', 'f4', ('one'))
    ncfile.createVariable('theta_s', 'f4', ('one'))
    ncfile.createVariable('theta_b', 'f4', ('one'))
    ncfile.createVariable('Tcline', 'f4', ('one'))
    ncfile.createVariable('hc', 'f4', ('one'))

    ncfile.createVariable('sc_r', 'f4', ('s_rho'))
    ncfile.createVariable('Cs_r', 'f4', ('s_rho'))
    ncfile.createVariable('sc_w', 'f4', ('s_w'))
    ncfile.createVariable('Cs_w', 'f4', ('s_w'))
    
    ncfile.createVariable('ocean_time', 'f4', ('ocean_time'))

    ncfile.createVariable('u', 'f4', ('ocean_time','s_rho','eta_u','xi_u'))
    ncfile.createVariable('v', 'f4', ('ocean_time','s_rho','eta_v','xi_v'))
    ncfile.createVariable('ubar', 'f4', ('ocean_time','eta_u','xi_u'))
    ncfile.createVariable('vbar', 'f4', ('ocean_time','eta_v','xi_v'))
    ncfile.createVariable('zeta', 'f4', ('ocean_time','eta_rho','xi_rho'))
    ncfile.createVariable('temp', 'f4', ('ocean_time','s_rho','eta_rho','xi_rho'))
    ncfile.createVariable('salt', 'f4', ('ocean_time','s_rho','eta_rho','xi_rho'))

    #ncfile.createVariable('uice', 'f4', ('ocean_time','eta_rho','xi_rho'))
    #ncfile.createVariable('vice', 'f4', ('ocean_time','eta_rho','xi_rho'))
    #ncfile.createVariable('aice', 'f4', ('ocean_time','eta_rho','xi_rho'))
    #ncfile.createVariable('hice', 'f4', ('ocean_time','eta_rho','xi_rho'))

    ncfile['Vtransform'].long_name='vertical terrain-following transformation equation'
    ncfile['Vstretching'].long_name='vertical terrain-following stretching function'

    ncfile['tstart'].long_name='start processing day'
    ncfile['tstart'].units='day'

    ncfile['tend'].long_name='end processing day'
    ncfile['tend'].units='day'

    ncfile['theta_s'].long_name='S-coordinate surface control parameter'
    ncfile['theta_s'].units='nondimensional'
    ncfile['theta_b'].long_name='S-coordinate bottom control parameter'
    ncfile['theta_b'].units='nondimensional'


    ncfile['Tcline'].long_name='S-coordinate surface/bottom layer width'
    ncfile['Tcline'].units='meter'
    ncfile['hc'].long_name='S-coordinate parameter, critical depth'
    ncfile['hc'].units='meter'


    ncfile['sc_r'].long_name='S-coordinate at RHO-points'
    ncfile['sc_r'].units='nondimensional'
    
    ncfile['Cs_r'].long_name='S-coordinate stretching curves at RHO-points'
    ncfile['Cs_r'].units='nondimensional'
    
    ncfile['sc_w'].long_name='S-coordinate at W-points'
    ncfile['sc_w'].units='nondimensional'
    
    ncfile['Cs_w'].long_name='S-coordinate stretching curves at W-points'
    ncfile['Cs_w'].units='nondimensional'

    ncfile['u'].long_name='u-momentum component'
    ncfile['u'].units='meter second-1'
    ncfile['v'].long_name='v-momentum component'
    ncfile['v'].units='meter second-1'

    ncfile['ubar'].long_name='vertically integrated u-momentum component'
    ncfile['ubar'].units='meter second-1'
    ncfile['vbar'].long_name='vertically integrated v-momentum component'
    ncfile['vbar'].units='meter second-1'

    ncfile['zeta'].long_name='free-surface'
    ncfile['zeta'].units='meter'

    ncfile['temp'].long_name='potential temperature'
    ncfile['temp'].units='Celsius'
    ncfile['salt'].long_name='salinity'
    ncfile['salt'].units='PSU'

    #ncfile['uice'].long_name='u-component of ice velocity'
    #ncfile['uice'].units='meter second-1'
    #ncfile['vice'].long_name='v-component of ice velocity'
    #ncfile['vice'].units='meter second-1'
    #ncfile['aice'].long_name='fraction of cell covered by ice'
    #ncfile['aice'].units='nondimensional'
    #ncfile['hice'].long_name='average ice thickness in cell'
    #ncfile['hice'].units='meter'

    ncfile.title=Title
    ncfile.clim_file=My_Ini
    ncfile.grd_file=''
    ncfile.type=My_type
    ncfile.history=history
    
    sc_r,Cs_r=ru.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],0)

    sc_w,Cs_w=ru.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],1)

    ncfile['spherical'][:]='T'
    ncfile['Vtransform'][:]=MyVar['Vtransform']
    ncfile['Vstretching'][:]=MyVar['Vstretching']
    ncfile['tstart'][:]=0
    ncfile['tend'][:]=0
    ncfile['theta_s'][:]=MyVar['Theta_s']
    ncfile['theta_b'][:]=MyVar['Theta_b']
    ncfile['Tcline'][:]=MyVar['Tcline']
    ncfile['hc'][:]=MyVar['Tcline']
    ncfile['sc_r'][:]=sc_r
    ncfile['Cs_r'][:]=Cs_r
    ncfile['sc_w'][:]=sc_w
    ncfile['Cs_w'][:]=Cs_w
    ncfile['ocean_time'][:]=Ini_time
    ncfile['u'][:]=0
    ncfile['v'][:]=0
    ncfile['zeta'][:]=0
    ncfile['ubar'][:]=0
    ncfile['vbar'][:]=0
    ncfile['temp'][:]=0
    ncfile['salt'][:]=0

    #ncfile['aice'][:]=0
    #ncfile['hice'][:]=0
    #ncfile['uice'][:]=0
    #ncfile['vice'][:]=0

    ncfile.close()
    


def create_bry_VVV(My_Bry,mask,topo,MyVar,NSEW,Bry_time,My_time_ref,Title,ncFormat='NETCDF3_CLASSIC'):

    hmin_=np.min(topo[mask==1]);
    if MyVar['Vtransform']==1 and MyVar['Tcline']>hmin_:
        raise
        
    Mp,Lp=topo.shape
    L,M,Np=Lp-1,Mp-1,MyVar['Layer_N']+1
    
    My_type='INITIAL file'
    history='ROMS'
    
    ncfile = Dataset(My_Bry,mode='w',format=ncFormat)

    ncfile.createDimension('xi_u',L)
    ncfile.createDimension('xi_v',Lp)
    ncfile.createDimension('xi_rho',Lp)
    ncfile.createDimension('eta_u',Mp)
    ncfile.createDimension('eta_v',M)
    ncfile.createDimension('eta_rho',Mp)
    ncfile.createDimension('s_rho',MyVar['Layer_N'])
    ncfile.createDimension('s_w',Np)
    ncfile.createDimension('tracer',2)
    ncfile.createDimension('one',1)
    ncfile.createDimension('bry_time',len(Bry_time))

    ncfile.createVariable('spherical', 'S1', ('one'))
    ncfile['spherical'].long_name='grid type logical switch'
    ncfile['spherical'].flag_values='T,F'
    ncfile['spherical'].flag_meanings='spherical cartesian'
    
    ncfile.createVariable('Vtransform', 'f4', ('one'))
    ncfile['Vtransform'].long_name='vertical terrain-following transformation equation'

    ncfile.createVariable('Vstretching', 'f4', ('one'))
    ncfile['Vstretching'].long_name='vertical terrain-following stretching function'

    ncfile.createVariable('tstart', 'f4', ('one'))
    ncfile['tstart'].long_name='start processing day'
    ncfile['tstart'].units='day'

    ncfile.createVariable('tend', 'f4', ('one'))
    ncfile['tend'].long_name='end processing day'
    ncfile['tend'].units='day'

    ncfile.createVariable('theta_s', 'f4', ('one'))
    ncfile['theta_s'].long_name='S-coordinate surface control parameter'
    ncfile['theta_s'].units='nondimensional'
    
    ncfile.createVariable('theta_b', 'f4', ('one'))
    ncfile['theta_b'].long_name='S-coordinate bottom control parameter'
    ncfile['theta_b'].units='nondimensional'
    
    ncfile.createVariable('Tcline', 'f4', ('one'))
    ncfile['Tcline'].long_name='S-coordinate surface/bottom layer width'
    ncfile['Tcline'].units='meter'
    
    ncfile.createVariable('hc', 'f4', ('one'))
    ncfile['hc'].long_name='S-coordinate parameter, critical depth'
    ncfile['hc'].units='meter'

    ncfile.createVariable('sc_r', 'f4', ('s_rho'))
    ncfile['sc_r'].long_name='S-coordinate at RHO-points'
    ncfile['sc_r'].valid_min = -1.
    ncfile['sc_r'].valid_max = 0.
    ncfile['sc_r'].positive = 'up'
    if MyVar['Vtransform']==1:
        ncfile['sc_r'].standard_name = 'ocena_s_coordinate_g1'
    elif MyVar['Vtransform']==2:
        ncfile['sc_r'].standard_name = 'ocena_s_coordinate_g2'
    ncfile['sc_r'].formula_terms = 's: s_rho C: Cs_r eta: zeta depth: h depth_c: hc'
    
    ncfile.createVariable('sc_w', 'f4', ('s_w'))
    ncfile['sc_w'].long_name='S-coordinate at W-points'
    ncfile['sc_w'].valid_min = -1.;
    ncfile['sc_w'].valid_max = 0.;
    ncfile['sc_w'].positive = 'up';
    if MyVar['Vtransform']==1:
        ncfile['sc_w'].standard_name = 'ocena_s_coordinate_g1';
    elif MyVar['Vtransform']==2:
        ncfile['sc_w'].standard_name = 'ocena_s_coordinate_g2';
    ncfile['sc_r'].formula_terms = 's: s_w C: Cs_w eta: zeta depth: h depth_c: hc';


    ncfile.createVariable('Cs_r', 'f4', ('s_rho'))
    ncfile['Cs_r'].long_name='S-coordinate stretching curves at RHO-points'
    ncfile['Cs_r'].units='nondimensional'
    ncfile['Cs_r'].valid_min = -1;
    ncfile['Cs_r'].valid_max = 0;

    ncfile.createVariable('Cs_w', 'f4', ('s_w'))
    ncfile['Cs_w'].long_name='S-coordinate stretching curves at W-points'
    ncfile['Cs_w'].units='nondimensional'
    ncfile['Cs_w'].valid_min = -1;
    ncfile['Cs_w'].valid_max = 0;

    ncfile.createVariable('bry_time', 'f4', ('bry_time'))
    ncfile['bry_time'].long_name='ime for boundary'
    ncfile['bry_time'].units=My_time_ref

    m=-1
    for n in NSEW:
        m+=1
        if not n:
            continue
        else:
            if m==0:
                myGrid='xi'
                mydirc='north'
            elif m==1:
                myGrid='xi'
                mydirc='south'
            elif m==2:
                myGrid='eta'
                mydirc='east'
            elif m==3:
                myGrid='eta'
                mydirc='west'
        
        print('!!! Make bry: ',mydirc+' !!!')

        ncfile.createVariable('u_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_u'))
        ncfile['u_'+mydirc].long_name='u-momentum component'
        ncfile['u_'+mydirc].units='meter second-1'
        ncfile['u_'+mydirc].coordinates = 'lon_u s_rho bry_time'
        
        ncfile.createVariable('v_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_v'))
        ncfile['v_'+mydirc].long_name='v-momentum component'
        ncfile['v_'+mydirc].units='meter second-1'
        ncfile['v_'+mydirc].coordinates = 'lon_v s_rho bry_time'
        
        ncfile.createVariable('temp_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_rho'))
        ncfile['temp_'+mydirc].long_name='potential temperature'
        ncfile['temp_'+mydirc].units='Celsius'
        ncfile['temp_'+mydirc].coordinates = 'lon_rho s_rho bry_time'
    
        ncfile.createVariable('salt_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_rho'))
        ncfile['salt_'+mydirc].long_name='salinity'
        ncfile['salt_'+mydirc].units='PSU'
        ncfile['salt_'+mydirc].coordinates = 'lon_rho s_rho bry_time'
    
        ncfile.createVariable('ubar_'+mydirc, 'f4', ('bry_time',myGrid+'_u'))
        ncfile['ubar_'+mydirc].long_name='vertically integrated u-momentum component'
        ncfile['ubar_'+mydirc].units='meter second-1'
        ncfile['ubar_'+mydirc].coordinates = 'lon_u bry_time'
        
        ncfile.createVariable('vbar_'+mydirc, 'f4', ('bry_time',myGrid+'_v'))
        ncfile['vbar_'+mydirc].long_name='vertically integrated v-momentum component'
        ncfile['vbar_'+mydirc].units='meter second-1'
        ncfile['vbar_'+mydirc].coordinates = 'lon_v bry_time'
        
        ncfile.createVariable('zeta_'+mydirc, 'f4', ('bry_time',myGrid+'_rho'))
        ncfile['zeta_'+mydirc].long_name='free-surface'
        ncfile['zeta_'+mydirc].units='meter'
        ncfile['zeta_'+mydirc].coordinates = 'lon_rho bry_time';
        
        ncfile['u_'+mydirc][:]=0
        ncfile['v_'+mydirc][:]=0
        ncfile['zeta_'+mydirc][:]=0
        ncfile['ubar_'+mydirc][:]=0
        ncfile['vbar_'+mydirc][:]=0
        ncfile['temp_'+mydirc][:]=0
        ncfile['salt_'+mydirc][:]=0


    ncfile.title=Title
    ncfile.clim_file=My_Bry
    ncfile.grd_file=''
    ncfile.type=My_type
    ncfile.history=history
    
    sc_r,Cs_r=ru.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],0)

    sc_w,Cs_w=ru.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],1)

    ncfile['spherical'][:]='T'
    ncfile['Vtransform'][:]=MyVar['Vtransform']
    ncfile['Vstretching'][:]=MyVar['Vstretching']
    ncfile['tstart'][:]=0
    ncfile['tend'][:]=0
    ncfile['theta_s'][:]=MyVar['Theta_s']
    ncfile['theta_b'][:]=MyVar['Theta_b']
    ncfile['Tcline'][:]=MyVar['Tcline']
    ncfile['hc'][:]=MyVar['Tcline']
    ncfile['sc_r'][:]=sc_r
    ncfile['Cs_r'][:]=Cs_r
    ncfile['sc_w'][:]=sc_w
    ncfile['Cs_w'][:]=Cs_w
    ncfile['bry_time'][:]=Bry_time


    ncfile.close()
    
def create_bry_ust(My_Bry,mask,topo,MyVar,NSEW,Bry_time,My_time_ref,Title,ncFormat='NETCDF3_CLASSIC'):

    hmin_=np.min(topo[mask==1]);
    if MyVar['Vtransform']==1 and MyVar['Tcline']>hmin_:
        raise
        
    Mp,Lp=topo.shape
    L,M,Np=Lp-1,Mp-1,MyVar['Layer_N']+1
    
    My_type='INITIAL file'
    history='ROMS'
    
    ncfile = Dataset(My_Bry,mode='w',format=ncFormat)

    ncfile.createDimension('xi_u',L)
    ncfile.createDimension('xi_v',Lp)
    ncfile.createDimension('xi_rho',Lp)
    ncfile.createDimension('eta_u',Mp)
    ncfile.createDimension('eta_v',M)
    ncfile.createDimension('eta_rho',Mp)
    ncfile.createDimension('s_rho',MyVar['Layer_N'])
    ncfile.createDimension('s_w',Np)
    ncfile.createDimension('tracer',2)
    ncfile.createDimension('one',1)
    ncfile.createDimension('bry_time',len(Bry_time))

    ncfile.createVariable('spherical', 'S1', ('one'))
    ncfile['spherical'].long_name='grid type logical switch'
    ncfile['spherical'].flag_values='T,F'
    ncfile['spherical'].flag_meanings='spherical cartesian'
    
    ncfile.createVariable('Vtransform', 'f4', ('one'))
    ncfile['Vtransform'].long_name='vertical terrain-following transformation equation'

    ncfile.createVariable('Vstretching', 'f4', ('one'))
    ncfile['Vstretching'].long_name='vertical terrain-following stretching function'

    ncfile.createVariable('tstart', 'f4', ('one'))
    ncfile['tstart'].long_name='start processing day'
    ncfile['tstart'].units='day'

    ncfile.createVariable('tend', 'f4', ('one'))
    ncfile['tend'].long_name='end processing day'
    ncfile['tend'].units='day'

    ncfile.createVariable('theta_s', 'f4', ('one'))
    ncfile['theta_s'].long_name='S-coordinate surface control parameter'
    ncfile['theta_s'].units='nondimensional'
    
    ncfile.createVariable('theta_b', 'f4', ('one'))
    ncfile['theta_b'].long_name='S-coordinate bottom control parameter'
    ncfile['theta_b'].units='nondimensional'
    
    ncfile.createVariable('Tcline', 'f4', ('one'))
    ncfile['Tcline'].long_name='S-coordinate surface/bottom layer width'
    ncfile['Tcline'].units='meter'
    
    ncfile.createVariable('hc', 'f4', ('one'))
    ncfile['hc'].long_name='S-coordinate parameter, critical depth'
    ncfile['hc'].units='meter'

    ncfile.createVariable('sc_r', 'f4', ('s_rho'))
    ncfile['sc_r'].long_name='S-coordinate at RHO-points'
    ncfile['sc_r'].valid_min = -1.
    ncfile['sc_r'].valid_max = 0.
    ncfile['sc_r'].positive = 'up'
    if MyVar['Vtransform']==1:
        ncfile['sc_r'].standard_name = 'ocena_s_coordinate_g1'
    elif MyVar['Vtransform']==2:
        ncfile['sc_r'].standard_name = 'ocena_s_coordinate_g2'
    ncfile['sc_r'].formula_terms = 's: s_rho C: Cs_r eta: zeta depth: h depth_c: hc'
    
    ncfile.createVariable('sc_w', 'f4', ('s_w'))
    ncfile['sc_w'].long_name='S-coordinate at W-points'
    ncfile['sc_w'].valid_min = -1.;
    ncfile['sc_w'].valid_max = 0.;
    ncfile['sc_w'].positive = 'up';
    if MyVar['Vtransform']==1:
        ncfile['sc_w'].standard_name = 'ocena_s_coordinate_g1';
    elif MyVar['Vtransform']==2:
        ncfile['sc_w'].standard_name = 'ocena_s_coordinate_g2';
    ncfile['sc_r'].formula_terms = 's: s_w C: Cs_w eta: zeta depth: h depth_c: hc';


    ncfile.createVariable('Cs_r', 'f4', ('s_rho'))
    ncfile['Cs_r'].long_name='S-coordinate stretching curves at RHO-points'
    ncfile['Cs_r'].units='nondimensional'
    ncfile['Cs_r'].valid_min = -1;
    ncfile['Cs_r'].valid_max = 0;

    ncfile.createVariable('Cs_w', 'f4', ('s_w'))
    ncfile['Cs_w'].long_name='S-coordinate stretching curves at W-points'
    ncfile['Cs_w'].units='nondimensional'
    ncfile['Cs_w'].valid_min = -1;
    ncfile['Cs_w'].valid_max = 0;

    ncfile.createVariable('bry_time', 'f4', ('bry_time'))
    ncfile['bry_time'].long_name='ime for boundary'
    ncfile['bry_time'].units=My_time_ref

    m=-1
    for n in NSEW:
        m+=1
        if not n:
            continue
        else:
            if m==0:
                myGrid='eta'
                mydirc='north'
            elif m==1:
                myGrid='eta'
                mydirc='south'
            elif m==2:
                myGrid='xi'
                mydirc='east'
            elif m==3:
                myGrid='xi'
                mydirc='west'
        
        print('!!! Make bry: ',mydirc+' !!!')

        ncfile.createVariable('u_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_u'))
        ncfile['u_'+mydirc].long_name='u-momentum component'
        ncfile['u_'+mydirc].units='meter second-1'
        ncfile['u_'+mydirc].coordinates = 'lon_u s_rho bry_time'
        
        ncfile.createVariable('v_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_v'))
        ncfile['v_'+mydirc].long_name='v-momentum component'
        ncfile['v_'+mydirc].units='meter second-1'
        ncfile['v_'+mydirc].coordinates = 'lon_v s_rho bry_time'
        
        ncfile.createVariable('temp_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_rho'))
        ncfile['temp_'+mydirc].long_name='potential temperature'
        ncfile['temp_'+mydirc].units='Celsius'
        ncfile['temp_'+mydirc].coordinates = 'lon_rho s_rho bry_time'
    
        ncfile.createVariable('salt_'+mydirc, 'f4', ('bry_time','s_rho',myGrid+'_rho'))
        ncfile['salt_'+mydirc].long_name='salinity'
        ncfile['salt_'+mydirc].units='PSU'
        ncfile['salt_'+mydirc].coordinates = 'lon_rho s_rho bry_time'
    
        ncfile.createVariable('ubar_'+mydirc, 'f4', ('bry_time',myGrid+'_u'))
        ncfile['ubar_'+mydirc].long_name='vertically integrated u-momentum component'
        ncfile['ubar_'+mydirc].units='meter second-1'
        ncfile['ubar_'+mydirc].coordinates = 'lon_u bry_time'
        
        ncfile.createVariable('vbar_'+mydirc, 'f4', ('bry_time',myGrid+'_v'))
        ncfile['vbar_'+mydirc].long_name='vertically integrated v-momentum component'
        ncfile['vbar_'+mydirc].units='meter second-1'
        ncfile['vbar_'+mydirc].coordinates = 'lon_v bry_time'
        
        ncfile.createVariable('zeta_'+mydirc, 'f4', ('bry_time',myGrid+'_rho'))
        ncfile['zeta_'+mydirc].long_name='free-surface'
        ncfile['zeta_'+mydirc].units='meter'
        ncfile['zeta_'+mydirc].coordinates = 'lon_rho bry_time';
        
        ncfile['u_'+mydirc][:]=0
        ncfile['v_'+mydirc][:]=0
        ncfile['zeta_'+mydirc][:]=0
        ncfile['ubar_'+mydirc][:]=0
        ncfile['vbar_'+mydirc][:]=0
        ncfile['temp_'+mydirc][:]=0
        ncfile['salt_'+mydirc][:]=0


    ncfile.title=Title
    ncfile.clim_file=My_Bry
    ncfile.grd_file=''
    ncfile.type=My_type
    ncfile.history=history
    
    sc_r,Cs_r=ru.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],0)

    sc_w,Cs_w=ru.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],1)

    ncfile['spherical'][:]='T'
    ncfile['Vtransform'][:]=MyVar['Vtransform']
    ncfile['Vstretching'][:]=MyVar['Vstretching']
    ncfile['tstart'][:]=0
    ncfile['tend'][:]=0
    ncfile['theta_s'][:]=MyVar['Theta_s']
    ncfile['theta_b'][:]=MyVar['Theta_b']
    ncfile['Tcline'][:]=MyVar['Tcline']
    ncfile['hc'][:]=MyVar['Tcline']
    ncfile['sc_r'][:]=sc_r
    ncfile['Cs_r'][:]=Cs_r
    ncfile['sc_w'][:]=sc_w
    ncfile['Cs_w'][:]=Cs_w
    ncfile['bry_time'][:]=Bry_time


    ncfile.close()
    
    
    
    
    
    
def create_bry(My_Bry,mask,topo,MyVar,Bry_time,Title,ncFormat='NETCDF3_CLASSIC'):

    hmin_=np.min(topo[mask==1]);
    if MyVar['Vtransform']==1 and MyVar['Tcline']>hmin_:
        raise
        
    Mp,Lp=topo.shape
    L,M,Np=Lp-1,Mp-1,MyVar['Layer_N']+1
    
    My_type='INITIAL file'
    history='ROMS'
    
    ncfile = Dataset(My_Bry,mode='w',format=ncFormat)

    ncfile.createDimension('xi_u',L)
    ncfile.createDimension('xi_v',Lp)
    ncfile.createDimension('xi_rho',Lp)
    ncfile.createDimension('eta_u',Mp)
    ncfile.createDimension('eta_v',M)
    ncfile.createDimension('eta_rho',Mp)
    ncfile.createDimension('s_rho',MyVar['Layer_N'])
    ncfile.createDimension('s_w',Np)
    ncfile.createDimension('tracer',2)
    ncfile.createDimension('one',1)
    ncfile.createDimension('bry_time',len(Bry_time))

    ncfile.createVariable('spherical', 'S1', ('one'))
    ncfile.createVariable('Vtransform', 'f4', ('one'))
    ncfile.createVariable('Vstretching', 'f4', ('one'))
    ncfile.createVariable('tstart', 'f4', ('one'))
    ncfile.createVariable('tend', 'f4', ('one'))
    ncfile.createVariable('theta_s', 'f4', ('one'))
    ncfile.createVariable('theta_b', 'f4', ('one'))
    ncfile.createVariable('Tcline', 'f4', ('one'))
    ncfile.createVariable('hc', 'f4', ('one'))

    ncfile.createVariable('sc_r', 'f4', ('s_rho'))
    ncfile.createVariable('Cs_r', 'f4', ('s_rho'))
    ncfile.createVariable('sc_w', 'f4', ('s_w'))
    ncfile.createVariable('Cs_w', 'f4', ('s_w'))
    
    ncfile.createVariable('bry_time', 'f4', ('bry_time'))

    ncfile.createVariable('u_north', 'f4', ('bry_time','s_rho','xi_u'))
    ncfile.createVariable('v_north', 'f4', ('bry_time','s_rho','xi_v'))
    ncfile.createVariable('ubar_north', 'f4', ('bry_time','xi_u'))
    ncfile.createVariable('vbar_north', 'f4', ('bry_time','xi_v'))
    ncfile.createVariable('zeta_north', 'f4', ('bry_time','xi_rho'))
    ncfile.createVariable('temp_north', 'f4', ('bry_time','s_rho','xi_rho'))
    ncfile.createVariable('salt_north', 'f4', ('bry_time','s_rho','xi_rho'))

    ncfile.createVariable('uice_north', 'f4', ('bry_time','xi_rho'))
    ncfile.createVariable('vice_north', 'f4', ('bry_time','xi_rho'))
    ncfile.createVariable('aice_north', 'f4', ('bry_time','xi_rho'))
    ncfile.createVariable('hice_north', 'f4', ('bry_time','xi_rho'))

    ncfile['Vtransform'].long_name='vertical terrain-following transformation equation'
    ncfile['Vstretching'].long_name='vertical terrain-following stretching function'

    ncfile['tstart'].long_name='start processing day'
    ncfile['tstart'].units='day'

    ncfile['tend'].long_name='end processing day'
    ncfile['tend'].units='day'

    ncfile['theta_s'].long_name='S-coordinate surface control parameter'
    ncfile['theta_s'].units='nondimensional'
    ncfile['theta_b'].long_name='S-coordinate bottom control parameter'
    ncfile['theta_b'].units='nondimensional'

    ncfile['Tcline'].long_name='S-coordinate surface/bottom layer width'
    ncfile['Tcline'].units='meter'
    ncfile['hc'].long_name='S-coordinate parameter, critical depth'
    ncfile['hc'].units='meter'

    ncfile['sc_r'].long_name='S-coordinate at RHO-points'
    ncfile['sc_r'].units='nondimensional'
    
    ncfile['Cs_r'].long_name='S-coordinate stretching curves at RHO-points'
    ncfile['Cs_r'].units='nondimensional'
    
    ncfile['sc_w'].long_name='S-coordinate at W-points'
    ncfile['sc_w'].units='nondimensional'
    
    ncfile['Cs_w'].long_name='S-coordinate stretching curves at W-points'
    ncfile['Cs_w'].units='nondimensional'

    ncfile['u_north'].long_name='u-momentum component'
    ncfile['u_north'].units='meter second-1'
    ncfile['v_north'].long_name='v-momentum component'
    ncfile['v_north'].units='meter second-1'

    ncfile['ubar_north'].long_name='vertically integrated u-momentum component'
    ncfile['ubar_north'].units='meter second-1'
    ncfile['vbar_north'].long_name='vertically integrated v-momentum component'
    ncfile['vbar_north'].units='meter second-1'

    ncfile['zeta_north'].long_name='free-surface'
    ncfile['zeta_north'].units='meter'

    ncfile['temp_north'].long_name='potential temperature'
    ncfile['temp_north'].units='Celsius'
    ncfile['salt_north'].long_name='salinity'
    ncfile['salt_north'].units='PSU'

    ncfile['uice_north'].long_name='u-component of ice velocity'
    ncfile['uice_north'].units='meter second-1'
    ncfile['vice_north'].long_name='v-component of ice velocity'
    ncfile['vice_north'].units='meter second-1'
    ncfile['aice_north'].long_name='fraction of cell covered by ice'
    ncfile['aice_north'].units='nondimensional'
    ncfile['hice_north'].long_name='average ice thickness in cell'
    ncfile['hice_north'].units='meter'

    ncfile.title=Title
    ncfile.clim_file=My_Bry
    ncfile.grd_file=''
    ncfile.type=My_type
    ncfile.history=history
    
    sc_r,Cs_r=jr.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],0)

    sc_w,Cs_w=jr.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],1)

    ncfile['spherical'][:]='T'
    ncfile['Vtransform'][:]=MyVar['Vtransform']
    ncfile['Vstretching'][:]=MyVar['Vstretching']
    ncfile['tstart'][:]=0
    ncfile['tend'][:]=0
    ncfile['theta_s'][:]=MyVar['Theta_s']
    ncfile['theta_b'][:]=MyVar['Theta_b']
    ncfile['Tcline'][:]=MyVar['Tcline']
    ncfile['hc'][:]=MyVar['Tcline']
    ncfile['sc_r'][:]=sc_r
    ncfile['Cs_r'][:]=Cs_r
    ncfile['sc_w'][:]=sc_w
    ncfile['Cs_w'][:]=Cs_w
    ncfile['bry_time'][:]=Bry_time
    ncfile['u_north'][:]=0
    ncfile['v_north'][:]=0
    ncfile['zeta_north'][:]=0
    ncfile['ubar_north'][:]=0
    ncfile['vbar_north'][:]=0
    ncfile['temp_north'][:]=0
    ncfile['salt_north'][:]=0

    ncfile['aice_north'][:]=0
    ncfile['hice_north'][:]=0
    ncfile['uice_north'][:]=0
    ncfile['vice_north'][:]=0

    ncfile.close()
    
    

    
def create_bry2(My_Bry,mask,topo,MyVar,Bry_time,My_time_ref,Title,ncFormat='NETCDF3_CLASSIC'):

    hmin_=np.min(topo[mask==1]);
    if MyVar['Vtransform']==1 and MyVar['Tcline']>hmin_:
        raise
        
    Mp,Lp=topo.shape
    L,M,Np=Lp-1,Mp-1,MyVar['Layer_N']+1
    
    My_type='INITIAL file'
    history='ROMS'
    
    ncfile = Dataset(My_Bry,mode='w',format=ncFormat)

    ncfile.createDimension('xi_u',L)
    ncfile.createDimension('xi_v',Lp)
    ncfile.createDimension('xi_rho',Lp)
    ncfile.createDimension('eta_u',Mp)
    ncfile.createDimension('eta_v',M)
    ncfile.createDimension('eta_rho',Mp)
    ncfile.createDimension('s_rho',MyVar['Layer_N'])
    ncfile.createDimension('s_w',Np)
    ncfile.createDimension('tracer',2)
    ncfile.createDimension('one',1)
    ncfile.createDimension('bry_time',len(Bry_time))

    ncfile.createVariable('spherical', 'S1', ('one'))
    ncfile['spherical'].long_name='grid type logical switch'
    ncfile['spherical'].flag_values='T,F'
    ncfile['spherical'].flag_meanings='spherical cartesian'
    
    ncfile.createVariable('Vtransform', 'f4', ('one'))
    ncfile['Vtransform'].long_name='vertical terrain-following transformation equation'

    ncfile.createVariable('Vstretching', 'f4', ('one'))
    ncfile['Vstretching'].long_name='vertical terrain-following stretching function'

    ncfile.createVariable('tstart', 'f4', ('one'))
    ncfile['tstart'].long_name='start processing day'
    ncfile['tstart'].units='day'

    ncfile.createVariable('tend', 'f4', ('one'))
    ncfile['tend'].long_name='end processing day'
    ncfile['tend'].units='day'

    ncfile.createVariable('theta_s', 'f4', ('one'))
    ncfile['theta_s'].long_name='S-coordinate surface control parameter'
    ncfile['theta_s'].units='nondimensional'
    
    ncfile.createVariable('theta_b', 'f4', ('one'))
    ncfile['theta_b'].long_name='S-coordinate bottom control parameter'
    ncfile['theta_b'].units='nondimensional'
    
    ncfile.createVariable('Tcline', 'f4', ('one'))
    ncfile['Tcline'].long_name='S-coordinate surface/bottom layer width'
    ncfile['Tcline'].units='meter'
    
    ncfile.createVariable('hc', 'f4', ('one'))
    ncfile['hc'].long_name='S-coordinate parameter, critical depth'
    ncfile['hc'].units='meter'

    ncfile.createVariable('sc_r', 'f4', ('s_rho'))
    ncfile['sc_r'].long_name='S-coordinate at RHO-points'
    ncfile['sc_r'].valid_min = -1.
    ncfile['sc_r'].valid_max = 0.
    ncfile['sc_r'].positive = 'up'
    if MyVar['Vtransform']==1:
        ncfile['sc_r'].standard_name = 'ocena_s_coordinate_g1'
    elif MyVar['Vtransform']==2:
        ncfile['sc_r'].standard_name = 'ocena_s_coordinate_g2'
    ncfile['sc_r'].formula_terms = 's: s_rho C: Cs_r eta: zeta depth: h depth_c: hc'
    
    ncfile.createVariable('sc_w', 'f4', ('s_w'))
    ncfile['sc_w'].long_name='S-coordinate at W-points'
    ncfile['sc_w'].valid_min = -1.;
    ncfile['sc_w'].valid_max = 0.;
    ncfile['sc_w'].positive = 'up';
    if MyVar['Vtransform']==1:
        ncfile['sc_w'].standard_name = 'ocena_s_coordinate_g1';
    elif MyVar['Vtransform']==2:
        ncfile['sc_w'].standard_name = 'ocena_s_coordinate_g2';
    ncfile['sc_r'].formula_terms = 's: s_w C: Cs_w eta: zeta depth: h depth_c: hc';


    ncfile.createVariable('Cs_r', 'f4', ('s_rho'))
    ncfile['Cs_r'].long_name='S-coordinate stretching curves at RHO-points'
    ncfile['Cs_r'].units='nondimensional'
    ncfile['Cs_r'].valid_min = -1;
    ncfile['Cs_r'].valid_max = 0;

    ncfile.createVariable('Cs_w', 'f4', ('s_w'))
    ncfile['Cs_w'].long_name='S-coordinate stretching curves at W-points'
    ncfile['Cs_w'].units='nondimensional'
    ncfile['Cs_w'].valid_min = -1;
    ncfile['Cs_w'].valid_max = 0;

    ncfile.createVariable('bry_time', 'f4', ('bry_time'))
    ncfile['bry_time'].long_name='ime for boundary'
    ncfile['bry_time'].units=My_time_ref

    ncfile.createVariable('u_north', 'f4', ('bry_time','s_rho','xi_u'))
    ncfile['u_north'].long_name='u-momentum component'
    ncfile['u_north'].units='meter second-1'
    ncfile['u_north'].coordinates = 'lon_u s_rho bry_time'
    
    ncfile.createVariable('v_north', 'f4', ('bry_time','s_rho','xi_v'))
    ncfile['v_north'].long_name='v-momentum component'
    ncfile['v_north'].units='meter second-1'
    ncfile['v_north'].coordinates = 'lon_v s_rho bry_time'
    
    ncfile.createVariable('temp_north', 'f4', ('bry_time','s_rho','xi_rho'))
    ncfile['temp_north'].long_name='potential temperature'
    ncfile['temp_north'].units='Celsius'
    ncfile['temp_north'].coordinates = 'lon_rho s_rho bry_time'

    ncfile.createVariable('salt_north', 'f4', ('bry_time','s_rho','xi_rho'))
    ncfile['salt_north'].long_name='salinity'
    ncfile['salt_north'].units='PSU'
    ncfile['salt_north'].coordinates = 'lon_rho s_rho bry_time'

    ncfile.createVariable('ubar_north', 'f4', ('bry_time','xi_u'))
    ncfile['ubar_north'].long_name='vertically integrated u-momentum component'
    ncfile['ubar_north'].units='meter second-1'
    ncfile['ubar_north'].coordinates = 'lon_u bry_time'
    
    ncfile.createVariable('vbar_north', 'f4', ('bry_time','xi_v'))
    ncfile['vbar_north'].long_name='vertically integrated v-momentum component'
    ncfile['vbar_north'].units='meter second-1'
    ncfile['vbar_north'].coordinates = 'lon_v bry_time'
    
    ncfile.createVariable('zeta_north', 'f4', ('bry_time','xi_rho'))
    ncfile['zeta_north'].long_name='free-surface'
    ncfile['zeta_north'].units='meter'
    ncfile['zeta_north'].coordinates = 'lon_rho bry_time';

    ncfile.createVariable('uice_north', 'f4', ('bry_time','xi_rho'))
    ncfile['uice_north'].long_name='u-component of ice velocity'
    ncfile['uice_north'].units='meter second-1'
    ncfile['uice_north'].coordinates = 'lon_rho bry_time'
    
    ncfile.createVariable('vice_north', 'f4', ('bry_time','xi_rho'))
    ncfile['vice_north'].long_name='v-component of ice velocity'
    ncfile['vice_north'].units='meter second-1'
    ncfile['vice_north'].coordinates = 'lon_rho bry_time'
    
    ncfile.createVariable('aice_north', 'f4', ('bry_time','xi_rho'))
    ncfile['aice_north'].long_name='fraction of cell covered by ice'
    ncfile['aice_north'].units='nondimensional'
    ncfile['aice_north'].coordinates = 'lon_rho bry_time'
    
    ncfile.createVariable('hice_north', 'f4', ('bry_time','xi_rho'))
    ncfile['hice_north'].long_name='average ice thickness in cell'
    ncfile['hice_north'].units='meter'
    ncfile['hice_north'].coordinates = 'lon_rho bry_time'

    ncfile.title=Title
    ncfile.clim_file=My_Bry
    ncfile.grd_file=''
    ncfile.type=My_type
    ncfile.history=history
    
    sc_r,Cs_r=jr.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],0)

    sc_w,Cs_w=jr.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],1)

    ncfile['spherical'][:]='T'
    ncfile['Vtransform'][:]=MyVar['Vtransform']
    ncfile['Vstretching'][:]=MyVar['Vstretching']
    ncfile['tstart'][:]=0
    ncfile['tend'][:]=0
    ncfile['theta_s'][:]=MyVar['Theta_s']
    ncfile['theta_b'][:]=MyVar['Theta_b']
    ncfile['Tcline'][:]=MyVar['Tcline']
    ncfile['hc'][:]=MyVar['Tcline']
    ncfile['sc_r'][:]=sc_r
    ncfile['Cs_r'][:]=Cs_r
    ncfile['sc_w'][:]=sc_w
    ncfile['Cs_w'][:]=Cs_w
    ncfile['bry_time'][:]=Bry_time
    ncfile['u_north'][:]=0
    ncfile['v_north'][:]=0
    ncfile['zeta_north'][:]=0
    ncfile['ubar_north'][:]=0
    ncfile['vbar_north'][:]=0
    ncfile['temp_north'][:]=0
    ncfile['salt_north'][:]=0

    ncfile['aice_north'][:]=0
    ncfile['hice_north'][:]=0
    ncfile['uice_north'][:]=0
    ncfile['vice_north'][:]=0

    ncfile.close()
    
    

def create_ini_WOA(My_Ini,mask,topo,MyVar,Title,ncFormat='NETCDF3_CLASSIC'):
    hmin_=np.min(topo[mask==1]);
    if MyVar['Vtransform']==1 and MyVar['Tcline']>hmin_:
        raise
        
    Mp,Lp=topo.shape
    L,M,Np=Lp-1,Mp-1,MyVar['Layer_N']+1
    
    My_type='INITIAL file'
    history='ROMS'
    
    ncfile = Dataset(My_Ini,mode='w',format=ncFormat)

    ncfile.createDimension('xi_u',L)
    ncfile.createDimension('xi_v',Lp)
    ncfile.createDimension('xi_rho',Lp)
    ncfile.createDimension('eta_u',Mp)
    ncfile.createDimension('eta_v',M)
    ncfile.createDimension('eta_rho',Mp)
    ncfile.createDimension('s_rho',MyVar['Layer_N'])
    ncfile.createDimension('s_w',Np)
    ncfile.createDimension('tracer',2)
    ncfile.createDimension('one',1)
    ncfile.createDimension('ocean_time',1)

    ncfile.createVariable('spherical', 'S1', ('one'))
    ncfile.createVariable('Vtransform', 'f4', ('one'))
    ncfile.createVariable('Vstretching', 'f4', ('one'))
    ncfile.createVariable('tstart', 'f4', ('one'))
    ncfile.createVariable('tend', 'f4', ('one'))
    ncfile.createVariable('theta_s', 'f4', ('one'))
    ncfile.createVariable('theta_b', 'f4', ('one'))
    ncfile.createVariable('Tcline', 'f4', ('one'))
    ncfile.createVariable('hc', 'f4', ('one'))

    ncfile.createVariable('sc_r', 'f4', ('s_rho'))
    ncfile.createVariable('Cs_r', 'f4', ('s_rho'))
    ncfile.createVariable('sc_w', 'f4', ('s_w'))
    ncfile.createVariable('Cs_w', 'f4', ('s_w'))
    
    ncfile.createVariable('ocean_time', 'f4', ('ocean_time'))

    ncfile.createVariable('u', 'f4', ('ocean_time','s_rho','eta_u','xi_u'))
    ncfile.createVariable('v', 'f4', ('ocean_time','s_rho','eta_v','xi_v'))
    ncfile.createVariable('ubar', 'f4', ('ocean_time','eta_u','xi_u'))
    ncfile.createVariable('vbar', 'f4', ('ocean_time','eta_v','xi_v'))
    ncfile.createVariable('zeta', 'f4', ('ocean_time','eta_rho','xi_rho'))
    ncfile.createVariable('temp', 'f4', ('ocean_time','s_rho','eta_rho','xi_rho'))
    ncfile.createVariable('salt', 'f4', ('ocean_time','s_rho','eta_rho','xi_rho'))

    ncfile.createVariable('uice', 'f4', ('ocean_time','eta_rho','xi_rho'))
    ncfile.createVariable('vice', 'f4', ('ocean_time','eta_rho','xi_rho'))
    ncfile.createVariable('aice', 'f4', ('ocean_time','eta_rho','xi_rho'))
    ncfile.createVariable('hice', 'f4', ('ocean_time','eta_rho','xi_rho'))

    ncfile['Vtransform'].long_name='vertical terrain-following transformation equation'
    ncfile['Vstretching'].long_name='vertical terrain-following stretching function'

    ncfile['tstart'].long_name='start processing day'
    ncfile['tstart'].units='day'

    ncfile['tend'].long_name='end processing day'
    ncfile['tend'].units='day'

    ncfile['theta_s'].long_name='S-coordinate surface control parameter'
    ncfile['theta_s'].units='nondimensional'
    ncfile['theta_b'].long_name='S-coordinate bottom control parameter'
    ncfile['theta_b'].units='nondimensional'


    ncfile['Tcline'].long_name='S-coordinate surface/bottom layer width'
    ncfile['Tcline'].units='meter'
    ncfile['hc'].long_name='S-coordinate parameter, critical depth'
    ncfile['hc'].units='meter'


    ncfile['sc_r'].long_name='S-coordinate at RHO-points'
    ncfile['sc_r'].units='nondimensional'
    
    ncfile['Cs_r'].long_name='S-coordinate stretching curves at RHO-points'
    ncfile['Cs_r'].units='nondimensional'
    
    ncfile['sc_w'].long_name='S-coordinate at W-points'
    ncfile['sc_w'].units='nondimensional'
    
    ncfile['Cs_w'].long_name='S-coordinate stretching curves at W-points'
    ncfile['Cs_w'].units='nondimensional'

    ncfile['u'].long_name='u-momentum component'
    ncfile['u'].units='meter second-1'
    ncfile['v'].long_name='v-momentum component'
    ncfile['v'].units='meter second-1'

    ncfile['ubar'].long_name='vertically integrated u-momentum component'
    ncfile['ubar'].units='meter second-1'
    ncfile['vbar'].long_name='vertically integrated v-momentum component'
    ncfile['vbar'].units='meter second-1'

    ncfile['zeta'].long_name='free-surface'
    ncfile['zeta'].units='meter'

    ncfile['temp'].long_name='potential temperature'
    ncfile['temp'].units='Celsius'
    ncfile['salt'].long_name='salinity'
    ncfile['salt'].units='PSU'

    ncfile['uice'].long_name='u-component of ice velocity'
    ncfile['uice'].units='meter second-1'
    ncfile['vice'].long_name='v-component of ice velocity'
    ncfile['vice'].units='meter second-1'
    ncfile['aice'].long_name='fraction of cell covered by ice'
    ncfile['aice'].units='nondimensional'
    ncfile['hice'].long_name='average ice thickness in cell'
    ncfile['hice'].units='meter'

    ncfile.title=Title
    ncfile.clim_file=My_Ini
    ncfile.grd_file=''
    ncfile.type=My_type
    ncfile.history=history
    
    sc_r,Cs_r=jr.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],0)

    sc_w,Cs_w=jr.stretching(MyVar['Vstretching'],MyVar['Theta_s'],MyVar['Theta_b'],\
                         MyVar['Layer_N'],1)

    ncfile['spherical'][:]='T'
    ncfile['Vtransform'][:]=MyVar['Vtransform']
    ncfile['Vstretching'][:]=MyVar['Vstretching']
    ncfile['tstart'][:]=0
    ncfile['tend'][:]=0
    ncfile['theta_s'][:]=MyVar['Theta_s']
    ncfile['theta_b'][:]=MyVar['Theta_b']
    ncfile['Tcline'][:]=MyVar['Tcline']
    ncfile['hc'][:]=MyVar['Tcline']
    ncfile['sc_r'][:]=sc_r
    ncfile['Cs_r'][:]=Cs_r
    ncfile['sc_w'][:]=sc_w
    ncfile['Cs_w'][:]=Cs_w
    ncfile['ocean_time'][:]=0
    ncfile['u'][:]=0
    ncfile['v'][:]=0
    ncfile['zeta'][:]=0
    ncfile['ubar'][:]=0
    ncfile['vbar'][:]=0
    ncfile['temp'][:]=0
    ncfile['salt'][:]=0

    ncfile['aice'][:]=0
    ncfile['hice'][:]=0
    ncfile['uice'][:]=0
    ncfile['vice'][:]=0

    ncfile.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
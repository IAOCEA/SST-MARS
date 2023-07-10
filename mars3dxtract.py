import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import sys

tag='/home/datawork-lops-iaocea/data/fish-intel/tag/nc/A18832.nc'

t0=datetime(2022,6,17,0,0,0)
t1=datetime(2022,6,17,11,1,0)

nside=256

f = netCDF4.Dataset(tag)

time = f.variables['time'][:].data #seconds since 2022-06-17 00:00:00
p    = f.variables['pressure'][:].data
t    = f.variables['water_temperature'][:].data

lat = f.variables['latitude'][:].data
lon = f.variables['longitude'][:].data
tim = f.variables['times'][:].data #minutes since 2022-06-17 11:01:00
f.close()


print((t0+timedelta(seconds=int(time[0]))).ctime(), \
       (t0+timedelta(seconds=int(time[-1]))).ctime())

print((t1+timedelta(seconds=60*int(tim[0]))).ctime(), \
       (t1+timedelta(seconds=60*int(tim[-2]))).ctime())

#read corresponding marc data
path='/dataref/ref3/public/modeles_marc/f1_e2500/best_estimate/'

nh=256 #int((tim[1]-tim[0])//60)+1

ii=0
dt=t1+timedelta(hours=ii)
deltas=(dt-t0).seconds
tidx=np.where((time>=deltas)*(time<deltas+3600))[0]

print(tidx)

name=path+'%d/MARC_F1-MARS3D-MANGAE2500_%d%02d%02dT%02d00Z.nc'%(dt.year,dt.year,dt.month,dt.day,dt.hour)

f = netCDF4.Dataset(name)
TEMP=f.variables['TEMP'][0,:,:,:].data
LON=f.variables['longitude'][:,:].data
LAT=f.variables['latitude'][:,:].data
H0=f.variables['H0'][:,:].data
XE=f.variables['XE'][:,:,:].data
SIG=f.variables['Csu_sig'][:].data
f.close()

ISIG=np.zeros([10000],dtype='int')
ISIG[(-(SIG[1:]+SIG[:-1])/2*10000).astype('int')]=1
ISIG=39-np.cumsum(ISIG)

lon0=(lon[1]+lon[0])/2
lat0=(lat[1]+lat[0])/2

pidx=np.argmin((LON.flatten()-lon0)**2+(LAT.flatten()-lat0)**2)
xc=pidx//H0.shape[1]
yc=pidx%H0.shape[1]

lon0=(lon[0])
lat0=(lat[0])

pidx=np.argmin((LON.flatten()-lon0)**2+(LAT.flatten()-lat0)**2)
x1=pidx//H0.shape[1]
y1=pidx%H0.shape[1]

lon0=(lon[1])
lat0=(lat[1])

pidx=np.argmin((LON.flatten()-lon0)**2+(LAT.flatten()-lat0)**2)
x2=pidx//H0.shape[1]
y2=pidx%H0.shape[1]

ni,nj=H0.shape
sigma=1.0
nd=TEMP.shape[0]
H0=H0[xc-nside//2:xc+nside//2,yc-nside//2:yc+nside//2]
res=np.zeros([nh,nd,nside,nside])
pres=np.zeros([nh,nside,nside])
tres=np.zeros([nh,nside,nside])

x2=nside//2+(x2-xc)
y2=nside//2+(y2-yc)
x1=nside//2+(x1-xc)
y1=nside//2+(y1-yc)

pathsst='/home/ref-cersat-public/sea-surface-temperature/odyssea/l3s/atl/nrt/data/v3.0/'
for ii in range(nh):
    dt=t1+timedelta(hours=ii)
    daynum=(dt-datetime(dt.year,1,1,0,0)).days+1
    deltas=(dt-t0).total_seconds()

    name=pathsst+'%d/%d/%d%02d%02d000000-IFR-L3S_GHRSST-SSTfnd-ODYSSEA-ATL_002-v02.1-fv01.0.nc'%(dt.year,daynum,dt.year,dt.month,dt.day)

    f = netCDF4.Dataset(name)
    tlat=f.variables['lat'][:].data
    tlon=f.variables['lon'][:].data
    sst=f.variables['sea_surface_temperature'][0,:,:].data
    qua=f.variables['quality_level'][0,:,:].data
    f.close()

    name=path+'%d/MARC_F1-MARS3D-MANGAE2500_%d%02d%02dT%02d00Z.nc'%(dt.year,dt.year,dt.month,dt.day,dt.hour)

    f = netCDF4.Dataset(name)
    TEMP=f.variables['TEMP'][0,:,xc-nside//2:xc+nside//2,yc-nside//2:yc+nside//2].data
    XE=f.variables['XE'][:,xc-nside//2:xc+nside//2,yc-nside//2:yc+nside//2].data
    f.close()
    
    sst[qua!=5]=-32768.0
    
    sst_idx_lon=(tlon.shape[0]*(LON.flatten()-tlon.min())/(tlon.max()-tlon.min())).astype('int')
    sst_idx_lat=(tlat.shape[0]*(LAT.flatten()-tlat.min())/(tlat.max()-tlat.min())).astype('int')

    lsst=sst[sst_idx_lat,sst_idx_lon].reshape(LON.shape[0],LON.shape[1])[xc-nside//2:xc+nside//2,yc-nside//2:yc+nside//2]-273
    """
    print(lat.shape,lon.shape,sst.shape,qua.shape,LON.shape,LAT.shape)
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(,origin='lower',vmin=9,vmax=19)
    plt.subplot(1,2,2)
    plt.imshow(TEMP[-1],origin='lower',vmin=9,vmax=19)
    plt.show()
    """
    PRES=(H0[:,:]+XE[0,:,:])
    PRES[(PRES<0)]=-1.0

    res[ii,:,:,:]=TEMP
    pres[ii,:,:]=PRES
    tres[ii,:,:]=lsst
    print(dt.year,dt.month,dt.day,dt.hour,ii,nh)
    sys.stdout.flush()
    #plt.figure(figsize=(16,6))
    #plt.subplot(1,4,1)
    #plt.imshow(res[ii],cmap='jet',origin='lower')
    #plt.subplot(1,4,2)
    #plt.plot(p[tidx])
    #plt.subplot(1,4,3)
    #plt.plot(t[tidx])
    #plt.subplot(1,4,4)
    #plt.imshow(tabn,cmap='jet',origin='lower')
    #plt.show()

np.save('/home1/scratch/jmdeloui/TAOS/mars3dT.npy',res)
np.save('/home1/scratch/jmdeloui/TAOS/mars3dP.npy',pres)
np.save('/home1/scratch/jmdeloui/TAOS/mars3dSST.npy',tres)
exit(0)

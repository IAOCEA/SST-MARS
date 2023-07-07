import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from datetime import datetime, timedelta

tag='/home/datawork-lops-iaocea/data/fish-intel/tag/nc/A18832.nc'

t0=datetime(2022,6,17,0,0,0)
t1=datetime(2022,6,17,11,1,0)

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

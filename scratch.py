import numpy as np
import xarray as xr

a=xr.open_dataset('Seattle_colocated_20220701_20221231.nc')

print(np.nanmax(a['xco']))
print(a.where(a['xco']==np.nanmax(a['xco']),drop=True).day)

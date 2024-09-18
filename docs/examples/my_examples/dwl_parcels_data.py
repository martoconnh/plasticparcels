import numpy as np
import pandas as pd
import xarray as xr
import requests as rq
import os
import time
from argparse import ArgumentParser


def download_moi(var):
    if var in ['I']:
        vname = 'icemod'
    else:
        vname = f"grid{var}"
    datapgn = xr.open_dataset(f"http://tds.mercator-ocean.fr/thredds/dodsC/psy4v3r1-daily-{vname}")
    print(f"loaded datastore {vname}")

    for i in range(0, 6000):
        fname = "../psy4v3r1-daily_%s_%s.nc" % (var, str(datapgn.isel(time_counter=i).time_counter.values)[:10])
        if os.path.isfile(fname):
            print("%s exists; skipping" %fname)
        else:
            for attempt in range(50):
                try:
                    if var in ['U']:
                        visc = datapgn.drop(labels=["vozocrtx", "deptht"]).isel(time_counter=i)
                        upper_vel = datapgn.vozocrtx.isel(deptht=slice(0,25), time_counter=i)
                        lower_vel = datapgn.vozocrtx.isel(deptht=slice(25,50), time_counter=i)
                        total_vel = xr.concat([upper_vel, lower_vel], dim='deptht')
                        total = xr.merge([visc, total_vel])
                    elif var in ['W', 'KZ']:
                        upper = datapgn.isel(depthw=slice(0,25), time_counter=i)
                        lower = datapgn.isel(depthw=slice(25,50), time_counter=i)
                        total = xr.concat([upper,lower], dim='depthw')
                    elif var in ['I']:  # 2D fields only
                        total = datapgn.isel(time_counter=i)
                    else:
                        upper = datapgn.isel(deptht=slice(0,25), time_counter=i)
                        lower = datapgn.isel(deptht=slice(25,50), time_counter=i)
                        total = xr.concat([upper,lower], dim='deptht')
                    print('writing %s' %fname)
                    total.to_netcdf(fname)
                    print('writing done')
                except:
                    print('failed download %d for %s' % (attempt+1, var))
                    time.sleep(60*(attempt+1))
                    continue
                else:
                    break
            else:
                raise RuntimeError('%s could not be loaded' %fname)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('-var')
    args = p.parse_args()
    download_moi(args.var)
 
import netCDF4
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def day_to_hour(day):
    start = datetime(2026, 1, 1, 1, 0, 0)
    return int((day - start).total_seconds() // 3600)

def get_days(ds):
    time_var = ds.variables['time']
    times = netCDF4.num2date(time_var[:], units=time_var.units)

    if len(times) < 365 * 10 * 24:
        print(f"WARNING: Time variable has only {len(times)} points, expected {365 * 10 * 24}. Check dataset.")
    
    return np.array(times)

def interp_years(data, ds, years):
    days = get_days(ds)
    
    if len(data) < years * 365 * 24:
        print(f"WARNING: Dataset is shorter than expected ({len(data)} < {years * 365 * 24})")

    xp = np.arange(len(data))  # Original data points
    x = np.linspace(0, len(data) - 1, years * 365 * 24)  # Target time points

    f = np.interp(x, xp, data)

    print(f"DEBUG: Interpolation | Raw Data Sample: {data[:10]}")
    print(f"DEBUG: Interpolation | Interpolated Sample: {f[:10]}")

    return f

def get_atm_forcing(path, s, lat_ind, lon_ind):
    key = path.split('_')[0].split('\\')[-1].split('/')[-1]  # Extract variable name
    ds = netCDF4.Dataset(path)

    lat = float(ds.variables['lat'][lat_ind])
    lon = float(ds.variables['lon'][lon_ind])
    
    time_len = len(ds.variables['time'])
    s = slice(0, min(time_len, 365 * 10 * 24))  # Ensure safe slicing

    _forcing = ds.variables[key][s, lat_ind, lon_ind]
    forcing = interp_years(_forcing, ds, 10)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(forcing)
    axs[0].set_title(f"Interpolated {key}")
    axs[1].plot(_forcing)
    axs[1].set_title(f"Original {key}")

    print(f"LOADED: {key} | Units: {ds.variables[key].units}")
    ds.close()
    
    return key, forcing, lat, lon

def make_atm_forcing(paths, lat_ind, lon_ind):
    atm = {}
    for path in paths:
        key, forcing, lat, lon = get_atm_forcing(path, slice(None), lat_ind, lon_ind)
        atm[key] = forcing

    df = pd.DataFrame(atm)
    output_file = f"atm_{int(lat)}_{int(lon)}_10years.txt"
    df.to_csv(output_file, sep=' ', header=None, index=False, float_format='%.6f')

    print(f"FILE SAVED: {output_file}")
    return df

def get_ocn_forcing(path, s, ind1, ind2):
    key = path.split('_')[0].split('\\')[-1].split('/')[-1]  # Extract variable name
    ds = netCDF4.Dataset(path)

    if ds.variables['lat'].ndim == 2:
        lat = float(ds.variables['lat'][ind1, ind2])
        lon = float(ds.variables['lon'][ind1, ind2])
    elif ds.variables['lat'].ndim == 1:
        lat = float(ds.variables['lat'][ind1])
        lon = float(ds.variables['lon'][ind2])
    else:
        raise ValueError(f"ERROR: Unexpected lat/lon dimensions: {ds.variables['lat'].shape}")

    data = ds.variables[key][:]

    if data.ndim == 4:
        forcing = data[s, 0, ind1, ind2]
    elif data.ndim == 3:
        forcing = data[s, ind1, ind2]
    else:
        raise ValueError(f"ERROR: Unexpected data dimensions for {key}: {data.shape}")

    forcing = interp_years(forcing, ds, 10)

    print(f"LOADED: {key} | Units: {ds.variables[key].units}")
    ds.close()
    
    return key, forcing, lat, lon

def make_ocn_forcing(paths, ind1, ind2):
    ocn = {}
    for path in paths:
        key, forcing, lat, lon = get_ocn_forcing(path, slice(None), ind1, ind2)
        ocn[key] = forcing

    df = pd.DataFrame(ocn)
    output_file = f"ocn_{int(lat)}_{int(lon)}_10years.txt"
    df.to_csv(output_file, sep=' ', header=None, index=False, float_format='%.6f')

    print(f"FILE SAVED: {output_file}")
    return df

if __name__ == "__main__":
    forcing = make_atm_forcing(
        paths=[
            # r'C:\\Users\\Lee\\Desktop\\Data\\rsds_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\rlds_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\uas_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\vas_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\tas_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\huss_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\pr_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\rsds_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\rlds_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\uas_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\vas_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\tas_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\huss_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\pr_day_CNRM-CM5_rcp45_r1i1p1_20160101-20201231.nc',            
        ],
        lat_ind=83,
        lon_ind=347,
    )
    
    ocn_forcing = make_ocn_forcing(
        paths=[
            # r'C:\\Users\\Lee\\Desktop\\Data\\tos_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\sos_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\mlotst_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\uo_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
            # r'C:\\Users\\Lee\\Desktop\\Data\\vo_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\tos_Omon_CNRM-CM5_rcp45_r1i1p1_201601-202512.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\sos_Omon_CNRM-CM5_rcp45_r1i1p1_201601-202512.nc',            
            r'C:\\Users\\Lee\\Desktop\\thesis\\mlotst_Omon_CNRM-CM5_rcp45_r1i1p1_201601-202512.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\uo_Omon_CNRM-CM5_rcp45_r1i1p1_201601-202512.nc',
            r'C:\\Users\\Lee\\Desktop\\thesis\\vo_Omon_CNRM-CM5_rcp45_r1i1p1_201601-202512.nc',                                    
        ],
        ind1=276,
        ind2=267,
    )
a

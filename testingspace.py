#Intializations
import netCDF4
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen, GoogleTiles
import xarray as xr
import os
import subprocess
import re

in_file_path = r'\\wsl.localhost\Ubuntu\home\leeeee05\Icepack-vargas_thesis_2018\perrycase\icepack_in'


with open(in_file_path, 'r') as file:
            file_contents = file.read()
            
# Pre's from Vargas New_Data
def day_to_hour(day):
    start = datetime(2021, 1, 1, 1, 0, 0)
    print(11)
    return int((day - start).total_seconds() // 3600)
    

def get_days(ds):
    start = datetime(2006, 1, 1)
    min_date = datetime(2021, 1, 1, 0, 0, 0)
    max_date = datetime(2031, 1, 1, 0, 0, 0)
    times = [start + timedelta(days=d) for d in ds['time'][:]]
    times = list(filter(lambda d: min_date <= d < max_date, times))
    print(12)
    return times

def interp_years(data, ds, years):
    days = get_days(ds)
    xp = np.array([day_to_hour(day) for day in days])
    x = np.arange(1, years * 365 * 24 + 1)
    f = np.interp(x, xp, data)
    print(13)
    return f

def get_atm_forcing(path, s, lat_ind, lon_ind):
    key = path.split('_')[0] 
    print(f'{key}')
    key = key.split('\\')[-1] # Change 1
    key = key.split('/')[-1]
    print(f'{key}')
    ds = netCDF4.Dataset(path)
    lat = ds.variables['lat'][:].data[lat_ind]
    lon = ds.variables['lon'][:].data[lon_ind]
    _forcing = ds.variables[key][s, lat_ind, lon_ind].data
    forcing = interp_years(_forcing, ds, 10)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax.plot(forcing)
    ax.set_title(key)
    ax = fig.add_subplot(122)
    ax.plot(_forcing)
    ax.set_title(key)
    print(f'{key} : {ds.variables[key].units}')
    # forcing = forcing.round(5)
    ds.close()
    print(14)
    return key, forcing, lat, lon

def make_atm_forcing(paths, lat_ind, lon_ind):
    # Slice two years 2016-2020
    s = slice(0, 365 * 10 + 4)
    atm = {}
    for path in paths:
        key, forcing, lat, lon = get_atm_forcing(path, s, lat_ind, lon_ind)
        atm[key] = forcing
    # print(atm['rsds'])
    df = pd.DataFrame(atm, columns=list(atm.keys()))
    df.to_csv(f'atm_{lat:.0f}_{lon:.0f}_10years.txt', sep=' ', header=None, index=False, float_format='%.6f')
    print(15)
    return df

def get_ocn_forcing(path, s, ind1, ind2):
    key = path.split('_')[0]
    key = key.split('\\')[-1]  # Handle Windows paths
    key = key.split('/')[-1]  # Handle Unix paths
    with netCDF4.Dataset(path) as ds:
        # Extract latitude and longitude
        lat = ds.variables['lat'][:].data[ind1, ind2]
        lon = ds.variables['lon'][:].data[ind1, ind2]
        data = ds.variables[key][:].data

        # Check data dimensions
        if data.ndim == 4:
            forcing = data[s, 0, ind1, ind2]
        elif data.ndim == 3:
            forcing = data[s, ind1, ind2]
        else:
            raise ValueError(f"Unexpected data dimensions: {data.ndim}D")

        # Interpolate for 10 years
        forcing = interp_years(forcing, ds, 10)

        # Debugging
        print(f"Processed variable: {key}, Shape: {data.shape}")
        print(f"Original forcing data: {forcing[:20]} ... {forcing[-20:]}")

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(forcing)
        ax.set_title(key)
        print(f'{key} : {ds.variables[key].units}')
        print(16)
        return key, forcing, lat, lon


def make_ocn_forcing(paths, ind1, ind2):
    s = slice(0, 12 * 10)  # Adjust based on dataset time steps
    ocn = {}
    print(17)
    for path in paths:
        try:
            key, forcing, lat, lon = get_ocn_forcing(path, s, ind1, ind2)
            ocn[key] = forcing
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    if ocn:
        df = pd.DataFrame(ocn, columns=list(ocn.keys()))
        df.to_csv(f'ocn_{lat:.0f}_{lon:.0f}_10years.txt', sep=' ', header=None, index=False,
                  float_format='%.6f')
        return df
    else:
        print("No data processed.")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#create a matrix of all the values I want here

lat = [123]
lon = [247]
ind1 = [276]
ind2= [267]
n = 0
c = 0

atm_file = f'atm_{lat[n]:.1f}_{lon[n]:.1f}_10years.txt'
print(1)
ocn_file = f'ocn_{lat[n]:.1f}_{lon[n]:.1f}_10years.txt'
bgc_file = f'bgc_{lat[n]:.1f}_{lon[n]:.1f}_10years.txt'

input_file_path = r'\\wsl.localhost\Ubuntu\home\leeeee05\Icepack-vargas_thesis_2018\perrycase\icepack_in'
print(2)
replacement_text = "control.txt" if c == 0 else "pumping.txt"

for n in range(len(lat)):
    for c in [0, 1]: 
        print(f"Processing lat={lat[n]}, lon={lon[n]}, scenario={'control' if c == 0 else 'pumping'}")

        # Read the file and replace the text
        with open(input_file_path, "r") as file:
            content = file.read()
        print(3)
        # Replace the existing file name with the new one
        content = content.replace("pumping.txt", replacement_text).replace("control.txt", replacement_text)

        # Write the modified content back to the file
        with open(input_file_path, "w") as file:
            file.write(content)

        print(f"File updated successfully to use {replacement_text}.")
        
        forcing = make_atm_forcing(
            paths=[
                r'C:\Users\Lee\Desktop\Data\rsds_data.nc',
                r'C:\Users\Lee\Desktop\Data\rlds_data.nc',
                r'C:\Users\Lee\Desktop\Data\uas_data.nc',
                r'C:\Users\Lee\Desktop\Data\vas_data.nc',
                r'C:\Users\Lee\Desktop\Data\tas_data.nc',
                r'C:\Users\Lee\Desktop\Data\huss_data.nc',
                r'C:\Users\Lee\Desktop\Data\pr_data.nc',
            ],
            lat_ind=lat[n],
            lon_ind=lon[n],
        )
    
        ocn_forcing = make_ocn_forcing(
            paths=[
                r'C:\Users\Lee\Desktop\Data\tos_data.nc',
                r'C:\Users\Lee\Desktop\Data\sos_data.nc',
                r'C:\Users\Lee\Desktop\Data\mlotst_data.nc',
                r'C:\Users\Lee\Desktop\Data\uo_data.nc',
                r'C:\Users\Lee\Desktop\Data\vo_data.nc',
            ],
            ind1=ind1[n],
            ind2=ind2[n],
        )
        
        
        # Open and modify the file
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        
        # Set up your file replacements for c
        if c == 0:
            pump_data_file = f"control.txt" 
            print(4)
        else:
            pump_data_file = f"pumping.txt"

        figmap = plt.figure()  # Create a new figure for the map plot

        # Set up the polar projection
        ax = figmap.add_subplot(111, projection=ccrs.NorthPolarStereo())

        # Use a valid extent and projection
        try:
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        except Exception as e:
            print(f"Error setting extent: {e}")

        # Display map
        ax.coastlines()

        # Show the plot
        plt.show()
        print(5)
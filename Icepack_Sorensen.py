#Intializations
import netCDF4
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import Stamen, GoogleTiles
import xarray as xr
import os
import subprocess

#def functions
def day_to_hour(day):
    start = datetime(2021, 1, 1, 1, 0, 0)
    return int((day - start).total_seconds() // 3600)

def get_days(ds):
    start = datetime(2006, 1, 1)
    min_date = datetime(2021, 1, 1, 0, 0, 0)
    max_date = datetime(2031, 1, 1, 0, 0, 0)
    times = [start + timedelta(days=d) for d in ds['time'][:]]
    times = list(filter(lambda d: min_date <= d < max_date, times))
    return times

def interp_years(data, ds, years):
    days = get_days(ds)
    xp = np.array([day_to_hour(day) for day in days])
    x = np.arange(1, years * 365 * 24 + 1)
    f = np.interp(x, xp, data)
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
        return key, forcing, lat, lon


def make_ocn_forcing(paths, ind1, ind2):
    s = slice(0, 12*10)  # Adjust based on dataset time steps
    ocn = {}
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


def get_bgc_forcing(path, s, ind1, ind2):
    key = path.split('_')[0]
    key = key.split('\\')[-1]
    key = key.split("/")[-1]
    ds = netCDF4.Dataset(path)
    lat = ds.variables['lat'][:].data[ind1, ind2]
    lon = ds.variables['lon'][:].data[ind1, ind2]
    data = ds.variables[key][:].data
    if data.ndim == 4:
        forcing = data[s, 0 , ind1, ind2] * 1000
    else:
        forcing = data[s, ind1, ind2] * 1000
    print(f'{key} : {ds.variables[key].units}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(forcing)
    ax.set_title(key)
    # forcing = forcing.round(5)
    ds.close()
    return key, forcing, lat, lon

def make_bgc_forcing(paths, ind1, ind2):
    s = slice(0, 60)
    ocn = {}
    for path in paths:
        key, forcing, lat, lon = get_bgc_forcing(path, s, ind1, ind2)
        ocn[key] = forcing
    df = pd.DataFrame(ocn, columns=list(ocn.keys()))
    df.to_csv(f'bgc_{lat:.0f}_{lon:.0f}_5years.txt', sep=' ', header=None, index=False,
float_format='%.6f')
    return df

#Part 1

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
    lat_ind=123,
    lon_ind=247,
)

ocn_forcing = make_ocn_forcing(
    paths=[
        r'C:\Users\Lee\Desktop\Data\tos_data.nc',
        r'C:\Users\Lee\Desktop\Data\sos_data.nc',
        r'C:\Users\Lee\Desktop\Data\mlotst_data.nc',
        r'C:\Users\Lee\Desktop\Data\uo_data.nc',
        r'C:\Users\Lee\Desktop\Data\vo_data.nc',
    ],
    ind1=276,
    ind2=267,
)

# Create a figure and add an axes with a specified projection
fig, ax = plt.subplots(1, 1, figsize=(15, 6),
                       subplot_kw={'projection': ccrs.NorthPolarStereo()})

# Add a natural earth feature (like the bluemarble equivalent)
ax.stock_img()

# Plot the point (longitude, latitude)
ax.plot(348, 83, 'ro', transform=ccrs.Geodetic())

# Set the extent (bounding box) of the map in degrees (lon_min, lon_max, lat_min, lat_max)
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())

# Add coastlines and other features
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.BORDERS, linestyle=':')

plt.show()

in_file_path = r'\\wsl.localhost\Ubuntu\home\leeeee05\Icepack-vargas_thesis_2018\perrycase\icepack_in'

#Location 1: Control

modified_atm_contents = file_contents.replace("atm_83_347_5years.txt", "atm...")
modified_ocn_contents = file_contents.replace("ocn_83_348_5years.txt", "ocn...")
modified_bgc_contents = file_contents.replace("bgc_83_348_5years.txt", "bgc...")

    
with open(in_file_path, 'r') as file:
    file_contents = file.read()
            
with open(input_file_path, 'w') as file:
    file.write(modified_atm_contents)
    file.write(modified_ocn_contents)
    file.write(modified_bgc_contents)
    
subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.build"', shell=True)
subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.submit"', shell=True)

result = subprocess.run("wsl cat ~/icepack-dirs/runs/perrycase/ice_diag.full_ITD", shell=True, capture_output=True, text=True)

new_filename = "....txt"

# Save the output to the new file
with open(new_filename, "w") as file:
    file.write(result.stdout)

# Print a confirmation message
print(f"Contents saved to {new_filename}")

#Location 1: Pumping

modified_atm_contents = file_contents.replace("atm_83_347_5years.txt", "atm...")
modified_ocn_contents = file_contents.replace("ocn_83_348_5years.txt", "ocn...")
modified_bgc_contents = file_contents.replace("bgc_83_348_5years.txt", "pumping.txt")

    
with open(in_file_path, 'r') as file:
    file_contents = file.read()
            
with open(input_file_path, 'w') as file:
    file.write(modified_atm_contents)
    file.write(modified_ocn_contents)
    file.write(modified_bgc_contents)
    
subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.build"', shell=True)
subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.submit"', shell=True)

result = subprocess.run("wsl cat ~/icepack-dirs/runs/perrycase/ice_diag.full_ITD", shell=True, capture_output=True, text=True)

new_filename = "....txt"

# Save the output to the new file
with open(new_filename, "w") as file:
    file.write(result.stdout)

# Print a confirmation message
print(f"Contents saved to {new_filename}")




#repeat etc



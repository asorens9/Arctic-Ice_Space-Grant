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
    s = slice(0, 12 * 10)  # Adjust based on dataset time steps
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
    df.to_csv(f'bgc_{lat:.0f}_{lon:.0f}_10years.txt', sep=' ', header=None, index=False,
float_format='%.6f')
    return df
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#create a matrix of all the values I want here

lat = [123]
lon = [247]
ind1 = [276]
ind2= [267]
n = 0
c = 0

atm_file = f'atm_{lat[n]:.1f}_{lon[n]:.1f}_10years.txt'
ocn_file = f'ocn_{lat[n]:.1f}_{lon[n]:.1f}_10years.txt'
bgc_file = f'bgc_{lat[n]:.1f}_{lon[n]:.1f}_10years.txt'

input_file_path = r'\\wsl.localhost\Ubuntu\home\leeeee05\Icepack-vargas_thesis_2018\perrycase\icepack_in'

replacement_text = "control.txt" if c == 0 else "pumping.txt"

for n in range(len(lat)):
    for c in [0, 1]: 
        print(f"Processing lat={lat[n]}, lon={lon[n]}, scenario={'control' if c == 0 else 'pumping'}")

        # Read the file and replace the text
        with open(input_file_path, "r") as file:
            content = file.read()

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
        else:
            pump_data_file = f"pumping.txt"
    
        # Set up the polar projection
        ax = plt.axes(projection=ccrs.NorthPolarStereo())

        #   Use a    valid extent and projection
        try:
            ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        except Exception as e:
            print(f"Error setting extent: {e}")

        # Display map
        ax.coastlines()
        plt.show()
        
        #running control
        new_modified_atm_contents = f'atm_{lat[n]}_{lon[n]}_10years.txt'
        print(f"{new_modified_atm_contents}")
        new_modified_ocn_contents = f'ocn_{lat[n]}_{lon[n]}_10years.txt'
        new_modified_bgc_contents = f'bgc_{lat[n]}_{lon[n]}_10years.txt'

        modified_atm_contents = file_contents.replace(atm_file, "new_modified_atm_contents")
        modified_ocn_contents = file_contents.replace(ocn_file, "new_modified_ocn_contents")
        modified_bgc_contents = file_contents.replace(bgc_file, "new_modified_bgc_contents")
        
        
                
        with open(input_file_path, 'w') as file:
            file.write(modified_atm_contents)
            file.write(modified_ocn_contents)
            file.write(modified_bgc_contents)
        
        subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.build"', shell=True)
        subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.submit"', shell=True)

        result = subprocess.run("wsl cat ~/icepack-dirs/runs/perrycase/ice_diag.full_ITD", shell=True, capture_output=True, text=True)

        if c == 0:
            type = str(f'control')
        else:
            type = str(f'test')

        new_filename = f"{lat[n]}_{lon[n]}_{type}.txt"

        # Save the output to the new file
        with open(new_filename, "w") as file:
            file.write(result.stdout)

        # Print a confirmation message
        print(f"Contents saved to {new_filename}")
        
        atm_file = new_modified_atm_contents
        ocn_file = new_modified_ocn_contents
        bgc_file = new_modified_bgc_contents

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        #Plotting results
        
        # Initialize lists to store the data
        time_steps = []
        air_temp = []
        ice_area_fraction = []
        top_melt = []
        bottom_melt = []
        lateral_melt = []
        ice_thickness = []

        def extract_value(line, label):
            """
            Extracts a numerical value associated with a given label from a line of text.

            Args:
                line (str): The input line from the file.
                label (str): The label to search for in the line.

            Returns:
                float: The extracted numerical value, or None if not found.
            """
            # Use a regex pattern to match "label = value" format
            pattern = rf"{re.escape(label)}\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
            match = re.search(pattern, line)
            return float(match.group(1)) if match else None

        # Predefined labels for consistency
        LABELS = {
            "air_temp": "air temperature (C)",
            "area_fraction": "area fraction",
            "top_melt": "top melt (m)",
            "bottom_melt": "bottom melt (m)",
            "lateral_melt": "lateral melt (m)",
            "ice_thickness": "ice thickness (m)"
        }

        # Read and parse the file
        with open(f"{new_filename}", "r") as file:
            current_time_step = None
            
            for line in file:
                # Extract time step
                if "istep1:" in line:
                    match = re.search(r"istep1:\s+(\d+)", line)
                    if match:
                        current_time_step = int(match.group(1))
                        time_steps.append(current_time_step)
                
                # Skip empty lines or unrelated lines
                if not line:
                    continue

                # Extract data based on predefined labels
                if LABELS["air_temp"] in line:
                    temp = extract_value(line, LABELS["air_temp"])
                    air_temp.append(temp if temp is not None else float("nan"))

                if LABELS["area_fraction"] in line and "melt pond" not in line:  # Exclude 'melt pond'
                    area = extract_value(line, LABELS["area_fraction"])
                    ice_area_fraction.append(area if area is not None else float("nan"))

                if LABELS["top_melt"] in line:
                    melt = extract_value(line, LABELS["top_melt"])
                    top_melt.append(melt if melt is not None else float("nan"))

                if LABELS["bottom_melt"] in line:
                    melt = extract_value(line, LABELS["bottom_melt"])
                    bottom_melt.append(melt if melt is not None else float("nan"))

                if LABELS["lateral_melt"] in line:
                    melt = extract_value(line, LABELS["lateral_melt"])
                    lateral_melt.append(melt if melt is not None else float("nan"))

                if LABELS["ice_thickness"] in line:
                    thickness = extract_value(line, LABELS["ice_thickness"])
                    ice_thickness.append(thickness if thickness is not None else float("nan"))

        # Align lengths of all lists
        min_length = min(len(time_steps), len(air_temp), len(ice_area_fraction), len(top_melt), len(bottom_melt), len(lateral_melt), len(ice_thickness))

        time_steps = time_steps[:min_length]
        air_temp = air_temp[:min_length]
        ice_area_fraction = ice_area_fraction[:min_length]
        top_melt = top_melt[:min_length]
        bottom_melt = bottom_melt[:min_length]
        lateral_melt = lateral_melt[:min_length]
        ice_thickness = ice_thickness[:min_length]

        print(f"Length of time_steps: {time_steps[:5]}")
        print(f"Length of air_temp: {air_temp[:5]}")
        print(f"Length of ice_area_fraction: {ice_area_fraction[:5]}")
        print(f"Sample time_steps: {time_steps[:5]}")
        print(f"Sample air_temp: {air_temp[:5]}")

        time_steps_in_days = [step / 24 for step in time_steps]

        # Create a plot for each variable
        plt.figure(figsize=(12, 8))

        # Plot air temperature
        plt.subplot(3, 2, 1)
        plt.plot(time_steps_in_days, air_temp, label="Air Temperature (C)", color="blue")
        plt.xlabel("Time Step (days)")
        plt.ylabel("Air Temp (C)")
        plt.title("Air Temperature")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(time_steps_in_days, ice_area_fraction, label="Ice Area Fraction", color="green")
        plt.xlabel("Time Step (days)")
        plt.ylabel("Area Fraction")
        plt.title("Ice Area Fraction")
        plt.grid(True)


        # Plot ice thickness
        plt.subplot(3, 2, 3)
        plt.plot(time_steps_in_days, ice_thickness, label="Ice Thickness (m)", color="orange")
        plt.xlabel("Time Step (days)")
        plt.ylabel("Thickness (m)")
        plt.title("Ice Thickness")
        plt.grid(True)

        # Plot top melt
        plt.subplot(3, 2, 4)
        plt.plot(time_steps_in_days, top_melt, label="Top Melt (m)", color="red")
        plt.xlabel("Time Step (days)")
        plt.ylabel("Top Melt (m)")
        plt.title("Top Melt")
        plt.grid(True)

        # Plot bottom melt
        plt.subplot(3, 2, 5)
        plt.plot(time_steps_in_days, bottom_melt, label="Bottom Melt (m)", color="purple")
        plt.xlabel("Time Step (days)")
        plt.ylabel("Bottom Melt (m)")
        plt.title("Bottom Melt")
        plt.grid(True)

        # Plot lateral melt
        plt.subplot(3, 2, 6)
        plt.plot(time_steps_in_days, lateral_melt, label="Lateral Melt (m)", color="brown")
        plt.xlabel("Time Step (days)")
        plt.ylabel("Lateral Melt (m)")
        plt.title("Lateral Melt")
        plt.grid(True)


        output_plot_filename = f"{lat[n]}_{lon[n]}_{type}_plots.png"
        plt.savefig(output_plot_filename, dpi=300)
        print(f"Graph saved as {output_plot_filename}")
        
        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()
        


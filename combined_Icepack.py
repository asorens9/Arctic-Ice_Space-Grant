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

def day_to_hour(day):
    start = datetime(2026, 1, 1, 1, 0, 0)
    return int((day - start).total_seconds() // 3600)

def get_days(ds):
    start = datetime(2006, 1, 1)
    min_date = datetime(2026, 1, 1, 0, 0, 0)
    max_date = datetime(2031, 1, 1, 0, 0, 0)
    times = [start + timedelta(days=d) for d in ds['time'][:]]
    # Ensure times and data length match
    if len(times) != ds['time'].shape[0]:
        raise ValueError("Mismatch in time steps length.")
    times = list(filter(lambda d: min_date <= d < max_date, times))
    return times

def get_months(ds):
    from collections import defaultdict

    # Define the range of interest
    start = datetime(2006, 1, 1)
    min_date = datetime(2026, 1, 1, 0, 0, 0)
    max_date = datetime(2031, 1, 1, 0, 0, 0)

    # Convert time dimension into datetime objects
    times = [start + timedelta(days=d) for d in ds['time'][:]]
    if len(times) != ds['time'].shape[0]:
        raise ValueError("Mismatch in time steps length.")
    times = list(filter(lambda d: min_date <= d < max_date, times))
    print(f"Filtered times: {times}")

    # Create a dictionary to capture months for each year
    monthly_dates = defaultdict(list)

    # Group times by year and month
    for time in times:
        monthly_dates[(time.year, time.month)].append(time)

    # Capture the closest date to the middle of each month for every year
    months = []
    for (year, month), dates in sorted(monthly_dates.items()):
        # Filter for dates closest to the 14th-16th of the month
        closest_date = min(dates, key=lambda d: abs(d.day - 15), default=None)
        if closest_date:
            months.append(closest_date)

    print(f"Months: {months}")
    return months

def interp_years(data, ds, years, frequency='daily'):
    if frequency == 'daily':
        days = get_days(ds)
        xp = np.array([day_to_hour(day) for day in days])
        x = np.arange(1, years * 365 * 24 + 1)
    elif frequency == 'monthly':
        months = get_months(ds)
        xp = np.array([day_to_hour(month) for month in months])
        x = np.arange(1, years * 12 + 1)
    else:
        raise ValueError("Unsupported frequency: choose 'daily' or 'monthly'")
    
    if len(xp) != len(data):
        raise ValueError(
            f"Mismatch between 'xp' length ({len(xp)}) and 'data' length ({len(data)}). "
            "Ensure data matches the expected temporal resolution."
        )
    
    f = np.interp(x, xp, data)
    return f


def get_atm_forcing(path, s, lat_ind, lon_ind):
    key = path.split('_')[0]
    key = key.split('\\')[-1]
    key = key.split('/')[-1]
    ds = netCDF4.Dataset(path)
    lat = ds.variables['lat'][:].data[lat_ind]
    lon = ds.variables['lon'][:].data[lon_ind]
    _forcing = ds.variables[key][s, lat_ind, lon_ind].data
    forcing = interp_years(_forcing, ds, 5, frequency='daily')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax.plot(forcing)
    ax.set_title(key)
    ax = fig.add_subplot(122)
    ax.plot(_forcing)
    ax.set_title(key)
    print(f'{key} : {ds.variables[key].units}')
    ds.close()
    return key, forcing, lat, lon

def make_atm_forcing(paths, lat_ind, lon_ind):
    s = slice(0, 365 * 5 + 2)
    atm = {}
    for path in paths:
        key, forcing, lat, lon = get_atm_forcing(path, s, lat_ind, lon_ind)
        print(f"ATM Key: {key}, Forcing Length: {len(forcing)}, Lat: {lat}, Lon: {lon}")
        atm[key] = forcing
    df = pd.DataFrame(atm, columns=list(atm.keys()))
    df.to_csv(f'atm_{lat:.0f}_{lon:.0f}_5years.txt', sep=' ', header=None, index=False,
    float_format='%.6f')
    return df

def get_ocn_forcing(path, s, ind1, ind2):
    key = path.split('_')[0]
    key = key.split('\\')[-1]
    key = key.split('/')[-1]
    ds = netCDF4.Dataset(path)
    
    # Debug: Print the shape of the latitude and longitude arrays
    print(f"Latitude array shape: {ds.variables['lat'][:].shape}")
    print(f"Longitude array shape: {ds.variables['lon'][:].shape}")
    
    lat = ds.variables['lat'][:].data[ind1, ind2]
    lon = ds.variables['lon'][:].data[ind1, ind2]
    
    # Debug: Print the extracted latitude and longitude values
    print(f"Extracted latitude: {lat}")
    print(f"Extracted longitude: {lon}")
    
    data = ds.variables[key][:].data
    xp = 60
    if data.ndim == 4:
        forcing = data[s, 0, ind1, ind2]
    else:
        forcing = data[s, ind1, ind2]
    
    print(f"Original data shape: {data.shape}, Sliced data shape: {forcing.shape}")
    
    try:
        # Pass 'monthly' frequency to interp_years for ocean data
        forcing = interp_years(forcing, ds, 5, frequency='monthly')
    except ValueError as e:
        print(f"Error in interp_years for key {key}: {e}")
        print(f"Data: {forcing}, DS: {ds}")
        raise
    
    ds.close()
    return key, forcing, lat, lon

def make_ocn_forcing(paths, ind1, ind2):
    s = slice(0, 60)
    ocn = {}
    for path in paths:
        print(f"Processing path: {path}")
        key, forcing, lat, lon = get_ocn_forcing(path, s, ind1, ind2)
        print(f"OCN Key: {key}, Forcing Length: {len(forcing)}, Lat: {lat}, Lon: {lon}")
        ocn[key] = forcing
    df = pd.DataFrame(ocn, columns=list(ocn.keys()))
    df.to_csv(f'ocn_{lat:.0f}_{lon:.0f}_5years.txt', sep=' ', header=None, index=False,
              float_format='%.6f')
    return df
        
# Print available time range
ds = netCDF4.Dataset(r'C:\\Users\\Lee\\Desktop\\Data\\rsds_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc')
time_var = ds.variables['time']
times = netCDF4.num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False)
print(f"Available times: {times[0]} to {times[-1]}")
ds.close()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#create a matrix of all the values I want here
n = 0
c = 0
lat = [123]
lon = [247]
ind1 = [276]
ind2 = [267]
atm_file = f'atm_83_347_5years.txt'
ocn_file = f'ocn_83_347_5years.txt'

# ds = netCDF4.Dataset(path)
# lat = ds.variables['lat'][:].data[lat_ind]
# lon = ds.variables['lon'][:].data[lon_ind]

# lat = ds.variables['lat'][:].data[lat_ind]
# lon = ds.variables['lon'][:].data[lon_ind]
    
input_file_path = r'\\wsl.localhost\Ubuntu\home\leeeee05\Icepack-vargas_thesis_2018\perrycase\icepack_in'

for n in range(len(lat)):
    for c in [0, 1]: 
        replacement_text = "control.txt" if c == 0 else "pumping.txt"

        # Read the file and replace the text
        with open(input_file_path, "r") as file:
            content = file.read()

        # Replace the existing file name with the new one
        content = content.replace("pumping.txt", replacement_text).replace("control.txt", replacement_text)

        # Write the modified content back to the file
        with open(input_file_path, "w") as file:
            file.write(content)

        print(f"File updated successfully to use {replacement_text}.")
            
        # Example usage
        forcing = make_atm_forcing(
            paths=[
                r'C:\\Users\\Lee\\Desktop\\Data\\rsds_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\rlds_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\uas_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\vas_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\tas_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\huss_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\pr_day_CNRM-CM5_rcp45_r1i1p1_20260101-20301231.nc',
            ],
            lat_ind=123,
            lon_ind=247,
        )

        ocn_forcing = make_ocn_forcing(
            paths=[
                r'C:\\Users\\Lee\\Desktop\\Data\\tos_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\sos_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\mlotst_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\uo_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
                r'C:\\Users\\Lee\\Desktop\\Data\\vo_Omon_CNRM-CM5_rcp45_r1i1p1_202601-203512.nc',
            ],
            ind1=276,
            ind2=267,
        )

        # Open and modify the file
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        # Set up your file replacements for c
        if c == 0:
            pump_data_file = f"control.txt"
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
        # plt.show()

        # Running control
        new_modified_atm_contents = f'atm_{lat[n]}_{lon[n]}_5years.txt'
        print(f"{new_modified_atm_contents}")
        new_modified_ocn_contents = f'ocn_{lat[n]}_{lon[n]}_5years.txt'

        # Read the content of the file into file_contents
        with open(input_file_path, "r") as file:
            file_contents = file.read()

        # Perform the replacement
        modified_atm_contents = file_contents.replace(atm_file, new_modified_atm_contents)
        modified_ocn_contents = file_contents.replace(ocn_file, new_modified_ocn_contents)

        with open(input_file_path, 'w') as file:
            file.write(modified_atm_contents)
            file.write(modified_ocn_contents)

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
        "ice_area_fraction": "area fraction",
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

            if LABELS["ice_area_fraction"] in line and "melt pond" not in line:  # Exclude 'melt pond'
                area = extract_value(line, LABELS["ice_area_fraction"])
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
    print(f"{ice_area_fraction}")

    time_steps_in_days = [step / 24 for step in time_steps]

# Create a plot for each variable
plt.figure(figsize=(12, 8))

# Plot ice area fraction
plt.subplot(2, 2, 1)
plt.plot(time_steps_in_days, [((1-value)/10+.9) for value in ice_area_fraction], label="Ice Area Fraction", color="green")
plt.xlabel("Time Step (days)")
plt.ylabel("Area Fraction")
plt.title("Ice Area Fraction")
plt.grid(True)

# Plot ice thickness
plt.subplot(2, 2, 2)
plt.plot(time_steps_in_days, np.array(ice_thickness) / 2, label="Ice Thickness (m)", color="orange")
plt.xlabel("Time Step (days)")
plt.ylabel("Thickness (m)")
plt.title("Ice Thickness")
plt.grid(True)

# Plot top melt
plt.subplot(2, 2, 3)
plt.plot(time_steps_in_days, top_melt, label="Top Melt (m)", color="red")
plt.xlabel("Time Step (days)")
plt.ylabel("Top Melt (m)")
plt.title("Top Melt")
plt.grid(True)

# Plot bottom melt
plt.subplot(2, 2, 4)
plt.plot(time_steps_in_days, bottom_melt, label="Bottom Melt (m)", color="purple")
plt.xlabel("Time Step (days)")
plt.ylabel("Bottom Melt (m)")
plt.title("Bottom Melt")
plt.grid(True)

# Save the first figure
output_plot_filename = f"83_247_{type}_plots.png"
plt.tight_layout()
plt.savefig(output_plot_filename, dpi=300)
print(f"Graph saved as {output_plot_filename}")

# Create a new figure for the location map
plt.figure(figsize=(8, 8))

# Plot map with coordinate dotted
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)

# Ensure the coordinates are correctly plotted
lat_value = 83
lon_value = 247
ax.plot(lon_value, lat_value, 'ro', markersize=10, transform=ccrs.Geodetic())
plt.title("Location Map")

# Save the second figure
output_map_filename = f"83_247_{type}_location_map.png"
plt.savefig(output_map_filename, dpi=300)
print(f"Location map saved as {output_map_filename}")

# Show the plots
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Function to read data from a file
def read_data(filename):
    time_steps = []
    air_temp = []
    ice_area_fraction = []
    top_melt = []
    bottom_melt = []
    lateral_melt = []
    ice_thickness = []

    def extract_value(line, label):
        pattern = rf"{re.escape(label)}\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
        match = re.search(pattern, line)
        return float(match.group(1)) if match else None

    LABELS = {
        "air_temp": "air temperature (C)",
        "ice_area_fraction": "area fraction",
        "top_melt": "top melt (m)",
        "bottom_melt": "bottom melt (m)",
        "lateral_melt": "lateral melt (m)",
        "ice_thickness": "ice thickness (m)"
    }

    with open(filename, "r") as file:
        current_time_step = None
        for line in file:
            if "istep1:" in line:
                match = re.search(r"istep1:\s+(\d+)", line)
                if match:
                    current_time_step = int(match.group(1))
                    time_steps.append(current_time_step)
            if not line:
                continue
            if LABELS["air_temp"] in line:
                temp = extract_value(line, LABELS["air_temp"])
                air_temp.append(temp if temp is not None else float("nan"))
            if LABELS["ice_area_fraction"] in line and "melt pond" not in line:
                area = extract_value(line, LABELS["ice_area_fraction"])
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

    min_length = min(len(time_steps), len(air_temp), len(ice_area_fraction), len(top_melt), len(bottom_melt), len(lateral_melt), len(ice_thickness))
    time_steps = time_steps[:min_length]
    air_temp = air_temp[:min_length]
    ice_area_fraction = ice_area_fraction[:min_length]
    top_melt = top_melt[:min_length]
    bottom_melt = bottom_melt[:min_length]
    lateral_melt = lateral_melt[:min_length]
    ice_thickness = ice_thickness[:min_length]

    return {
        "time_steps": time_steps,
        "air_temp": air_temp,
        "ice_area_fraction": ice_area_fraction,
        "top_melt": top_melt,
        "bottom_melt": bottom_melt,
        "lateral_melt": lateral_melt,
        "ice_thickness": ice_thickness
    }

# Read data from control and test files
control_data = read_data("83_247_control.txt")
test_data = read_data("83_247_test.txt")

# Calculate differences
differences = {
    "time_steps": control_data["time_steps"],
    "air_temp": np.array(test_data["air_temp"]) - np.array(control_data["air_temp"]),
    "ice_area_fraction": np.array(test_data["ice_area_fraction"]) - np.array(control_data["ice_area_fraction"]),
    "top_melt": np.array(test_data["top_melt"]) - np.array(control_data["top_melt"]),
    "bottom_melt": np.array(test_data["bottom_melt"]) - np.array(control_data["bottom_melt"]),
    "lateral_melt": np.array(test_data["lateral_melt"]) - np.array(control_data["lateral_melt"]),
    "ice_thickness": np.array(test_data["ice_thickness"]) - np.array(control_data["ice_thickness"])
}

time_steps_in_days = [step / 24 for step in differences["time_steps"]]

# Create a plot for each variable difference
plt.figure(figsize=(12, 8))

# Plot air temperature difference
plt.subplot(3, 2, 1)
plt.plot(time_steps_in_days, differences["air_temp"], label="Air Temp Difference (C)", color="blue")
plt.xlabel("Time Step (days)")
plt.ylabel("Difference (C)")
plt.title("Air Temp Difference")
plt.grid(True)

# Plot ice area fraction difference
plt.subplot(3, 2, 2)
plt.plot(time_steps_in_days, differences["ice_area_fraction"], label="Ice Area Fraction Difference", color="green")
plt.xlabel("Time Step (days)")
plt.ylabel("Difference")
plt.title("Ice Area Fraction Difference")
plt.grid(True)

# Plot top melt difference
plt.subplot(3, 2, 3)
plt.plot(time_steps_in_days, differences["top_melt"], label="Top Melt Difference (m)", color="red")
plt.xlabel("Time Step (days)")
plt.ylabel("Difference (m)")
plt.title("Top Melt Difference")
plt.grid(True)

# Plot bottom melt difference
plt.subplot(3, 2, 4)
plt.plot(time_steps_in_days, differences["bottom_melt"], label="Bottom Melt Difference (m)", color="purple")
plt.xlabel("Time Step (days)")
plt.ylabel("Difference (m)")
plt.title("Bottom Melt Difference")
plt.grid(True)

# Plot lateral melt difference
plt.subplot(3, 2, 5)
plt.plot(time_steps_in_days, differences["lateral_melt"], label="Lateral Melt Difference (m)", color="brown")
plt.xlabel("Time Step (days)")
plt.ylabel("Difference (m)")
plt.title("Lateral Melt Difference")
plt.grid(True)

# Plot ice thickness difference
plt.subplot(3, 2, 6)
plt.plot(time_steps_in_days, differences["ice_thickness"], label="Ice Thickness Difference (m)", color="orange")
plt.xlabel("Time Step (days)")
plt.ylabel("Difference (m)")
plt.title("Ice Thickness Difference")
plt.grid(True)

# Save the difference figure
output_diff_plot_filename = f"83_247_{type}_differences.png"
plt.tight_layout()
plt.savefig(output_diff_plot_filename, dpi=300)
print(f"Difference graph saved as {output_diff_plot_filename}")

# Show the plots
plt.show()
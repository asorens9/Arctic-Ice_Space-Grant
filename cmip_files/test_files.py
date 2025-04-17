import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the atmospheric forcing file (update the filename as needed)
atm_file = 'atm_83_347_10years.txt'

# Assuming columns: time longwave sw_rad aira_temp sfc_pressure specific_humidity rainfall
col_names = ['time', 'longwave', 'sw_rad', 'air_temp', 'sfc_pressure', 'specific_humidity', 'rainfall']
df = pd.read_csv(atm_file, sep='\s+', names=col_names, engine='python')

# Generate a synthetic time axis (assuming hourly forcing)
time = np.arange(len(df)) / 24 / 365  # Convert to years for x-axis

# Plot air temperature
plt.figure(figsize=(12, 6))
plt.plot(time, df['air_temp'], label='Air Temperature (°C)')
plt.xlabel('Years')
plt.ylabel('Air Temperature (°C)')
plt.title('Air Temperature from Forcing File')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

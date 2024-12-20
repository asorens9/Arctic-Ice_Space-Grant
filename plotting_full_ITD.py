import re
import matplotlib.pyplot as plt

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
with open("ice_diag.full_ITD", "r") as file:
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

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

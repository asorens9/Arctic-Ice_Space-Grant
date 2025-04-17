import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def parse_ice_thickness(file_path):
    """
    Parses the avg ice thickness (m) from the given log file.
    Returns a pandas DataFrame with date and ice thickness.
    """
    data = []
    date_pattern = re.compile(r"^istep1:\s+\d+\s+idate:\s+(\d{8})")
    thickness_pattern = re.compile(r"avg ice thickness \(m\)\s*=\s*([\d\.\-Ee]+)")

    with open(file_path, "r") as file:
        current_date = None
        for line in file:
            date_match = date_pattern.search(line)
            if date_match:
                current_date = datetime.strptime(date_match.group(1), "%Y%m%d")
            
            thickness_match = thickness_pattern.search(line)
            if thickness_match and current_date:
                thickness = float(thickness_match.group(1))
                data.append((current_date, thickness))

    df = pd.DataFrame(data, columns=["Date", "Ice Thickness"])
    df.set_index("Date", inplace=True)

    return df

def plot_ice_thickness(control_file, test_file):
    """
    Reads and plots the avg ice thickness for both the control and test file.
    The full dataset (2026-2036) is plotted, but x-axis labels are rescaled to show 2026-2031.
    """
    df_control = parse_ice_thickness(control_file)
    df_test = parse_ice_thickness(test_file)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # First subplot: Delta plot
    axes[1].plot(df_control.index, df_test["Ice Thickness"] - df_control["Ice Thickness"], 
                 label="Delta", color="blue", linestyle="--")
    axes[1].set_ylabel("Delta Ice Thickness (m)")
    axes[1].set_title("Delta Ice Thickness: Test - Control")
    axes[1].legend()
    axes[1].grid(True)

    # Second subplot: Control and Test plots
    axes[0].plot(df_control.index, df_control["Ice Thickness"], label="Control", color="blue", linestyle="--")
    axes[0].plot(df_test.index, df_test["Ice Thickness"], label="Test", color="red", linestyle="--")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Average Ice Thickness (m)")
    axes[0].set_title("Comparison of Ice Thickness: Control vs. Test")
    axes[0].legend()
    axes[0].grid(True)

    # # Rescale x-axis labels
    # original_dates = df_control.index
    # num_ticks = 6  # Control how many labels to show (2026, 2027, ..., 2031)
    # new_labels = pd.date_range(start="2026-01-01", periods=num_ticks, freq="Y")  # Fake new labels from 2026-2031
    # Rescale x-axis labels to match the data's x-axis
    original_dates = df_control.index
    num_ticks = 6  # Control how many labels to show
    new_labels = pd.date_range(start=original_dates.min(), end=original_dates.max(), periods=num_ticks)


    # Map real dates to new fake labels
    tick_positions = np.linspace(0, len(original_dates)-1, num_ticks, dtype=int)
    plt.xticks(original_dates[tick_positions], new_labels.year)  # Apply new fake labels

    plt.show()

# Run the script
if __name__ == "__main__":
    control_file = "83_347_control.full_ITD"  # Update with actual control file
    test_file = "83_347_volume.full_ITD"                    # Update with actual test file
    plot_ice_thickness(control_file, test_file)

#.09548

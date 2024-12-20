# pumping file
# Define constants
# pump_start = int(input("Enter pump_start in days: "))
# pump_end = int(input("Enter pump_end in days: "))
# pump_repeats = int(input("Enter the number of pump repeats: "))
# pump_amnt = float(input("Enter the pump amount for rain: "))

pump_start = 318
pump_end = 320
pump_repeats = 1
pump_amnt = .1
total = 365 * 10 
pump_start_nt = pump_start * 24
pump_end_nt = pump_end * 24
total_nt = total * 24  # Total time steps in hours for the whole duration

# Create an empty list to hold rain values, initializing all to zero
rainfall_data = [0] * total_nt

# Loop through the number of repeats to calculate rainfall
for i in range(1, pump_repeats + 1):
    # Adjust time for each repeat cycle
    cycle_start_nt = pump_start_nt + (24 * 365 * (i - 1))
    cycle_end_nt = pump_end_nt + (24 * 365 * (i - 1))
    
    # Set the rainfall amount for each time step in the pumping period
    for nt in range(cycle_start_nt, min(cycle_end_nt + 1, total_nt)):
        rainfall_data[nt] = pump_amnt

# Write results to a text file
with open("pumping.txt", "w") as file:
    for nt, frain in enumerate(rainfall_data):
        file.write(f"{frain}\n")

print("Rainfall data with zeroes for non-rain points written to 'pumping.txt'")
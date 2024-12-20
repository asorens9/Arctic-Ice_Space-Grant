import subprocess

# Step 1: Define the path to your input file
input_file_path = r'\\wsl.localhost\Ubuntu\home\leeeee05\Icepack-vargas_thesis_2018\perrycase\icepack_in'

# Step 2: Open the file and replace the text
with open(input_file_path, 'r') as file:
    file_contents = file.read()

# Replace 'atm_83.....' with your new text (e.g., 'atm_new_value')
modified_contents = file_contents.replace("testing.txt", "atm_83_347_5years.txt")

# Step 3: Save the modified contents back to the file
with open(input_file_path, 'w') as file:
    file.write(modified_contents)

# Step 4: Run the command to use the modified file
subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.build"', shell=True)
subprocess.run('wsl bash -c "cd ~/Icepack-vargas_thesis_2018/perrycase && ./icepack.submit"', shell=True)

# Step 5: Open and read the ice_diag.full_ITD file
result = subprocess.run("wsl cat ~/icepack-dirs/runs/perrycase/ice_diag.full_ITD", shell=True, capture_output=True, text=True)

# Print the file contents
print("Contents of ice_diag.full_ITD:\n", result.stdout)


#################################################

import subprocess

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
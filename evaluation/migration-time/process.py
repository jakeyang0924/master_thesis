# convert raw data to migrate and resume time with cumulative distribution

# Open the input file for reading
with open('ho2_raw.txt', 'r') as input_file:
    # Read all lines from the input file and process them
    lines = input_file.readlines()

# Initialize lists to store the calculated differences
resume_diffs = []
migrate_diffs = []

# Process each line
for line in lines:
    # Split the line into a list of numbers
    numbers = line.strip().split()
    
    # Ensure there are three numbers in the list
    if len(numbers) == 3:
        # Convert the numbers to floats and multiply by 1000
        num1 = float(numbers[0]) * 1000
        num2 = float(numbers[1]) * 1000
        num3 = float(numbers[2]) * 1000
        
        # Calculate the differences
        resume_diff = num3 - num1
        migrate_diff = num3 - num2
        
        # Append the differences to the respective lists
        resume_diffs.append(resume_diff)
        migrate_diffs.append(migrate_diff)
    else:
        print(f"Skipping line with incorrect number of values: {line}")

# Sort the calculated differences
resume_diffs.sort()
migrate_diffs.sort()

# Open the output files for writing
with open('ho2-resume.txt', 'w') as resume_file, open('ho2-migrate.txt', 'w') as migrate_file:
    # Calculate the cumulative distribution and write to output files
    total_values = len(resume_diffs)
    for i in range(total_values):
        resume_cdf = (i + 1) / total_values
        migrate_cdf = (i + 1) / total_values
        resume_file.write(f"{resume_diffs[i]} {resume_cdf:.2f}\n")
        migrate_file.write(f"{migrate_diffs[i]} {migrate_cdf:.2f}\n")

print("Calculation and writing completed.")

# convert the sorted raw data in microsecond to data in millisecond with cumulative distribution

# Read data from input file
input_filename = 'training_time_raw.txt'
output_filename = 'training_time.txt'

data = []
with open(input_filename, 'r') as input_file:
    for line in input_file:
        data.append(float(line.strip()))

# Sort the data
sorted_data = sorted(data)

# Divide all data by 1000 and round to 3 decimal places
processed_data = [round(value / 1000, 3) for value in sorted_data]

# Calculate cumulative distribution
total_data = len(processed_data)
cumulative_dist = [i / total_data for i in range(1, total_data + 1)]

# Write processed data and cumulative distribution to output file
with open(output_filename, 'w') as output_file:
    for value, dist in zip(processed_data, cumulative_dist):
        output_file.write(f"{value:.3f}\t{dist}\n")

print(f"Processing complete. Result saved in '{output_filename}'.")

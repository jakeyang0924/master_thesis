# calculate each column's average

input_filename = 'throughput_7f_online_100+.txt'

column_sums = []
num_lines = 0

with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()
    num_columns = len(lines[0].split())  # Assuming all lines have the same number of columns
    
    # Initialize the list to hold column sums
    column_sums = [0] * num_columns
    
    for line in lines:
        values = list(map(float, line.strip().split()))
        
        # Update column sums
        for i, value in enumerate(values):
            column_sums[i] += value
        
        num_lines += 1  # Increment the count of lines

# Calculate and print column averages
column_averages = [sum_ / num_lines for sum_ in column_sums]
print(" ".join(f"{avg:.2f}" for avg in column_averages))


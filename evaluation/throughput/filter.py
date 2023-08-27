# read a raw file and filter with condition, change line 11

file_name = 'throughput_6f_pretrain'
input_filename = f'{file_name}.txt'
output_filename = f'{file_name}_100+.txt'

with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()

# Filter lines based on the condition
filtered_lines = [line for line in lines if abs(float(line.split()[0]) - float(line.split()[1])) > 100]

with open(output_filename, 'w') as output_file:
    output_file.writelines(filtered_lines)

print(f"Processing complete. Result saved in '{output_filename}'.")

input_filename = 'test_rw_pretrain2400.txt'
output_filename = 'rw_pretrain.txt'

with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()

# Remove specific lines and prepare modified lines with line numbers
m = 2
modified_lines = [f"{line_number // m} {line}" for line_number, line in enumerate(lines, start=1) if line_number % m == 0]

with open(output_filename, 'w') as output_file:
    output_file.writelines(modified_lines)

print(f"Processing complete. Result saved in '{output_filename}'.")


import random

input_filename = 'data_online.txt'
output_filename = 'data_online_rand.txt'

with open(input_filename, 'r') as input_file:
    lines = input_file.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

with open(output_filename, 'w') as output_file:
    output_file.writelines(lines)

print(f"Shuffling complete. Result saved in '{output_filename}'.")

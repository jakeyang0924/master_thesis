# Read input lines from the user, one line at a time
lines = []
while True:
    try:
        line = input()
        if line.lower() == 'done':
            break
        lines.append(float(line))
    except ValueError:
        print("Invalid input. Please enter a valid number or 'done' to calculate average.")

# Calculate the average of the numbers
if len(lines) > 0:
    average = sum(lines) / len(lines)
    print(f"Average: {average:.2f}")
else:
    print("No numbers entered.")

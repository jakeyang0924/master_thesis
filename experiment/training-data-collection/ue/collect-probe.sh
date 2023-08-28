#!/bin/bash

# Function to get the current received bytes of the interface
get_received_bytes() {
    local interface=$1
    local received_bytes=$(cat /proc/net/dev | grep "$interface" | awk '{print $2}')
    echo "$received_bytes"
}

# Function to convert bytes to Mbit/s
convert_to_mbits() {
    local bytes=$1
    local mbits=$(echo "scale=2; $bytes / 125000" | bc)
    printf "%.2f" "$mbits"
}

# Set the interface as the first argument passed to the script
output_file="data.txt"
interface=$1

# Create or clear the data.txt file
> $output_file

# Infinite loop to calculate and print the difference every second
pre_date=$(date +%s%3N)
pre_bytes=$(get_received_bytes "$interface")
while true; do
    # Sleep for 1 second
    sleep 1

    cur_bytes=$(get_received_bytes "$interface")
    difference=$((cur_bytes - pre_bytes))
    difference_mbits=$(convert_to_mbits "$difference")

    # Print the difference in bytes and Mbit/s on the terminal
    printf "\n$difference bytes ($difference_mbits Mbit/s)\n"
    echo "$difference_mbits" >> $output_file
    pre_bytes=$cur_bytes

    cur_date=$(date +%s%3N)
    date_dif=$((cur_date - pre_date))
    echo "$(echo "scale=3; $date_dif / 1000" | bc)"
done

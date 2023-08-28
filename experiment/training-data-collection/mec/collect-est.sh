#!/bin/bash

# Set the interface as the first argument passed to the script
output_file="data.txt"

# Create or clear the data.txt file
> $output_file

pre_date=$(date +%s%3N)
while true; do
    # try to align the timing of collecting data with UE
    sleep 0.93

    est_thrp=$(python3 /home/nems/get-est-thrp.py)
    echo "est: $est_thrp"
    printf "$est_thrp\n" >> $output_file

    cur_date=$(date +%s%3N)
    date_dif=$((cur_date - pre_date))
    echo "sec: $(echo "scale=3; $date_dif / 1000" | bc)"
    echo ""

done

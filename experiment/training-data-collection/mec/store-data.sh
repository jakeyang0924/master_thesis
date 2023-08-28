#!/bin/bash
trunc_lines=$1
output_file="/home/nems/est-data/$2"
src_file="data.txt"

cp $src_file $output_file
sed -i "1,${trunc_lines}d" "$output_file"

clear
unset label
reset

filename = "rw_pretrain"
data_file = sprintf("./%s.txt", filename)
output_file = sprintf("./%s.eps", filename) 

set output output_file
set terminal postscript eps enhanced color  "Times-Roman" 28
set size 1,1

set ylabel "Reward" font ",40" offset 1
set yrange [-8000:2600]
set ytics -8000,2000 font ",34" offset 0.4

set xrange [0:1201]
set xtics 0,300 font ",34" offset -0.2,0.3
set xlabel "Round" font ",40" offset 0,0.5

set key right bottom
set key samplen 2.5 font ",34"
set border lw 2

set key noautotitle

plot data_file using 1:2 with lines lw 0.5 lt 1 lc rgb "#2860ed"

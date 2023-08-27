clear
unset label
reset

filename = "training_time"
output_file = sprintf("./%s.eps", filename)

set output output_file
set terminal postscript eps enhanced color  "Times-Roman" 28
set size 1,1
set bmargin 3

set ylabel "CDF" font ",40" offset 1.7
set yrange [0:1]
set ytics 0,0.2 font ",34" offset 0.4

set xlabel "Latency (ms)" font ",40" offset 0,0.2
set xrange [0:100]
set xtics (0,20,40,60,80,100) font ",34" offset 0,0.1

set key samplen 2.5 font ",34"
set key left top
set border lw 2

set key noautotitle
plot "training_time.txt" using 1:2 with lines lw 3 dashtype 1 lt 2 lc rgb "#2860ed"

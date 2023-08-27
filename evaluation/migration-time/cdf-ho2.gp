clear
unset label
reset

filename = "ho2"
output_file = sprintf("./%s.eps", filename)

set output output_file
set terminal postscript eps enhanced color  "Times-Roman" 28
set size 1,1
set bmargin 3

set ylabel "CDF" font ",40" offset 1.7
set yrange [0:1]
set ytics 0,0.2 font ",34" offset 0.4

set xlabel "Latency (ms)" font ",40" offset 0,0.2
set xrange [0:470]
set xtics 0,75 font ",34" offset 0,0.1

set key samplen 2.5 font ",34"
set key right bottom
set border lw 2

plot "ho2-migrate.txt" using 1:2 title 'Migrate' with lines lw 3 dashtype 1 lt 2 lc rgb "#2860ed", \
    "ho2-resume.txt" using 1:2 title 'Resume' with lines lw 3 dashtype 3 lt 2 lc rgb "#ba2323", \

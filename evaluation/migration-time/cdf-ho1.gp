clear
unset label
reset

filename = "ho1"
output_file = sprintf("./%s.eps", filename)

set output output_file
set terminal postscript eps enhanced color  "Times-Roman" 28
set size 1,1
set bmargin 3

set ylabel "CDF" font ",40" offset 1.7
set yrange [0:1]
set ytics 0,0.2 font ",34" offset 0.4

set xlabel "Latency (sec)" font ",40" offset 0,0.2
set xrange [0:41]
set xtics (0,5,10,15,20,25,30,35,40) font ",34" offset 0,0.3

set key samplen 2.5 font ",34"
set key right bottom
set border lw 2

plot "ho1-migrate.txt" using 1:2 title 'Migrate' with lines lw 3 dashtype 1 lt 2 lc rgb "#2860ed", \
    "ho1-resume.txt" using 1:2 title 'Resume' with lines lw 3 dashtype 3 lt 2 lc rgb "#ba2323", \

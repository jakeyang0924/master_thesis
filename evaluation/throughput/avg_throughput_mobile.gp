clear
unset label
reset

filename = "avg_throughput_mobile"
output_file = sprintf("./%s.eps", filename)

set output output_file
set terminal postscript eps enhanced color  "Times-Roman" 28
set size 1,1
set bmargin 3

set xlabel "Throughput between 5G and Wi-Fi" font ",34" offset 0,0.5
set xtics ("In 100Mbps" -0.02, "Over 100Mbps" 1) font ",28" offset 0,0.3
set ylabel "Throughput (Mbps)" font ",34" offset 1.7
set yrange[0:220]
set ytics 0,40 font ",28" offset 0.4

set style data histogram
set style histogram cluster gap 3
set boxwidth 0.95

set key above
set border lw 2

plot \
"avg_throughput_mobile.txt" u 2 title "5G" lc rgb "black" fs pattern 0, \
"" u 3 title "Wi-Fi" lc rgb "#ba2323" fs pattern 4, \
"" u 4 title "Best" lc rgb "#edb128" fs pattern 5, \
"" u 5 title "Pretrained" lc rgb "#36a2b5" fs pattern 1, \
"" u 6 title "Online" lc rgb "#0762f5" fs pattern 2, \


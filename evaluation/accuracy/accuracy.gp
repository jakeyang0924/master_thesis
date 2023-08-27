clear
unset label
reset

filename = "accuracy"
output_file = sprintf("./%s.eps", filename)

set output output_file
set terminal postscript eps enhanced color  "Times-Roman" 28
set size 1,1
set bmargin 3

unset xtics
set ylabel "Accuracy (%)" font ",34" offset 1.7
set yrange[0:100]
set ytics 0,20 font ",28" offset 0.4

set style data histogram
set style histogram cluster gap 1
set boxwidth 0.95

set key above
set border lw 2

plot \
"accuracy.txt" u 2 title "Pretrained+Static" lc rgb "black" fs pattern 0, \
"" u 3 title "Pretrained+Mobile" lc rgb "#ba2323" fs pattern 4, \
"" u 4 title "Online+Mobile" lc rgb "#edb128" fs pattern 5, \

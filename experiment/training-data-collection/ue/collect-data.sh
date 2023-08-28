#!/bin/bash
# Set the interface as the first argument passed to the script
SESSION_NAME=data
interface=$1

tmux send-keys -t $SESSION_NAME:0 "./collect-probe.sh $interface" C-m

if [ "$interface" == "wlp1s0" ]
then
    # tmux select-pane -t $SESSION_NAME:0 -L
    # tmux send-keys -t $SESSION_NAME:0 "./collect-est.sh" C-m
    tmux select-pane -t $SESSION_NAME:0 -U
    tmux send-keys -t $SESSION_NAME:0 "iperf3 -c 172.18.2.1 -R -t0" C-m
else
    tmux select-pane -t $SESSION_NAME:0 -L
    tmux send-keys -t $SESSION_NAME:0 "./collect-est.sh" C-m
    tmux select-pane -t $SESSION_NAME:0 -U
    tmux send-keys -t $SESSION_NAME:0 "iperf3 -c 172.18.2.1 -R -t0 -P128" C-m
    ping 172.18.2.1 -I wlp1s0 > /dev/null    
fi


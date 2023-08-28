#!/bin/bash
# Set the interface as the first argument passed to the script
SESSION_NAME=data

# tmux kill-window -t $SESSION_NAME:0
tmux new-session -d -s $SESSION_NAME
tmux split-window -v -t $SESSION_NAME:0
tmux select-pane -t $SESSION_NAME:0 -U
tmux split-window -h -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0 "sshpass -p nems@704 ssh nems@172.18.2.1 -o strictHostKeyChecking=no" C-m
tmux select-pane -t $SESSION_NAME:0 -L

# terminal for control panel
tmux new-window
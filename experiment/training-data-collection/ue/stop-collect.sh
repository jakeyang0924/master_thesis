#!/bin/bash
SESSION_NAME=data

tmux send-keys -t $SESSION_NAME:0 C-c
tmux select-pane -t $SESSION_NAME:0 -U
tmux send-keys -t $SESSION_NAME:0 C-c
tmux select-pane -t $SESSION_NAME:0 -L
tmux send-keys -t $SESSION_NAME:0 C-c

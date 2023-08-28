# 3 threads
# - measure 5g throughput
# - measure wifi throughput
# - get estimated throughput (subscribe)
import sys
import subprocess
import time
import difflib
import threading
import paho.mqtt.client as mqtt

import numpy as np
import math 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('aioquic/examples')
from iproute import Route

# subscriber settings
broker_address = "localhost"
broker_port = 1883
sub_topic = "est"

est_5g = None
pre_smooth_5g = None
smooth_5g = None
est_wifi = None
pre_smooth_wifi = None
smooth_wifi = None


A = 0.2


def get_interface_bytes(interface):
    command = f"cat /proc/net/dev | grep '{interface}' | awk '{{print $2}}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    return int(output)

def calculate_5g_data_rate(interface, prev_bytes):
    global probe_5g
    while True:
        time.sleep(1)  # Wait for 1 second

        current_bytes = get_interface_bytes(interface)
        bytes_diff = current_bytes - prev_bytes
        prev_bytes = current_bytes

        # Convert bytes to kilobits per second
        probe_5g = bytes_diff * 8 / 1000000
        
        # Apply exponential smoothing on probe throughput
        if smooth_5g == None:
            smooth_5g = probe_5g
        else:
            smooth_5g = pre_smooth_5g + A * (probe_5g - pre_smooth_5g)
        pre_smooth_5g = smooth_5g
            
        print(f'5g speed: {smooth_5g:.3f} Mbps')

def calculate_wifi_data_rate(interface, prev_bytes):
    global pre_smooth_wifi
    global smooth_wifi
    while True:
        time.sleep(1)  # Wait for 1 second

        current_bytes = get_interface_bytes(interface)
        bytes_diff = current_bytes - prev_bytes
        prev_bytes = current_bytes

        # Convert bytes to kilobits per second
        probe_wifi = bytes_diff * 8 / 1000000
        
        # Apply exponential smoothing on probe throughput
        if smooth_wifi == None:
            smooth_wifi = probe_wifi
        else:
            smooth_wifi = pre_smooth_wifi + A * (probe_wifi - pre_smooth_wifi)
        pre_smooth_wifi = smooth_wifi
            
        print(f'wifi speed: {smooth_wifi:.3f} Mbps')


def probe_5g_thrp():
    interface_name = "usb0"  # Replace with the desired interface name
    prev_5g_bytes = get_interface_bytes(interface_name)
    calculate_5g_data_rate(interface_name, prev_5g_bytes)

def probe_wifi_thrp():
    interface_name = "wlp1s0"  # Replace with the desired interface name
    prev_wifi_bytes = get_interface_bytes(interface_name)
    calculate_wifi_data_rate(interface_name, prev_wifi_bytes)


def on_message(client, userdata, msg):
    global est_5g, est_wifi
    message = msg.payload.decode('utf-8').split()
    est_5g = message[0], est_wifi = message[1]
    print(message[0], message[1])


def subscribe_est_thrp():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker_address, broker_port)
    client.subscribe(sub_topic)
    client.loop_forever()


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.action_dim = 2
        self.state_dim = 4
        self.fc1 = nn.Linear(self.state_dim, 16)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(16, self.action_dim)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

def select_action(eval_net, state):
    state = torch.unsqueeze(torch.FloatTensor(state), 0)
    actions_value = eval_net(state)
    action = torch.max(actions_value, 1)[1].data.numpy()[0]
    return action 

# Collect state for every second, input to model and get decision
def start_ns_module():
    dqn = torch.load("dqn_model_4000.pt")
    
    cur_route = Route.get_current_route()
    if '172.18.3.' in cur_route['ip']:
        cur_nw = 1
        state = [est_5g, est_wifi, 0, smooth_wifi]
    else:
        cur_nw = 0
        state = [est_5g, est_wifi, smooth_5g, 0]
    
    nw = select_action(dqn, state)
    if nw != cur_nw:
        Route.switch_default_route()
    

if __name__ == "__main__":
    # probe_5g_thread = threading.Thread(target=probe_5g_thrp)
    # probe_5g_thread.start()
    # probe_wifi_thread = threading.Thread(target=probe_wifi_thrp)
    # probe_wifi_thread.start()
    # mqtt_thread = threading.Thread(target=subscribe_est_thrp)
    # mqtt_thread.start()
    dqn = torch.load("dqn_model_4000.pt") # load model in 1 ms
    state = [62.15, 60.71, 60.71, 0]
    print(select_action(dqn, state))
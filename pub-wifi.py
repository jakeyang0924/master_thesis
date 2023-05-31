import time
import os
import subprocess
import json
import paho.mqtt.client as mqtt

def parse_rx_time(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            rx_time = int(lines[3].split(': ')[1].strip())
            return rx_time
    return None

def parse_tx_value(file_path):
    with open(file_path, 'r') as file:
        tx_value = None
        for line in file:
            if line.startswith('TX:'):
                tx_value = int(line.split(':')[1].strip().split()[0])
                break
        return tx_value

def get_tx_bitrate():
    command = "iw dev wlan0 station dump"
    output = subprocess.check_output(command, shell=True, text=True)
    lines = output.splitlines()
    tx_bitrates = {}
    mac_address = None
    tx_bitrate = None
    for line in lines:
        if line.strip().startswith("Station"):
            parts = line.split()
            if len(parts) > 1:
                mac_address = parts[1]
        elif line.strip().startswith("tx bitrate"):
            parts = line.split(":")[1].split()
            if len(parts) > 0:
                tx_bitrate = parts[0].strip()

        if mac_address and tx_bitrate:
            tx_bitrates[mac_address] = tx_bitrate
            mac_address = None
            tx_bitrate = None
    return tx_bitrates


broker_address = "172.18.2.1"
broker_port = 1883
topic = "wifi"
client = mqtt.Client()
client.connect(broker_address, broker_port)

previous_rx_value = parse_rx_time('/sys/kernel/debug/ieee80211/phy0/mt76/time-info')
time.sleep(1)

# Initialize dictionary for previous TX values
previous_tx_values = {}

while True:
    # Deal with rx time
    rx_value = parse_rx_time('/sys/kernel/debug/ieee80211/phy0/mt76/time-info')
    if rx_value is not None:
        rx_diff = rx_value - previous_rx_value
        
        print(f"rx_time: {rx_diff}")
        
        previous_rx_value = rx_value
    else:
        print("rx_value are none.")
    
    # Deal with every device tx airtime
    user_cnt = 0
    directories = os.listdir('/sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/')
    tx_differences = {}
    for directory in directories:
        user_cnt += 1 
        file_path = '/sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/{}/airtime'.format(directory)
        try:
            tx_value = parse_tx_value(file_path)
            if directory not in previous_tx_values:
                # Assign -1 as the value for devices encountered for the first time
                tx_differences[directory] = -1
            else:
                # Calculate the TX difference for devices with previous values
                tx_difference = tx_value - previous_tx_values[directory]
                tx_differences[directory] = tx_difference
            previous_tx_values[directory] = tx_value
        except (ValueError, FileNotFoundError):
            tx_differences[directory] = None
    print("TX Differences:", tx_differences)
    print("user num:", user_cnt)
    
    # Deal with every device's bitrate
    tx_bitrates = get_tx_bitrate()
    if tx_bitrates:
        print("Stations bitrates:", tx_bitrates)
    else:
        print(f"Stations bitrates not found")
    
    # Pack and publish the message to MQTT broker
    message_dict = {
        'rx_time': rx_diff,
        'airtime': tx_differences,
        'bitrate': tx_bitrates,
        'user_num': user_cnt
    }
    message = json.dumps(message_dict)
    client.publish(topic, message)
    
    time.sleep(1)
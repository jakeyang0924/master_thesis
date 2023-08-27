import time
import json
import paho.mqtt.client as mqtt

file_paths = {
    "bwp": "/mnt/oai_tmpfs/bwp",
    "mcs": "/mnt/oai_tmpfs/mcs",
    "slot_ue": "/mnt/oai_tmpfs/slot_cnt"
}
ue_list_file = "/mnt/oai_tmpfs/ue_list"
user_cnt = 0

def access_files():
    global user_cnt
    ue_info_data = {}  # Dictionary to store the extracted information (scs, BWPSize, mcs, Qm, R)
    slot_ue_data = {}  # Dictionary to store the "slot_ue" content
    user_cnt = 0
    with open(ue_list_file, 'r') as file:
        for line in file:
            user_cnt += 1
            ue_id = line.strip()
            ue_info = {}  # Dictionary to store the extracted information

            # Access the bwp file
            bwp_file_path = file_paths["bwp"] + ue_id
            try:
                with open(bwp_file_path, 'r') as bwp_file:
                    bwp_content = bwp_file.readline()
                    bwp_values = bwp_content.split()
                    ue_info["scs"] = bwp_values[6]
                    ue_info["BWPSize"] = bwp_values[8]
            except Exception as e:
                print(f"Error accessing file: {bwp_file_path}")
                print(f"Error message: {str(e)}")
                continue

            # Access the mcs file
            mcs_file_path = file_paths["mcs"] + ue_id
            try:
                with open(mcs_file_path, 'r') as mcs_file:
                    mcs_content = mcs_file.readline()
                    mcs_values = mcs_content.split()
                    ue_info["mcs"] = mcs_values[6]
                    ue_info["Qm"] = mcs_values[8]
                    ue_info["R"] = mcs_values[10]
            except Exception as e:
                print(f"Error accessing file: {mcs_file_path}")
                print(f"Error message: {str(e)}")
                continue

            ue_info_data[ue_id] = ue_info

            # Access the slot_ue file
            slot_ue_file_path = file_paths["slot_ue"] + ue_id
            try:
                with open(slot_ue_file_path, 'r') as slot_ue_file:
                    slot_ue_content = slot_ue_file.read().strip()
                    slot_ue_data[ue_id] = slot_ue_content
            except Exception as e:
                print(f"Error accessing file: {slot_ue_file_path}")
                print(f"Error message: {str(e)}")
    return ue_info_data, slot_ue_data

broker_address = "172.18.2.1"
broker_port = 1883
topic = "5g"
client = mqtt.Client()
client.connect(broker_address, broker_port)

_, prev_slot_ue_data = access_files()
time.sleep(1)

while True:
    ue_info_data, cur_slot_ue_data = access_files()
    
    slot_ue_data = {}
    for k,v in cur_slot_ue_data.items():
        if k in prev_slot_ue_data:
            slot_ue_data[k] = int(v) - int(prev_slot_ue_data[k])
        else:
            slot_ue_data[k] = v
    prev_slot_ue_data = cur_slot_ue_data
    
    # Pack and publish the message to MQTT broker
    message_dict = {
        'ue_info': ue_info_data,
        'slot': slot_ue_data,
        'user_num': user_cnt
    }
    message = json.dumps(message_dict)
    print("ue info:", ue_info_data)
    print("slot:", slot_ue_data)
    print("user_num:", user_cnt)
    client.publish(topic, message)
    
    time.sleep(1)
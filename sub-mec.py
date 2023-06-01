import argparse
import json
import redis
import subprocess
import paho.mqtt.client as mqtt

# us per second
US_PER_SEC = 1000000
# wifi physical/theoretical throughput ratio
C = 0.8
# alpha for exponential smoothing
A = 0.8
# slots per second
SLOTS_PER_SEC = 2000
UL_SLOTS_PER_SEC = 500

target_mac = 'ff:ff:ff:ff:ff:ff'
target_rnti = 'ffff'
r = None

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        topic = [("5g", 0), ("wifi", 0)]
        client.subscribe(topic)  # Subscribe to the desired topic
    else:
        print("Failed to connect to MQTT broker", rc)


def on_message(client, userdata, msg):
    global r
    topic = msg.topic
    message = msg.payload.decode('utf-8')

    if topic == "wifi":
        data = json.loads(message)
        rx_time = int(data['rx_time'])
        airtime_dict = data['airtime']
        bitrate_dict = data['bitrate']
        user_num = data['user_num']

        if user_num == 0:
            print("wifi receive no data...")
            return

        avail_time = float(US_PER_SEC - rx_time)
        print("avail_time:", avail_time)
        fair_share = avail_time / user_num
        print("fair share:", fair_share)

        for k, v in airtime_dict.items():
            if k != target_mac and v != -1 and v <= fair_share:
                avail_time -= v
                user_num -= 1
            print(k, v)

        est_time = avail_time / user_num
        print("est_time:", est_time)
        for k, v in bitrate_dict.items():
            if k == target_mac:
                est_throughput = round(est_time * float(v) * C / US_PER_SEC, 3)
        print("wifi estimated throughput:", est_throughput)

        # calculate new exponential smoothing
        pre_exp_smoothing = r.hget('throughput', 'wifi')
        if pre_exp_smoothing is not None:
            pre_exp_smoothing = float(pre_exp_smoothing.decode('utf-8'))
        if pre_exp_smoothing == -1:
            new_exp_smoothing = est_throughput
        else:
            new_exp_smoothing = pre_exp_smoothing + A * (est_throughput - pre_exp_smoothing)
        r.hset('throughput', 'wifi', round(new_exp_smoothing, 3))

    elif topic == "5g":
        data = json.loads(message)
        ue_info_dict = data['ue_info']
        slot_dict = data['slot']
        user_num = data['user_num']

        if user_num == 0:
            print("5g receive no data...")
            return

        avail_slot = SLOTS_PER_SEC - UL_SLOTS_PER_SEC
        print("avail_slot:", avail_slot)
        fair_slot = avail_slot // user_num
        print("fair_slot:", fair_slot)

        for k, v in slot_dict.items():
            if k != target_rnti and v <= fair_slot:
                avail_slot -= v
                user_num -= 1
            print(k, v)

        est_slot = avail_slot // user_num
        print("est_slot:", est_slot)

        # get info from target ue
        info_dict = ue_info_dict.get(target_rnti)
        if info_dict is not None:
            Qm = info_dict['Qm']
            R = info_dict['R']
            BWPSize = info_dict['BWPSize']

        # compute tbs and convert to Mbit/s
        cmd = f"./compute-tbs {Qm} {R} {BWPSize}"
        tbs = subprocess.check_output(cmd, shell=True, text=True)
        est_throughput = round(int(tbs.strip()) * est_slot / 1000000, 3)
        print("5g estimated throughput:", est_throughput)

        pre_exp_smoothing = r.hget('throughput', '5g')
        if pre_exp_smoothing is not None:
            pre_exp_smoothing = float(pre_exp_smoothing.decode('utf-8'))
        if pre_exp_smoothing == -1:
            new_exp_smoothing = est_throughput
        else:
            new_exp_smoothing = pre_exp_smoothing + A * (est_throughput - pre_exp_smoothing)
        r.hset('throughput', '5g', new_exp_smoothing)
    else:
        print("Unknown topic:", topic)


def main():
    global r
    # Connect to Redis database
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.hset('throughput', 'wifi', -1)
    r.hset('throughput', '5g', -1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mac", default="f4:8c:50:a0:92:4a", help="Target MAC address")
    parser.add_argument("-r", "--rnti", default="11e5", help="Target RNTI")
    parser.add_argument("-a", "--address", default="172.18.2.1", help="Broker address")
    args = parser.parse_args()

    global target_mac, target_rnti
    target_mac = args.mac
    target_rnti = args.rnti
    broker_address = args.address

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    port = 1883  # Default MQTT port
    client.connect(broker_address, port)
    client.loop_forever()


if __name__ == "__main__":
    main()

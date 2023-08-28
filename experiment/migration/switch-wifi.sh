#!/bin/bash

SSID=$(iwgetid -r)
SSID1="thesis"
SSID2="thesis2"

if [ "$SSID" == "$SSID1" ]
then
    SSID=$SSID2
else
    SSID=$SSID1
fi
nmcli device wifi connect "$SSID"

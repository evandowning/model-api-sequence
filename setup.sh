#!/bin/bash

uid=`id -u`

# Check user permissions
if [[ $uid -ne 0 ]]; then
    echo 'must be root'
    exit 2
fi

# Stop on any error
set -e

# Update
apt update

# Requirements for parsing raw files
# Install python2.7
apt install -y python2.7

# Install bson
apt install -y python-bson

# Requirements for modeling
# Install python3.5
apt install -y python3.5

# Install pip
apt install -y python3-pip
pip3 install --upgrade pip

# Install python libraries
pip3 install -r requirements.txt

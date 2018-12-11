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
apt-get update

# Install python2.7
apt-get install -y python2.7

# Install pip
apt-get install -y python-pip

# Install python libraries
pip install -r requirements.txt

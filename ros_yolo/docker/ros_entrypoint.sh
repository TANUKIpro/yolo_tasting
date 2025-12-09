#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/noetic/setup.bash"

# Build catkin workspace if not yet built
if [ ! -f "/catkin_ws/devel/setup.bash" ]; then
    echo "Building catkin workspace..."
    cd /catkin_ws
    catkin_make
fi

source "/catkin_ws/devel/setup.bash"
exec "$@"

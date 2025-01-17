#!/usr/bin/env sh
# generated from catkin/cmake/template/local_setup.sh.in

# since this file is sourced either use the provided _CATKIN_SETUP_DIR
# or fall back to the destination set at configure time
<<<<<<< HEAD
: ${_CATKIN_SETUP_DIR:=/home/yoo/example_ws/devel}
=======
: ${_CATKIN_SETUP_DIR:=/home/highsky/example_ws/devel}
>>>>>>> 86eba9652a14b3c4fa8cb665a1bd320c01ac08e5
CATKIN_SETUP_UTIL_ARGS="--extend --local"
. "$_CATKIN_SETUP_DIR/setup.sh"
unset CATKIN_SETUP_UTIL_ARGS

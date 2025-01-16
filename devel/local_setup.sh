#!/usr/bin/env sh
# generated from catkin/cmake/template/local_setup.sh.in

# since this file is sourced either use the provided _CATKIN_SETUP_DIR
# or fall back to the destination set at configure time
<<<<<<< HEAD
<<<<<<< HEAD
: ${_CATKIN_SETUP_DIR:=/home/junseong/example_ws/devel}
=======
: ${_CATKIN_SETUP_DIR:=/home/yoo/example_ws/devel}
>>>>>>> origin/main
=======
: ${_CATKIN_SETUP_DIR:=/home/hannibal/example_ws/devel}
>>>>>>> origin/main
CATKIN_SETUP_UTIL_ARGS="--extend --local"
. "$_CATKIN_SETUP_DIR/setup.sh"
unset CATKIN_SETUP_UTIL_ARGS

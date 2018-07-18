#!/bin/bash

##
#
# A quick script to run our simulation a bunch of times for
# statistical model checking purposes
#
##

#echo "" > results.txt

for i in {1..100};
do
    result=$(roslaunch rnn_collvoid control.launch | grep "TRIAL FINISHED" | awk '{print $5}')
    echo $result >> results.txt
done



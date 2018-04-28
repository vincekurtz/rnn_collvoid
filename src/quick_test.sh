#!/bin/bash

##
#
# Quick script to generate and test identical models
#
##

starttime=$SECONDS
# generate the model as test.pm
./dynamic_verification_mc.py

# check the model
prism test.pm non_collision.pctl

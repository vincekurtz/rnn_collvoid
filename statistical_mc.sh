#!/bin/bash

##
#
# Use a Sequential Probability Ratio Test to verify the satisfaction of our prediction accuracy 
# specification with the given probability and confidence
#
##

##### Parameters ######
theta=0.80     # probability with which we'd like the specification to be satisfied
delta=0.05    # half-width of indifference region  
alpha=0.1     # target Type-I (false positive) error
beta=0.1      # target Type-II (false negative) error

##### Preliminaries #####
p0=$(echo "$theta + $delta" | bc)
p1=$(echo "$theta - $delta" | bc)

alpha_hat=$(echo "$alpha/(1-$beta)" | bc -l)    # guaranteed maximum Type-I (false positive) error
beta_hat=$(echo "$beta/(1-$alpha)" | bc -l)     # guaranteed maximum Type-II (false negative) error

A=$(echo "(1-$beta)/$alpha" | bc -l)            # SPRT bounds
B=$(echo "$beta/(1-$alpha)" | bc -l)

m=0           # number of iterations
dm=0          # sum of successful simulations

done=false

while ! $done
do
    # 1 if successful, otherwise 0
    b_i=$(roslaunch rnn_collvoid statistical_mc.launch | grep "===> Result" | awk '{print $3}')

    # keep track of number of trials and successes
    let dm+=b_i
    let m+=1

    # Calculate probability ratio
    p1m=$(echo "$p1^$dm*(1-$p1)^($m-$dm)" | bc -l)
    p0m=$(echo "$p0^$dm*(1-$p0)^($m-$dm)" | bc -l)

    ratio=$(echo "$p1m/$p0m" | bc -l)

    # Give the user a quick update
    echo ""
    echo "p0m: $p0m"
    echo "p1m: $p1m"
    echo "m: $m"
    echo "dm: $dm"
    echo "A: $A" 
    echo "B: $B"
    echo "p1m/p0m: $ratio"
    echo ""

    if (( $(echo "$ratio > $A" | bc -l) ))
    then
        echo "SPECIFICATION NOT SATISFIED"
        echo "p < $theta with maximum (alpha, beta)=($alpha_hat, $beta_hat)"
        done=true
    elif (( $(echo "$ratio < $B" | bc -l) ))
    then
        echo "SPECIFICATION SATISISFIED"
        echo "p > $theta with maximum (alpha, beta)=($alpha_hat, $beta_hat)"
        done=true
    fi


done



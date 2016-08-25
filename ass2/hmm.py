#!/usr/bin/env python3

#hmm.py: Implementation of a hidden markov model for the Umbrella domain.
#Author: Harald Husum
#Date: 02.03.2016

import numpy as np

def forward (sensor_model, transition_model, message):
    #As defined in Equation 15.12
    return_vector = sensor_model*transition_model.getT()*message
    #normalization
    return(return_vector/return_vector.sum())

def backward (sensor_model, transition_model, message):
    #As defined in Equation 15.13
    return(transition_model*sensor_model*message)

def forward_backward (evidence, prior):
    #model matrices
    transition_model = np.matrix('0.7 0.3; 0.3 0.7')
    utrue = np.matrix('0.9 0.0; 0.0 0.2')
    ufalse = np.matrix('0.1 0.0; 0.0 0.8')
    
    #initializing local variables for algorithm
    forward_messages = [np.matrix('0.0; 0.0') for i in range((len(evidence)+1))]
    smoothed_estimates = [np.matrix('0.0; 0.0') for i in range(len(evidence))]
    backward_message = np.matrix('1.0; 1.0')
    
    #here we pretty much follow the book
    forward_messages[0] = prior
    for i in range(1, len(evidence)+1):
        #an if statement to make sure we use the right sensor model
        if (evidence[i-1]):
            forward_messages[i] = forward(utrue, transition_model, forward_messages[i-1])
        else:
            forward_messages[i] = forward(ufalse, transition_model, forward_messages[i-1])
            
    #uncomment following line for first backward_message, if desireable
    #print(backward_message)
    
    for i in range(len(evidence)-1, -1, -1):
        #couldn't find any numpy functionality for hadamard products
        #here follows my own ad hoc implementation
        smoothed0 = forward_messages[i+1].item(0)*backward_message.item(0)
        smoothed1 = forward_messages[i+1].item(1)*backward_message.item(1)
        smoothed = np.matrix([[smoothed0], [smoothed1]])
        smoothed_estimates[i] = smoothed/smoothed.sum()
        #yet another conditional helping us choosing the right model
        if (evidence[i]):
            backward_message = backward(utrue, transition_model, backward_message)
        else:
            backward_message = backward(ufalse, transition_model, backward_message)
            
        #uncomment following line for the remaining backward_messages, if desireable
        #print(backward_message)
            
    return(smoothed_estimates)
    
#simple framework to generate test data    
original_message = np.matrix('0.5; 0.5')
evidence = [True, True, False, True, True]

fb = forward_backward(evidence, original_message)
print("Estimates:\n")
for estimate in fb:
    print(estimate)
    print("")
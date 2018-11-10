# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:44:45 2018

@author: AARUSHI
"""
"""Ref :- https://hackernoon.com/dl02-writing-a-neural-network-from-scratch-code-b32f4877c257 
https://github.com/sar-gupta/neural-network-from-scratch/blob/master/neuralnetwork.py """

import numpy as np
 


class layer:
    
    def __init__(self, num_nodes, layer_num,batch_size):
       
        self.num_nodes = num_nodes
        self.activation_function = "Sigmoid"
        self.is_last = False
        
        self.weight = None
        self.bias = None
        self.changed_weight = None
        self.changed_bias = None
        self.input_activation = None
        
        if(layer_num==len(num_nodes)-1):
            
            self.is_last = True
            self.activation_function = "Softmax"
            
        else:
            self.weight = np.random.normal(0, 0.001, size=(num_nodes[layer_num], num_nodes[layer_num+1]))
            self.bias = np.random.normal(0, 0.001, size=(1, num_nodes[layer_num+1]))
            self.changed_weight = np.zeros((num_nodes[layer_num], num_nodes[layer_num+1]),dtype = float)
            self.changed_bias = np.zeros((1, num_nodes[layer_num+1]),dtype = float)
        
            
class Neural_Network(object):
    
    def __init__(self, nlayer, num_node, batch_size):
        
         
        self.nlayer = nlayer
        self.num_node= num_node
        self.layers = []
        self.batch_size = batch_size

        
        for i in range(nlayer):
            layer_i = layer(num_node, i,self.batch_size)
            self.layers.append(layer_i)     
        
        
    def softmax(self, layer):
        num = np.exp(layer)
        den = np.sum(num)
        return num/den
 
       
       
    def sigmoid(self,x, Derivative = False):
        
        if(Derivative):
            return (x*(1.0 - x))
           # return (self.sigmoid(x)*1-self.sigmoid(x))
        
        return 1 / (1 + np.exp(-x))
    
    
    def cross_entropy_derivative(p,y):
        return (p-y)
    
    
    
        
    def train(self,batch_size, input, labels, num_epochs,learning_rate, filename):
        
        self.learning_rate = learning_rate
        num_batches = int(len(input)/batch_size)
        print("Started")
        for i in range(num_epochs):
                
            for j in range(num_batches):
                
                for k in range(j*batch_size , j*batch_size+batch_size):
                    
                    output = (np.zeros((self.num_node[self.nlayer-1], 1)))
                    output[labels[k]]=1 
                    output = output.T
                    
                    #Forward Pass
                    self.forward_pass(input[k])
                    ##ORIGINAL
                    ###self.error -= (np.sum((labels[j*batch_size : j*batch_size+batch_size] * np.log(self.layers[self.nlayer-1].input_activation ))))
                    
                    #Backward Pass
                    self.back_pass(output) 
                
                ##Gradient Descent
                self.grad_descent()
                
            print("Epoch "+str(i)+" done")
            
        print("Training Done")
 
                
                
    def forward_pass(self,input):

        self.layers[0].input_activation = np.reshape(input,(1,input.shape[0]))
        
        for i in range(self.nlayer-1):

            first_term = np.dot(self.layers[i].input_activation,self.layers[i].weight)
            second_term = self.layers[i].bias
            curr_temp = np.add(first_term ,second_term)

            if(self.layers[i+1].activation_function=="Sigmoid"):
                self.layers[i+1].input_activation = self.sigmoid(curr_temp)
                
            else:
                self.layers[i+1].input_activation = self.softmax(curr_temp)
                 
                
               
                
    def back_pass(self, labels):
        # if self.cost_function == "cross_entropy" and self.layers[self.num_layers-1].activation_function == "softmax":
        layer = self.layers[self.nlayer-1]
        prev_layer = self.layers[self.nlayer-2]
        delta = cross_entropy_derivative(layer.input_activation,labels)
        prev_layer.changed_bias += delta
        prev_layer.changed_weight += np.dot(prev_layer.input_activation.T, delta)
        
        for i in range(self.nlayer-2,0,-1):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]
            print("jjojo")
            print(np.dot(layer.weight, delta.T).shape)
            print(self.sigmoid(layer.input_activation,True).T.shape)
            delta = np.multiply(np.dot(layer.weight, delta.T), self.sigmoid(layer.input_activation,True).T)
            print(delta.shape)
            print(prev_layer.changed_bias.shape)
            
            prev_layer.changed_bias += delta.T
            print(delta.shape)
            print(prev_layer.input_activation.T.shape)
            print("byeeeeeeeeeeeeeeeeeeeee")
            prev_layer.changed_weight += np.dot(prev_layer.input_activation.T, delta.T) 
            
            
            
    def grad_descent(self):
        for i in range(0,self.nlayer-1):
            layer = self.layers[i]
            layer.weight = layer.weight - (self.learning_rate/self.batch_size)*layer.changed_weight
            layer.bias = layer.bias - (self.learning_rate/self.batch_size)*layer.changed_bias
            layer.changed_weight = np.zeros((self.num_node[i], self.num_node[i+1]),dtype = float)
            layer.changed_bias = np.zeros((1, self.num_node[i+1]),dtype = float)
            
            
            
            
    def forward_pass_testing(self,input):

        self.layers[0].input_activation = input
        
        for i in range(self.nlayer-1):

            first_term = np.dot(self.layers[i].input_activation,self.layers[i].weight)
            second_term = self.layers[i].bias
            curr_temp = np.add(first_term ,second_term)

            if(self.layers[i+1].activation_function=="Sigmoid"):
                self.layers[i+1].input_activation = self.sigmoid(curr_temp)
                
            else:
                self.layers[i+1].input_activation = self.softmax(curr_temp)
        
        
         
        
        
    def predict(self, filename, input):
       # self.batch_size = len(input)
        self.forward_pass_testing(input)
        a = self.layers[self.nlayer-1].input_activation
        predictions = []
        for i in range(len(a)):
            if(a[i][0]>a[i][1]):
                predictions.append(0)
            else:
                predictions.append(1)

        return np.asarray(predictions),a

    def check_accuracy(self, filename, inputs, labels):
        print("Testing Started")
        a, array = self.predict(filename,inputs)
        total=len(a)
        correct=0
        for i in range(len(a)):
            if (a[i] == labels[i]):
                correct += 1
        print("Accuracy: ", correct*100/total) 
            
            
             
            
            
        
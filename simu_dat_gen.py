#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:00:06 2020

@author: Brandon
"""

import numpy as np
import math
import pandas as pd


# Generating data function
#dat = dat_gen(N = 200, m = 100, T = 180, stdv = 0.05, theta_w = 0.02, stde = 0.05)

def dat_gen(N, m, T, stdv, theta_w, stde):
    rho = np.random.uniform(0.9, 1, m)
    c = np.zeros([N*T, m])
    
    for i in range(m): # for each covariate j 
        x = np.zeros([N,T])
        x[:,0] = np.random.normal(0, 1, N)
        for t in range(1,T):
            x[:, t] = rho[i] * x[:, t-1] + np.random.normal(0,math.sqrt(1- rho[i]**2), N)
        r = np.argsort(x, axis=0) # get the rank of elements in each column i.e. CSrank  AT: output 0 index 
        x1 = np.zeros([N, T]) # c_ij,t
        ridx = np.array([ range(0, N)]) + 1 # adjust 0 index to 1 index 
        for k in range(0, T): # at each time t 
                #         x1[:,k] = r[:,k]*2/(N+1) - 1
                x1[(r[:,k]), k] = ridx * 2 /(N+1) - 1 
                c[:,i] = x1.flatten('F') #c = (c_ijt) matrix , structure is like 200*1, 200*1, ..., 200*1 (180 times), (repeat m times)

#    per = np.tile(np.arange(N), T)
    time = np.tile(np.arange(T), N) # All times 36000

    vt = np.transpose(np.random.multivariate_normal((0,0,0),np.diag([1,1,1]), T))*stdv # python fill the matrix rowwise, matlab columnwise
    beta = c[:,[0,1,2]]
    betav = np.zeros([N*T])

# set_trace()

    for t in range(T):
        ind = (time == t)
        betav[ind] = np.dot(beta[ind,:], vt[:,t])  # beta * v

    # Simulate times series y_t = q*y_t-1 + u_t 
    y = np.zeros([T,1])
    y[0] = np.random.normal(0,1)

    q = 0.95 
    for t in range(1,T):
        y[t] = q * y[t-1] + np.random.normal(0,1)*math.sqrt(1-q**2)
    
    cy = c
    for t in range(T):
        ind = (time == T)
        cy[ind,:] = c[ind,:] * y[t] # z_i,t ???? where is the one t

# Epsilon 
    ep = np.random.standard_t(5, N*T)*stde



# Model 1 
    theta = np.array([1, 1]+ [0]* (m-2) + [0, 0, 1] + [0] * (m-3))*theta_w # for ease of calculation 

    # 36000 * 1
    r1 = np.dot(np.hstack((c,cy)), theta) + betav + ep
    
#    rt = np.dot(np.hstack((c,cy)), theta) 
    
    
    # Model 2
    z = np.hstack((c,cy))
    z[:,0] = c[:,0] **2 *2  # correspond to 0.04
    z[:,1] = c[:,0] * c[:,1] * 1.5 # correspond to 0.03
    z[:,m+3] = np.sign(cy[:,2]) * 0.6 # correspond to 0.012
    
    r1_m2 = np.dot(z, theta) + betav + ep 
#    rt_m2 = np.dot(z,theta)
    
    # Combine covariate and response 
    col_names = ['x' + str(x) for x in range(1,m+1)] + ['y']
    dat_m1 = pd.DataFrame(np.hstack((c,r1.reshape(-1,1))), columns = col_names)
    
    dat_m2 = pd.DataFrame(np.hstack((c,r1_m2.reshape(-1,1))), columns = col_names)
    # Equal Split to 3 pieces 
    #train_m1, vali_m1, test_m1 = np.split(dat_m1, 3)
    return([dat_m1, dat_m2])

# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:11:34 2020

@author: Bran
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:06:09 2020

@author: Brandon
"""

#%%
'''
Expanding window cross validation
Similar to sklearn format
@ germayne  
'''
import numpy as np
import pandas as pd
import pdb
#import multiprocessing
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA 
from functools import partial 
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.cross_decomposition import PLSRegression


#Ref 
# https://medium.com/eatpredlove/time-series-cross-validation-a-walk-forward-approach-in-python-8534dd1db51a


class expanding_window(object):
    '''	
    Parameters 
    ----------
    
    Note that if you define a horizon that is too far, then subsequently the split will ignore horizon length 
    such that there is validation data left. This similar to Prof Rob hyndman's TsCv  i.e. stop when there is no room for validation data
    
    
    initial: int
        initial train length 
    horizon: int 
        forecast(validation) horizon (forecast(Validation) length). Default = 1
    period: int 
        length of train data to add each iteration 
    
    Note 
    ---------
    1. Huber Loss param not tuned 
    2. Rolling procedure: Parallel ? -- now only the tuning within each roling is parallelized
    '''
    

    def __init__(self,initial= 1,horizon = 1,period = 1,test_p = 1):
        self.initial = initial
        self.horizon = horizon 
        self.period = period 
        self.test_p = test_p


    def split(self,data): # self.output_train, self.output_test, self.ourput_vali : lists containing the rollings 
        '''
        Parameters 
        ----------
        
        Data: Training data 
        
        Returns 
        -------
        train_index ,test_index: 
            index for train and valid set similar to sklearn model selection
        '''
        self.data = data
        self.counter = 0 # for us to iterate and track later 


        data_length = data.shape[0] # rows 
        self.n_col  = data.shape[1] # columns including y 
        data_index = list(np.arange(data_length))
         
        self.output_train = []
        self.output_vali = []
        self.output_test = []
        # append initial 
        self.output_train.append(list(np.arange(self.initial)))
        progress = [x for x in data_index if x not in list(np.arange(self.initial)) ] # indexes left to append to train 
        self.output_vali.append([x for x in data_index if x not in self.output_train[self.counter]][:self.horizon] ) #self.output_train is list of lists
        # check if the initial + horizon lengh exceed data length
        if len(progress) < self.horizon:
            return "Setup is not appropriate (1)"
        elif len(progress) == self.horizon:
            progress = []


        # clip initial indexes from progress since that is what we are left 
#        import pdb; pdb.set_trace()
#        while len(progress) != 0:
        while len(progress) >= (self.horizon + self.period + self.test_p):
            temp = progress[:self.period]
            to_add = self.output_train[self.counter] + temp
            # update the train index 
            self.output_train.append(to_add)
            # increment counter 
            self.counter +=1 
            # then we update the test index 
            
            to_add_test = [x for x in data_index if x not in self.output_train[self.counter] ][:self.horizon]
            self.output_vali.append(to_add_test)

            # update progress 
            progress = [x for x in data_index if x not in self.output_train[self.counter]]
        
        # Create the index for testing set 
        self.output_test = [[x[-1] + y for y in list(np.arange(1,self.test_p + 1))]for x in self.output_vali]
#        return [self.output_train, self.output_vali, self.output_test]

        # clip the last element of self.output_train and self.output_vali -- because the last one should not be included.
        # modified to be included in while len(progress) >= (self.horizon + self.period)
#        self.output_train = self.output_train[:-1]
#        self.output_vali = self.output_vali[:-1] 
        
        # mimic sklearn output 
#        index_output = [(train,test) for train,test in zip(self.output_train,self.output_vali)] #self.output_train and self.output_vali are lists of lists
#        return index_output
        # list for MSE 
    def pca_tune(self, n_cores, max_k): # Hyperparams : number of pc's
        if (max_k < 1) or (max_k > (self.n_col - 1)):
            return "max_k not appropriate"
            
        def pca_f(train_dat, vali_dat, k):
            pca = PCA()
            train_dat_X = pca.fit_transform(train_dat.drop('y', axis = 1))[:,:k]
            vali_dat_X = pca.fit_transform(vali_dat.drop('y', axis = 1))[:,:k]
          # Huber loss different from GKX and hyperparam not adjusted -- no use of Huber in GKX 
#            huber = HuberRegressor().fit(train_dat_X, train_dat['y'])
            reg = LinearRegression().fit(train_dat_X, train_dat['y'])
            # Mse on validation set 
            mse = mean_squared_error(reg.predict(vali_dat_X), vali_dat['y'])
            return mse 
        
        R_IS_l = []
        R_OOS_l =[]
        
        pp = Pool(n_cores)
        
        for train, vali, test in zip(self.output_train, self.output_vali, self.output_test):
            train_dat = self.data.loc[train,:]
            vali_dat = self.data.loc[vali,:]
            test_dat = self.data.loc[test,:]
            
#            pdb.set_trace()
            
            func = partial(pca_f, train_dat, vali_dat) # fix train_dat and vali_dat
            mse_l = pp.map(func, range(1,max_k+1,1))
            k_s = mse_l.index(min(mse_l)) + 1 # of PCs selected
            print(k_s)
#            pdb.set_trace()
            
            pca = PCA()
            test_dat_X = pca.fit_transform(test_dat.drop('y', axis = 1))[:,:k_s]
            train_dat_X = pca.fit_transform(train_dat.drop('y', axis = 1))[:,:k_s]
            # refit the train -- time inefficient
#            huber = HuberRegressor().fit(train_dat_X, train_dat['y']) 
            reg = LinearRegression().fit(train_dat_X, train_dat['y'])
#            R_OOS = huber.score(test_dat_X, test_dat['y']) # Notice that the denominator mean is not 0 , not as GKX
#             R_IS = huber.score(train_dat_X, train_dat['y'])
            # GKX R^2 
            R_OOS = 1 - np.sum((reg.predict(test_dat_X) - test_dat['y'])**2) / np.sum((test_dat['y']) ** 2)
            R_IS = 1 - np.sum((reg.predict(train_dat_X) - train_dat['y'])**2) / np.sum((train_dat['y']) ** 2)

            R_IS_l.append(R_IS)
            R_OOS_l.append(R_OOS)
        pp.close()
        pp.join()
        pp.clear()
        
        return(pd.DataFrame({'R_IS_l':R_IS_l, 'R_OOS_l':R_OOS_l}))
    def pls_tune(self, n_cores, max_k):
        if (max_k < 1) or (max_k > (self.n_col - 1)):
            return "max_k not appropriate"
        
        def pls_f(train_dat, vali_dat, k):
            pls = PLSRegression(n_components = k)
            pls.fit(train_dat.drop('y', axis = 1), train_dat['y'])
            # mse on validation set 
            mse = mean_squared_error(pls.predict(vali_dat.drop('y', axis = 1)), vali_dat['y'])
            return mse
        
        R_IS_l = []
        R_OOS_l =[]
        
        pp = Pool(n_cores)
        
        for train, vali, test in zip(self.output_train, self.output_vali, self.output_test):
            train_dat = self.data.loc[train,:]
            vali_dat = self.data.loc[vali,:]
            test_dat = self.data.loc[test,:]
            
#            pdb.set_trace()
            
            func = partial(pls_f, train_dat, vali_dat) # fix train_dat and vali_dat
            mse_l = pp.map(func, range(1,max_k+1,1))
            k_s = mse_l.index(min(mse_l)) + 1 # of PCs selected
            
#            pdb.set_trace()
            pls = PLSRegression(n_components = k_s)
            pls.fit(train_dat.drop('y', axis = 1), train_dat['y'])
            
            # GKX R^2 
            R_OOS = 1 - np.sum((pls.predict(test_dat.drop('y', axis = 1)) - test_dat['y'])**2) / np.sum((test_dat['y']) ** 2)
            R_IS = 1 - np.sum((pls.predict(train_dat_X) - train_dat['y'])**2) / np.sum((train_dat['y']) ** 2)

            R_IS_l.append(R_IS)
            R_OOS_l.append(R_OOS)
        pp.close()
        pp.join()
        pp.clear()
#    def Enet_tune(self, n_cores, rho_l, lambda_l):

        
        

        
#%%
#%%
        
#if __name__ == '__main__': 
# demo (Testing cases, need to convert to DataFrame with column names X1,...,Xn,y)
        
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]])

#y = np.array([1, 2, 3, 4, 5])

        
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
#
#y = np.array([1, 2, 3, 4, 5, 6])

        
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [5,6], [3,6], [3,4]])
#
#y = np.array([1, 2, 3, 4, 5, 6, 7, 8,9])
#
#tscv = expanding_window(initial= 3,horizon = 2,period = 2)
#for train_index, test_index in tscv.split(X):
#    print(train_index)
#    print(test_index)
#

#%%
#X = np.random.randint(0,1000,size = (120,2))
#y = np.random.randint(0,1000,size = (120,1))
#
#tscv = expanding_window(initial = 36, horizon = 24,period = 1)
#for train_index, test_index in tscv.split(X):
#    print(train_index)
#    print(test_index)



#%%
# view rawexpanding_window.py hosted with ❤ by GitHub
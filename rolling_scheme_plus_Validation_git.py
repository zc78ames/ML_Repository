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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from functools import partial 
#from pathos.multiprocessing import ProcessingPool as Pool
#from pathos.pools import ProcessPool as Pool 
from pathos.pools import ThreadPool as Pool 


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
    3. # Nested parallel: Not enough space if the data is too big.
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
    def pca_tune(self, max_k, pp): # Hyperparams : number of pc's
        if (max_k < 1) or (max_k > (self.n_col - 1)):
            return "max_k not appropriate"
            
        def pca_f(train_dat, vali_dat, train_y_mean, k):
            pca = PCA(n_components = k)
            # pc direction 
#            train_dat_X = pca.fit_transform(train_dat.drop('y', axis = 1))[:,:k]
#            vali_dat_X = pca.fit_transform(vali_dat.drop('y', axis = 1))[:,:k]
            pca.fit(train_dat.drop('y', axis = 1))
            train_dat_X = pca.transform(train_dat.drop('y', axis = 1)) # i.e. train_dat_pc
            vali_dat_X = pca.transform(vali_dat.drop('y', axis = 1))
          # Huber loss different from GKX and hyperparam not adjusted -- no use of Huber in GKX 
#            huber = HuberRegressor().fit(train_dat_X, train_dat['y'])
            reg = LinearRegression().fit(train_dat_X, train_dat['y'])
            # Mse on validation set . AT: vali_dat and test_dat not demeaned
            mse = mean_squared_error((reg.predict(vali_dat_X) + train_y_mean), vali_dat['y'])
#            R_square = 1 - np.sum(((reg.predict(vali_dat_X) + train_y_mean) - vali_dat['y'])**2) / np.sum((vali_dat['y'] - train_y_mean) ** 2)

            return mse 
#            return R_square
        
        
        
        for train, vali, test in zip(self.output_train, self.output_vali, self.output_test):
            train_dat = self.data.loc[train,:]
            vali_dat = self.data.loc[vali,:]
            test_dat = self.data.loc[test,:]
            
            # Standardize train_X and center train_y and store the info
            scaler_X = StandardScaler()
            scaler_X.fit(train_dat.drop('y', axis = 1))
            scaler_y = StandardScaler(with_std = False)
            scaler_y.fit(np.array(train_dat[['y']]))
             
            # Transformed data using mean and std from training data
            train_dat_t = pd.DataFrame(scaler_X.transform(train_dat.drop('y', axis = 1)),columns = train_dat.columns[:-1])
            train_dat_t['y'] = scaler_y.transform(np.array(train_dat['y']).reshape(-1,1)) # Do not demean on vali_dat and test_dat
            
            vali_dat_t = pd.DataFrame(scaler_X.transform(vali_dat.drop('y', axis = 1)),columns = vali_dat.columns[:-1])
            vali_dat_t['y'] = np.array(vali_dat['y'])

 
            test_dat_t = pd.DataFrame(scaler_X.transform(test_dat.drop('y', axis = 1)),columns = test_dat.columns[:-1])
            test_dat_t['y'] = np.array(test_dat['y'])
            
            train_y_mean = scaler_y.mean_
            
            func = partial(pca_f, train_dat_t, vali_dat_t, train_y_mean) # fix train_dat and vali_dat
            mse_l = pp.map(func, range(1,max_k+1,1))
            k_s = mse_l.index(min(mse_l)) + 1 # of PCs selected
#            k_s = mse_l.index(max(mse_l)) + 1 # of PCs selected

            print(k_s)
#            pdb.set_trace()
            
            pca = PCA(n_components = k_s)
            # pc direction 
            pca.fit(train_dat_t.drop('y', axis = 1))
           # refit the train -- time inefficient
            train_dat_pc = pca.transform(train_dat_t.drop('y', axis = 1))
            test_dat_pc = pca.transform(test_dat_t.drop('y', axis = 1))
          # Huber loss different from GKX and hyperparam not adjusted -- no use of Huber in GKX 
#            huber = HuberRegressor().fit(train_dat_X, train_dat['y'])
            reg = LinearRegression().fit(train_dat_pc, train_dat_t['y'])
#            R_OOS = huber.score(test_dat_X, test_dat['y']) # Notice that the denominator mean is not 0 , not as GKX
#             R_IS = huber.score(train_dat_X, train_dat['y'])
#            # GKX R^2 
            R_OOS = 1 - np.sum(((reg.predict(test_dat_pc) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y']) ** 2)
            R_IS = 1 - np.sum(((reg.predict(train_dat_pc) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y']) ** 2)
#            simu R^2
#            R_OOS = 1 - np.sum(((reg.predict(test_dat_pc) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y'] - train_y_mean) ** 2)
#            R_IS = 1 - np.sum(((reg.predict(train_dat_pc) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y'] - train_y_mean) ** 2)
        
        return([R_IS, R_OOS])
    def pls_tune(self, max_k, pp): # Hyperparams : number of pc's
        if (max_k < 1) or (max_k > (self.n_col - 1)):
            return "max_k not appropriate"
            
        def pls_f(train_dat, vali_dat, train_y_mean, k):
            pls = PLSRegression(n_components = k)
            # pc direction 
#            train_dat_X = pca.fit_transform(train_dat.drop('y', axis = 1))[:,:k]
#            vali_dat_X = pca.fit_transform(vali_dat.drop('y', axis = 1))[:,:k]
            pls.fit(train_dat.drop('y', axis = 1), train_dat['y'])
            # Mse on validation set . AT: vali_dat and test_dat not demeaned
            mse = mean_squared_error((pls.predict(vali_dat.drop('y', axis = 1)) + train_y_mean), vali_dat['y'])
#            R_square = 1 - np.sum(((reg.predict(vali_dat_X) + train_y_mean) - vali_dat['y'])**2) / np.sum((vali_dat['y'] - train_y_mean) ** 2)

            return mse 
#            return R_square
        
        
        
        for train, vali, test in zip(self.output_train, self.output_vali, self.output_test):
            train_dat = self.data.loc[train,:]
            vali_dat = self.data.loc[vali,:]
            test_dat = self.data.loc[test,:]
            
            # Standardize train_X and center train_y and store the info
            scaler_X = StandardScaler()
            scaler_X.fit(train_dat.drop('y', axis = 1))
            scaler_y = StandardScaler(with_std = False)
            scaler_y.fit(np.array(train_dat[['y']]))
             
            # Transformed data using mean and std from training data
            train_dat_t = pd.DataFrame(scaler_X.transform(train_dat.drop('y', axis = 1)),columns = train_dat.columns[:-1])
            train_dat_t['y'] = scaler_y.transform(np.array(train_dat['y']).reshape(-1,1)) # Do not demean on vali_dat and test_dat
            
            vali_dat_t = pd.DataFrame(scaler_X.transform(vali_dat.drop('y', axis = 1)),columns = vali_dat.columns[:-1])
            vali_dat_t['y'] = np.array(vali_dat['y'])

 
            test_dat_t = pd.DataFrame(scaler_X.transform(test_dat.drop('y', axis = 1)),columns = test_dat.columns[:-1])
            test_dat_t['y'] = np.array(test_dat['y'])
            
            train_y_mean = scaler_y.mean_
#            pdb.set_trace()
            
            func = partial(pls_f, train_dat_t, vali_dat_t, train_y_mean) # fix train_dat and vali_dat
            mse_l = pp.map(func, range(1,max_k+1,1))
            k_s = mse_l.index(min(mse_l)) + 1 # of PCs selected
#            k_s = mse_l.index(max(mse_l)) + 1 # of PCs selected

            print(k_s)
#            pdb.set_trace()
            
            
            pls = PLSRegression(n_components = k_s)
            # pc direction 
            pls.fit(train_dat_t.drop('y', axis = 1), train_dat_t['y'])
#            # GKX R^2 
            R_OOS = 1 - np.sum(((pls.predict(test_dat_t.drop('y', axis =1)) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y']) ** 2)
            R_IS = 1 - np.sum(((pls.predict(train_dat_t.drop('y', axis =1)) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y']) ** 2)
#            simu R^2
#            R_OOS = 1 - np.sum(((reg.predict(test_dat_pc) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y'] - train_y_mean) ** 2)
#            R_IS = 1 - np.sum(((reg.predict(train_dat_pc) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y'] - train_y_mean) ** 2)
        
        return([R_IS, R_OOS])
    def Enet_tune(self, lambda_v_l, pp): # rho is taken as 0.5
            
        def Enet_f(train_dat, vali_dat, train_y_mean, lambda_v):
            ereg = ElasticNet(alpha = lambda_v)
            ereg.fit(train_dat.drop('y', axis = 1), train_dat['y'])
            # Mse on validation set . AT: vali_dat and test_dat not demeaned
            mse = mean_squared_error((ereg.predict(vali_dat.drop('y', axis = 1)) + train_y_mean), vali_dat['y'])
#            R_square = 1 - np.sum(((ereg.predict(vali_dat_X) + train_y_mean) - vali_dat['y'])**2) / np.sum((vali_dat['y'] - train_y_mean) ** 2)

            return mse 
#            return R_square
        
        
        
        for train, vali, test in zip(self.output_train, self.output_vali, self.output_test):
            train_dat = self.data.loc[train,:]
            vali_dat = self.data.loc[vali,:]
            test_dat = self.data.loc[test,:]
            
            # Standardize train_X and center train_y and store the info
            scaler_X = StandardScaler()
            scaler_X.fit(train_dat.drop('y', axis = 1))
            scaler_y = StandardScaler(with_std = False)
            scaler_y.fit(np.array(train_dat[['y']]))
             
            # Transformed data using mean and std from training data
            train_dat_t = pd.DataFrame(scaler_X.transform(train_dat.drop('y', axis = 1)),columns = train_dat.columns[:-1])
            train_dat_t['y'] = scaler_y.transform(np.array(train_dat['y']).reshape(-1,1)) # Do not demean on vali_dat and test_dat
            
            vali_dat_t = pd.DataFrame(scaler_X.transform(vali_dat.drop('y', axis = 1)),columns = vali_dat.columns[:-1])
            vali_dat_t['y'] = np.array(vali_dat['y'])

 
            test_dat_t = pd.DataFrame(scaler_X.transform(test_dat.drop('y', axis = 1)),columns = test_dat.columns[:-1])
            test_dat_t['y'] = np.array(test_dat['y'])
            
            train_y_mean = scaler_y.mean_
#            pdb.set_trace()
            
            func = partial(Enet_f, train_dat_t, vali_dat_t, train_y_mean) # fix train_dat and vali_dat
            mse_l = pp.map(func, lambda_v_l)
            lambda_v = lambda_v_l[mse_l.index(min(mse_l))] # lambda_v selected

            print(lambda_v)
#            pdb.set_trace()
            
            
            ereg = ElasticNet(alpha = lambda_v)
            ereg.fit(train_dat_t.drop('y', axis = 1), train_dat_t['y'])
#            # GKX R^2 
            R_OOS = 1 - np.sum(((ereg.predict(test_dat_t.drop('y', axis =1)) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y']) ** 2)
            R_IS = 1 - np.sum(((ereg.predict(train_dat_t.drop('y', axis =1)) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y']) ** 2)
#            simu R^2
#            R_OOS = 1 - np.sum(((ereg.predict(test_dat_pc) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y'] - train_y_mean) ** 2)
#            R_IS = 1 - np.sum(((ereg.predict(train_dat_pc) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y'] - train_y_mean) ** 2)

        return([R_IS, R_OOS])

#    def Enet_tune_H(self, n_cores, rho_l, lambda_l):        



    def RF_tune(self, max_depth_l, n_tree, pp):
            
        def RF_f(train_dat, vali_dat, n_tree, max_depth): # notice arguments order 
            rfreg = RandomForestRegressor(max_depth = max_depth, n_estimators = n_tree)
            rfreg.fit(train_dat.drop('y', axis = 1), train_dat['y'])
            # Mse on validation set . AT: vali_dat and test_dat not demeaned
            mse = mean_squared_error((rfreg.predict(vali_dat.drop('y', axis = 1))), vali_dat['y'])
#            R_square = 1 - np.sum(((ereg.predict(vali_dat_X) + train_y_mean) - vali_dat['y'])**2) / np.sum((vali_dat['y'] - train_y_mean) ** 2)

            return mse 
#            return R_square
        
        
#        pp = Pool(n_cores)
        
        for train, vali, test in zip(self.output_train, self.output_vali, self.output_test):
            train_dat = self.data.loc[train,:]
            vali_dat = self.data.loc[vali,:]
            test_dat = self.data.loc[test,:]
            
#            pdb.set_trace()
            
            func = partial(RF_f, train_dat, vali_dat, n_tree) # fix train_dat and vali_dat
            mse_l = pp.map(func, max_depth_l)
            max_depth_s = max_depth_l[mse_l.index(min(mse_l))] # max_depth selected

            print(max_depth_s)
#            pdb.set_trace()
            
            
            rfreg = RandomForestRegressor(max_depth = max_depth_s, n_estimators = n_tree)

            rfreg.fit(train_dat.drop('y', axis = 1), train_dat['y'])
#            # GKX R^2 
            R_OOS = 1 - np.sum(((rfreg.predict(test_dat.drop('y', axis =1))) - test_dat['y'])**2) / np.sum((test_dat['y']) ** 2)
            R_IS = 1 - np.sum(((rfreg.predict(train_dat.drop('y', axis =1))) - train_dat['y'])**2) / np.sum((train_dat['y']) ** 2)
#            simu R^2
#            R_OOS = 1 - np.sum(((ereg.predict(test_dat_pc) + train_y_mean) - test_dat_t['y'])**2) / np.sum((test_dat_t['y'] - train_y_mean) ** 2)
#            R_IS = 1 - np.sum(((ereg.predict(train_dat_pc) + train_y_mean) - train_dat_t['y'])**2) / np.sum((train_dat_t['y'] - train_y_mean) ** 2)

        return([R_IS, R_OOS])
        
        

        
#%% Main.py 

#from rolling_scheme_plus_Validation import expanding_window
#import time 
##from pathos.pools import ProcessPool as Pool 
#from pathos.pools import ThreadPool as Pool   # Faster for small data, but could be slower for large data ? 
##from pathos.pools import ParallelPool as Pool # Fastest for small data -> can be slow on huge data 
#
## Nested parallel: Not enough space if the data is too big 
#start = time.time()
#result_pca = []
#result_pls = []
#result_Enet = []
#result_RF = []
#
#
#pp = Pool(4) #n_cores
#for _ in range(4): # of repeated iterations 
#    
#
#    dat_simu = dat_gen(N = 200, m = 100, T = 180, stdv = 0.05, theta_w = 0.02, stde = 0.05)
#    dat_m1 = dat_simu[0]
#
#    # Rolling procedure: T = 180 (if taken as monthly data), N = 200 
#    tscv = expanding_window(initial = 5*12*200 , horizon = 5*12*200, period = 1, test_p = 5*12*200)
#    #tscv.split(pd.concat([train_m1, vali_m1]))
#    tscv.split(dat_m1)
#
#
###    PCA
##    result_pca.append(tscv.pca_tune(30,pp)) # max PCs
##    result_pca_df = pd.DataFrame(result_pca, columns = ['R_IS', 'R_OOS'])
##    
##    
###    PLS  
##    result_pls.append(tscv.pls_tune(30,pp)) # n_cores and max PCs 
##    result_pls_df = pd.DataFrame(result_pls, columns = ['R_IS', 'R_OOS'])
##
##
###    Enet without Huber Loss 
##    result_Enet.append(tscv.Enet_tune([1e-4, 1e-3, 1e-2, 1e-1], pp))
##    result_Enet_df = pd.DataFrame(result_Enet, columns = ['R_IS', 'R_OOS'])
#
##   Random Forest  n_tree = 300
#    result_RF.append(tscv.RF_tune(list(range(1,7,1)), 300, pp))
#    result_RF_df = pd.DataFrame(result_RF, columns = ['R_IS', 'R_OOS'])
#    
#    
#pp.close()
#pp.join()
#pp.clear()
#end = time.time()
#print(end - start)

        
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


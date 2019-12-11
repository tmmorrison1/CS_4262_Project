#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:26:41 2019

@author: murraymorrison
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 


import pandas as pd
import numpy as np



#data = data values of game info
#num_components = number of principle components
#returns new data frame with principle components of the team data extracted
def run_pca(data, num_components):
    
    
    
    data_vals = data
    
    target_index = len(data_vals[0])-2
    
    
    
    team_1 = fetch_team_stats(data_vals[:,1])
    team_2 = fetch_team_stats(data_vals[:,2])
    
    x_data = np.concatenate((team_1, team_2), axis=1) 
    y_data = data_vals[:,target_index].astype("int")
    
    
    y_df = pd.DataFrame(data = y_data)
 
    x_data = StandardScaler().fit_transform(x_data)
    
    lol_pca = PCA(n_components=num_components)
    
    principal_components = lol_pca.fit_transform(x_data)
    
    
    
    principal_df = pd.DataFrame(data = principal_components)
    
    finalDf = pd.concat([principal_df, y_df], axis = 1)
    
    
    for i in range(1,1+ num_components):
        print('Principle Component ' + str(i) + ' explains ' + str(lol_pca.explained_variance_ratio_[i-1])+' of the total variance')
    
    print() 
    print('Total Explained Variance: ' + str(sum(lol_pca.explained_variance_ratio_))+' for '+str(num_components)+' principle components')
    
    #return finalDf
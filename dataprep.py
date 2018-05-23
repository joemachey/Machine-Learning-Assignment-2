# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:11:44 2018

@author: joemachey
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

folder_path = 'C:\\Users\\joemachey\\Desktop\\Space Exploration\\University of Sydney\\Machine Learning and Data Mining\\Assignment 2\\Assignment 2'
file_name = 'covtype.csv'



def Prepare_Data(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    # Open File, Split data and convert to numpy arrays
    file = pd.read_csv('covtype.csv')
    training_X = np.array(file.iloc[:,0:54])
    training_Y = np.array(file['Cover_Type'])
    training_X_columns = list(file.columns.values[0:54])
    del file
    return training_X, training_Y, training_X_columns

'''

Open up the data and split it into three variables. Stored as numpy variables
as this will be easier when using the classifier 

training_X = Contains the training data for the classifier, will later be split 
             up into train and test data. 
training_Y = True class labels of the data points
training_X_columns = The names of the training_X columns

'''
training_X, training_Y, training_X_columns = Prepare_Data(folder_path, file_name)

# Save the data for use later on 
np.save(os.path.join(folder_path, 'training_X.npy'), training_X)
np.save(os.path.join(folder_path, 'training_Y.npy'), training_X)
np.save(os.path.join(folder_path, 'training_X_columns.npy'), training_X_columns)


#%%
# Distributon of Classes 
cover_type = ['Spruce/Fur_1', 'Lodgepole Pine_2', 'Ponderosa Pine_3', 'Cottonwood/Willow_4', 'Aspen_5', 'Douglas-fir_6', 'Krummholz_7']
unique, counts = np.unique(training_Y, return_counts = True)
Total = dict(zip(unique, counts))

plt.bar(unique, counts)
plt.ylabel('#Occurences of class/Tree/Cover Type')
plt.title('Class Distribution')
locs = [1, 2, 3, 4, 5, 6, 7]
plt.xticks(locs, cover_type)

'''
You can see that the class distribution is far from uniform. This will probably
introduce bias into the accurcy results.
'''

#%%  EXPLORATORY ANALYSIS
''' Its easier to use panda data frame for the following. '''

explore_file = pd.read_csv('covtype.csv')

# Name and class of land cover
spruce_1 = explore_file[explore_file.Cover_Type==1]
lodgepole_2 = explore_file[explore_file.Cover_Type==2]
ponderosa_3 = explore_file[explore_file.Cover_Type==3]
willow_4 = explore_file[explore_file.Cover_Type==4]
aspen_5 = explore_file[explore_file.Cover_Type==5]
douglas_6 =  explore_file[explore_file.Cover_Type==6]
krummholz_7 = explore_file[explore_file.Cover_Type==7]

''' In the following just change the variable names in data below '''

plt.figure()
plt.title('Elevation of Cover Types') # Change accordingly 
plt.ylabel('Elevation [m]') # Change accordingly 

# Change the .Elevation below for the variable you want to observe. Names can 
#found in the trinaing_X_columns variable
data = [spruce_1.Elevation, lodgepole_2.Elevation, ponderosa_3.Elevation,\
        willow_4.Elevation, aspen_5.Elevation, douglas_6.Elevation, krummholz_7.Elevation]
plt.xticks(locs, cover_type)
plt.boxplot(data)
plt.show()















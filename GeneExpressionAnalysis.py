import  pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats


# Root Path and Names of Individual File Names
rootPath='E:/BioInformatics'
trainFileName='pp5i_train.gr.csv'
trainFileClassName='pp5i_train_class.txt'
testFileName='pp5i_test.gr.csv'

#Creating absolute Path from the relative Path
trainFilePath=rootPath+'/'+trainFileName
trainFileClassPath=rootPath+'/'+trainFileClassName
testFilePath=rootPath+'/'+testFileName

#Reading values in pandas data frame
pdTrainFileValues=pd.read_csv(trainFilePath)
pdTrainFileClassValues=pd.read_csv(trainFileClassPath,header=None,sep=" ")
pdTestFileValues=pd.read_csv(testFilePath)

pdTestFileValues.shape
pdTrainFileClassValues=pdTrainFileClassValues[1:]

pdTrainFileValues.shape

pdTrainFileValues=pdTrainFileValues.drop(["SNO"],axis=1)
pdTrainFileValues.shape
pdTrainFileValues.columns






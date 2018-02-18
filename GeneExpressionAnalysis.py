import  pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from scipy.cluster.hierarchy import  linkage,dendrogram
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib


def saveTrainedModel(model,path):
    print("model:",model)
    print("path:",path)
    joblib.dump(model,path)

def loadTrainedBasLineModel(path):
    return joblib.load(path)



# Root Path and Names of Individual File Names
rootPath='E:/BioInformatics'
trainFileName='pp5i_train.gr.csv'
trainFileClassName='pp5i_train_class.txt'
testFileName='pp5i_test.gr.csv'
trainedModelPath=rootPath+"/"+"TrainedModels"

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

#Displaying boxplot of Data



pdTrainOrderByGene=pdTrainFileValues.transpose()
pdTrainOrderByGeneSample=pdTrainOrderByGene.iloc[:,0:50]

figRawGenes = plt.figure(1)
figRawGenes.suptitle('BoxPlot', fontsize=14, fontweight='bold')
axRawGenes = figRawGenes.add_subplot(111)
axRawGenes.boxplot(pdTrainOrderByGeneSample)
axRawGenes.set_title('Gene Expression Dataset')
axRawGenes.set_xlabel('Genes 1 to 50')
axRawGenes.set_ylabel('Gene Expression Values')

figRawGenes.savefig("BoxPlot Raw Genes(1-50) Expression Data.png")

pdTrainFileValues=pdTrainFileValues.transpose()
pdTestFileValues=pdTestFileValues.drop(["SNO"],axis=1)
pdTestFileValues=pdTestFileValues.transpose()
pdTrainFileValues.shape,pdTrainFileClassValues.shape,pdTestFileValues.shape



# Transformation Normalization

pdTrainOrderByGeneNormalized=stats.zscore(pdTrainOrderByGene)


pdTrainOrderByGeneNormalizedSample=pdTrainOrderByGeneNormalized[:,0:50]
figNormalizedGenes = plt.figure(2)
figNormalizedGenes.suptitle('BoxPlot', fontsize=14, fontweight='bold')
axNormalizedGenes = figNormalizedGenes.add_subplot(111)
axNormalizedGenes.boxplot(pdTrainOrderByGeneNormalizedSample)
axNormalizedGenes.set_title('Normalized Gene Expression Dataset')
axNormalizedGenes.set_xlabel('Genes 1 to 50')
axNormalizedGenes.set_ylabel('Normalized Gene Expression Values')
figNormalizedGenes.savefig("BoxPlot Normalized Genes(1-50) Expression Data.png")



# Encoding Train Class label Data to Numeric

le=preprocessing.LabelEncoder()
le.fit(pdTrainFileClassValues)
trainClassNumericalValues=le.transform(pdTrainFileClassValues)


# Plotting Data using Hierarchical Clustering

mergings=linkage(pdTrainFileValues,method="complete")
dendrogram(mergings)


#Basline Classifier for Gene Expression Dataset

X,y=pdTrainOrderByGeneNormalized,trainClassNumericalValues
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)

svmClassifier = SVC()
baselineSVMTrainedModel=svmClassifier.fit(X_train,Y_train)
print("Basline Accuracy of SVM Classifier on Whole Gene Expression Dataset:",baselineSVMTrainedModel.score(X_test,Y_test))

absoluteBaselineTrainedModelPath=trainedModelPath+"/baselineSVMTrainedModel.pkl"
saveTrainedModel(baselineSVMTrainedModel,absoluteBaselineTrainedModelPath)


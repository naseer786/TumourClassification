import  pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from scipy.cluster.hierarchy import  linkage,dendrogram
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
import pickle
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn import cross_validation

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
pdTrainFileValuesWithGeneColumn=pd.read_csv(trainFilePath)
pdTrainFileClassValues=pd.read_csv(trainFileClassPath,header=None,sep=" ")
pdTestFileValues=pd.read_csv(testFilePath)

pdGeneNames=pdTrainFileValuesWithGeneColumn['SNO']
pdTrainFileValues=pdTrainFileValuesWithGeneColumn.drop(["SNO"],axis=1)
#pdTrainFileClassValues=pdTrainFileValuesWithGeneColumn[1:]
pdTrainOrderByGene=pdTrainFileValues.transpose()
pdTrainOrderByGeneSample=pdTrainOrderByGene.iloc[:,0:50]

pdTestFileValues.shape

pdTrainFileValues.shape

pdTrainFileValues.shape
pdTrainFileValues.columns

#Displaying boxplot of Data





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

svmClassifier = SVC()
X,y=pdTrainOrderByGeneNormalized,trainClassNumericalValues
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.5)

svmClassifier.fit(X_train,Y_train)

scores=cross_validation.cross_val_score(svmClassifier,X,y,cv=5)

baselineSVMTrainedModel = svmClassifier.fit(X_train, Y_train)
SPLITS=10
baseStratifiedKFold=StratifiedKFold(n_splits=SPLITS)
baseStratifiedKFold.get_n_splits(X,y)
stratifiedKCrossAccuracy=np.zeros(SPLITS)
index=0
for train_index,test_index in baseStratifiedKFold.split(X,y):
    X_train,X_test=X[train_index],X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    svmClassifier = SVC()
    baselineSVMTrainedModel=svmClassifier.fit(X_train,Y_train)
    stratifiedKCrossAccuracy[index]=baselineSVMTrainedModel.score(X_test,Y_test)
    print("Basline Accuracy of SVM Classifier each stratum:",baselineSVMTrainedModel.score(X_test,Y_test))
    index+=1

#Saving Baseline classifier model in the disk using sklean.joblib
absoluteBaselineTrainedModelPath=trainedModelPath+"/baselineSVMTrainedModel.pkl"
saveTrainedModel(baselineSVMTrainedModel,absoluteBaselineTrainedModelPath)

# Recursive Feature Selection

k=50
for i in range(100):
    recursiveFeatureSelection=RFE(estimator=SVR,n_features_to_select=k,step=0.2,verbose=0)
    recursiveFeatureSelection.fit(X,y)
    recursiveReducedX=recursiveFeatureSelection.transform(X)
    recReduced_Xtrain,recReduced_Xtest,recReduced_YTrain,recReduced_YTest=train_test_split(recursiveReducedX,y,test_size=0.2)
    svmClassifier.fit(recReduced_Xtrain,recReduced_YTrain)
    print("Features Selected:", k)
    print("train_accuracy:",svmClassifier.score(recReduced_Xtrain,recReduced_YTrain))
    print("test_accuracy:",svmClassifier.score(recReduced_Xtest,recReduced_YTest))
    print("------------------------------------------------------------------------")
    #  svmClassifier.score(recReduced_Xtrain,recReduced_YTrain)
    # svmClassifier.score(recReduced_Xtest,recReduced_YTest)
    k+=10



# Apply Kmeans Clustering
Initial_Clusters=1000
Final_Clusters=50
Decreate_Rate=0.1
GeneDataForClustering=X.transpose()
Threshold=0.30


#while Initial_Clusters>Final_Clusters:
def joinGenesWithClusters(geneDataClusters,totalClusters):
    dicOfGenesInCluster={}
    for i in range(totalClusters):
        dicOfGenesInCluster[i]=[]
    totalGenes=len(geneDataClusters)
    for i in range(totalGenes):
       dicOfGenesInCluster[geneDataClusters[i]].append(i)
    return dicOfGenesInCluster


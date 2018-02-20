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
from sklearn import K
from  sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import LeaveOneOut
from genetic_selection import GeneticSelectionCV



def saveTrainedModel(model,path):
    print("model:",model)
    print("path:",path)
    joblib.dump(model,path)

def loadTrainedBasLineModel(path):
    return joblib.load(path)

def selectKBestWithFeaturesWithIndices(features,k):
    dictionaryOfKBestFeaturesWithIndices={}
    indices=np.argpartition(features,-k)[-k:]
    values=features[indices]
    for index in range(len(indices)):
        key=indices[index]
        val=values[index]
        dictionaryOfKBestFeaturesWithIndices[key]=val
    return dictionaryOfKBestFeaturesWithIndices

def getMinAndMaxOfDictionary(dic):
    min=1
    max=0
    for key,value in dic.items():
        if min>dic[key]:
            min=dic[key]
        if max <dic[key]:
            max=dic[key]
    return min,max

def convertGeneKeyScoreToGeneIndexDic(dictOfKBestFeatures):
    n=len(dictOfKBestFeatures)
    dicOfGenesIndex={}
    keys=sorted(dictOfKBestFeatures.keys())
    count=0
    for index in range(len(keys)):
        dicOfGenesIndex[index]=keys[index]
        count+=1
    return dicOfGenesIndex

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



#Feature Selection Using Information Gain
X,y=pdTrainOrderByGeneNormalized,trainClassNumericalValues

FeaturesToBeSelected=4000
selectKBest=SelectKBest(mutual_info_classif, k=FeaturesToBeSelected).fit(X,y)
featureScores=selectKBest.scores_
dictOfKBestFeatures=selectKBestWithFeaturesWithIndices(featureScores,FeaturesToBeSelected)
sortedKeys=sorted(dictOfKBestFeatures.keys())
kBestFeaturesData=selectKBest.transform(X)
selectedGenesNamesUsingKBest=list(pdGeneNames[sortedKeys])
dicOfGenesIndices=convertGeneKeyScoreToGeneIndexDic(dictOfKBestFeatures)

mergings=linkage(kBestFeaturesData.transpose(),method="complete")
dendrogram(mergings,leaf_rotation=90.,leaf_font_size=12.,labels=selectedGenesNamesUsingKBest)



#Using Leave One out for checking accuracy
featureScoreUsingInformationGain=[]
leaveOneOut=LeaveOneOut()
for train_index,test_index in leaveOneOut.split(kBestFeaturesData):
    svmClassifier = SVC()
    XTrain,XTest=kBestFeaturesData[train_index],kBestFeaturesData[test_index]
    YTrain,YTest=y[train_index],y[test_index]
    svmClassifier.fit(XTrain,YTrain)
    featureScoreUsingInformationGain.append(svmClassifier.score(XTest,YTest))

#Basline Classifier for Gene Expression Dataset

X_train,X_test,Y_train,Y_test=train_test_split(selectedKBestFeatures,y,test_size=0.5)

svmClassifier.fit(X_train,Y_train)

score=svmClassifier.score(X_test,Y_test)
print(scores.mean())

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






#while Initial_Clusters>Final_Clusters:
def joinGenesWithClusters(geneDataClusters,totalClusters,dicOfGenesIndices):
    dicOfGenesInCluster={}
    for i in range(totalClusters):
        dicOfGenesInCluster[i]=[]
    totalGenes=len(geneDataClusters)
    for i in range(totalGenes):
       dicOfGenesInCluster[geneDataClusters[i]].append(dicOfGenesIndices[i])
    return dicOfGenesInCluster

def clusterScoreOfGenes(geneClusterData,totalClusters):
    clusterScoreDic={}
    for key,value in geneClusterData.items():
        geneIndices=geneClusterData[key]
        extracedGenesData=X[:,geneIndices]
        svmClusterClassifier=SVC()
        xTrain,xTest,yTrain,yTest=train_test_split(extracedGenesData,y,test_size=0.4)
        svmClusterClassifier.fit(xTrain,yTrain)
        clusterScoreDic[key]=svmClusterClassifier.score(xTest,yTest)
    return clusterScoreDic

def filterClustersWithThreshold(clusterScoreData,threshold):
    filteredClusters={}
    for key,value in clusterScoreData.items():
        if clusterScoreData[key] > threshold:
            filteredClusters[key]=value
    return filteredClusters

def combinedGenesUsingClusters(filteredClusters,dicOfGenesWithClusters):
    dicOfGenesIndicesTemp={}
    listOfGenes=[]
    for key,value in filteredClusters.items():
        tempGeneList=dicOfGenesWithClusters[key]
        for gene in tempGeneList:
            listOfGenes.append(gene)
        listOfGenes=sorted(listOfGenes)
    for index in range(len(listOfGenes)):
        dicOfGenesIndicesTemp[index]=listOfGenes[index]
    return X[:,listOfGenes],listOfGenes,dicOfGenesIndicesTemp



# Apply Kmeans Clustering
Initial_Clusters=600
Final_Clusters=10
Decreate_Rate=0.03
GeneDataForClustering=kBestFeaturesData.transpose()
Threshold=0.7
dicOfGenesIndices=convertGeneKeyScoreToGeneIndexDic(dictOfKBestFeatures)


count=1
samples=kBestFeaturesData.shape[0]
firstDim = GeneDataForClustering.shape[0]
while (Initial_Clusters>Final_Clusters) and (firstDim>Initial_Clusters)  :
        print("Iteration.....:",count)
        print("Initial Clusters...:",Initial_Clusters)
        print("Final Clusters...:", Final_Clusters)
        print("Data Shape...:",GeneDataForClustering.shape)
        kmeansCluster=KMeans(n_clusters=Initial_Clusters)
        kmeansCluster.fit(GeneDataForClustering)
        dicOfGenesWithClusters=joinGenesWithClusters(kmeansCluster.labels_,Initial_Clusters,dicOfGenesIndices)
        clusterScore=clusterScoreOfGenes(dicOfGenesWithClusters,Initial_Clusters)
        filteredClusters=filterClustersWithThreshold(clusterScore,Threshold)
        GeneDataForClustering,combineGenes,dicOfGenesIndices=combinedGenesUsingClusters(filteredClusters,dicOfGenesWithClusters)
        GeneDataForClustering=GeneDataForClustering.transpose()

        Initial_Clusters=int(Initial_Clusters-Decreate_Rate*Initial_Clusters)
        count+=1
        samples = GeneDataForClustering.shape[1]
        firstDim=GeneDataForClustering.shape[0]
        print("___________________________________________________________")


GeneDataForClustering=GeneDataForClustering.transpose()
finalXTrain,finalXTest,finalYTrain,finalYTest=train_test_split(GeneDataForClustering,y,test_size=0.4)
svmClassifier.fit(finalXTrain,finalYTrain)
print(svmClassifier.score(finalXTest,finalYTest))





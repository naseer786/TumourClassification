import  pandas as pd
import  numpy
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
from sklearn.cluster import KMeans
from sklearn import cross_validation
from  sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
import skrebate
import csv




def makePlotOfClassValues(listOfClassValues,plotName):
    dicOfClassLabels = {}
    if isinstance(listOfClassValues,numpy.ndarray):
        for index in range(len(listOfClassValues)):
            val=listOfClassValues[index]
            if val not in dicOfClassLabels:
                dicOfClassLabels[val]=1
            else:
                dicOfClassLabels[val] += 1
    else:
        for val in listOfClassValues.values.flatten():
            if val not in dicOfClassLabels:
                dicOfClassLabels[val] = 1
            else:
                dicOfClassLabels[val] += 1

    labels, freq = zip(*dicOfClassLabels.items())
    plt.bar(labels, freq)
    plt.savefig(plotName)
    plt.show()

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
pdGeneNamesList=list(pdGeneNames)
pdTrainFileValues=pdTrainFileValuesWithGeneColumn.drop(["SNO"],axis=1)

pdTrainOrderByGene=pdTrainFileValues.transpose()
# Encoding Train Class label Data to Numeric

le=preprocessing.LabelEncoder()
le.fit(pdTrainFileClassValues)
trainClassNumericalValues=le.transform(pdTrainFileClassValues)

X=pdTrainOrderByGene
y=trainClassNumericalValues

pdTrainOrderByGene=pd.DataFrame(X,columns=pdGeneNamesList)
#pdTrainFileClassValues=pdTrainFileValuesWithGeneColumn[1:]
pdTrainOrderByGene=pdTrainOrderByGene=pd.DataFrame(X,columns=pdGeneNamesList)
pdTrainOrderByGeneSample=pdTrainOrderByGene.iloc[:,0:50]
trainClassNumericalValues=y
pdTrainOrderByGene=X
pdTrainOrderByGeneSample=X.iloc[:,0:50]






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






# Make Frequency Distribution of class values

makePlotOfClassValues(pdTrainFileClassValues,"frequency_distribution_class_wise.png")

# Plotting Data using Hierarchical Clustering

#mergings=linkage(pdTrainFileValues,method="complete")
#dendrogram(mergings)



#Feature Selection Using Information Gain
X,y=pdTrainOrderByGeneNormalized,trainClassNumericalValues

FeaturesToBeSelected=500
selectKBest=SelectKBest(mutual_info_classif, k=FeaturesToBeSelected).fit(X,y)
featureScores=selectKBest.scores_
dictOfKBestFeatures=selectKBestWithFeaturesWithIndices(featureScores,FeaturesToBeSelected)
sortedKeys=sorted(dictOfKBestFeatures.keys())
kBestFeaturesData=selectKBest.transform(X)
selectedGenesNamesUsingKBest=list(pdGeneNames[sortedKeys])
dicOfGenesIndices=convertGeneKeyScoreToGeneIndexDic(dictOfKBestFeatures)




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
        kFold=KFold(n_splits=10,shuffle=True)
        tempScores=[]
        for i in range(50):
            for train_index,test_index in kFold.split(extracedGenesData):
                svmClusterClassifier = SVC()
                tempXTrain,tempXTest=extracedGenesData[train_index],extracedGenesData[test_index]
                tempYTrain,tempYTest=y_resampled[train_index],y_resampled[test_index]
                svmClusterClassifier.fit(tempXTrain,tempYTrain)
                tempScores.append(svmClusterClassifier.score(tempXTest,tempYTest))

        clusterScoreDic[key] = np.mean(tempScores)
        #xTrain,xTest,yTrain,yTest=train_test_split(extracedGenesData,y,test_size=0.4)
        #svmClusterClassifier.fit(xTrain,yTrain)
        #clusterScoreDic[key]=svmClusterClassifier.score(xTest,yTest)
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

def getFinalPruncedGenes(geneNameList,indices):
    finalGenes=[]
    for i in range(len(indices)):
        index=indices[i]
        finalGenes.append(geneNameList[index])
    return finalGenes

def dicOfGenesToIndex(pdGene):
    dic={}
    for i in range(len(pdGene)):
        val=pdGene[i]
        dic[val]=i
    return dic
def convertGeneNameToIndex(dic,geneList):
    result=[]
    for gene in geneList:
        index=dic[gene]
        result.append(index)
    return result
def saveSeletedGenesToFile(indices,path,geneFileNames):
    listOfGenes=[]
    f=open(path,'w')
    for index in indices:
        val=geneFileNames[index]
        listOfGenes.append(val)
        f.write(val+"\n")
    f.close()
    return listOfGenes
def saveSelectedGenesIndicesToFile(indices,path,geneFileNames):
    f=open(path,'w')
    for index in indices:
        f.write(str(index)+"\n")
    f.close()
def saveFullGeneList(geneList,path):
    fileFullGene = open(path, 'w')
    for gene in geneList:
        fileFullGene.write(gene + "\n")
    fileFullGene.close()
def saveGeneClusterCountToFile(dicOfGeneClusters,path):
    file_cluster_gene = open(path, 'w')
    dicCluster={}
    for key, var in dic_gene_clusters.items():
        listOfGenesCount = dic_gene_clusters[key]
        if(len(listOfGenesCount))!=-0:
            meanGeneCount = int(np.mean(listOfGenesCount))
            dicCluster[key]=meanGeneCount
    file_cluster_gene.write(str(dicCluster))
    file_cluster_gene.close()


def selectedGenesDictionary(path):
    file=open(path,'w')
    dic_of_genes={}
    for gene in file.read():
        if gene not in dic_of_genes:
            dic_of_genes[gene]=1
        else:
            dic_of_genes[gene]=0
    file.close()
    return dic_of_genes

def getFreqDistAndTopNValues(pdDataList,n):
    freqDist={}
    genesSelected=[]
    for gene in pdDataList:
       if gene in freqDist:
           freqDist[gene]+=1
       else:
           freqDist[gene]=0
    freqDist = sorted(freqDist.items(), key=lambda x: x[1],reverse=True)
    for k,v in freqDist[:n]:
        genesSelected.append(k)
    return freqDist,genesSelected

def getIndicesOfGenes(filteredGeneNames,fullGeneFile):
    geneIndices=[]
    for filter in range(len(filteredGeneNames)):
        for  gene in range(len(fullGeneFile)):
            if filteredGeneNames[filter] ==fullGeneFile[gene]:
                geneIndices.append(gene)
    return geneIndices





# Apply Kmeans Clustering
Initial_Clusters=100
Final_Clusters=10
Decreate_Rate=0.1
GeneDataForClustering=kBestFeaturesData.transpose()
Threshold=0.75
dicOfGenesIndices=convertGeneKeyScoreToGeneIndexDic(dictOfKBestFeatures)
dic_gene_clusters={}
M=Initial_Clusters
N=Final_Clusters
R=Decreate_Rate
while(M>N):
    dic_gene_clusters[M]=[]
    M=int(M-R*M)



count=1
samples=kBestFeaturesData.shape[0]
firstDim = GeneDataForClustering.shape[0]
mainLoopControl=0
cluster_gene_dist={}
accuracyPath = rootPath + "/TrainedModels/Accuracy/Accuracy_Genes.txt"
fileAccuracy = open(accuracyPath, 'w')
dic_Accuracy_genes={}

fullGeneList=[]



while mainLoopControl<0:


    print("**************************************************")
    #print(" Attempt:", K,"To Find Features < 100....")
    while (Initial_Clusters>Final_Clusters) and (firstDim>Initial_Clusters)  :
        print("Iteration.....:",count)
        print("Initial Clusters...:",Initial_Clusters)
        print("Final Clusters...:", Final_Clusters)
        print("Data Shape...:",GeneDataForClustering.shape)
        print("Features....",len(dicOfGenesIndices))
        kmeansCluster=KMeans(n_clusters=Initial_Clusters,max_iter=600,n_init=60)
        kmeansCluster.fit(GeneDataForClustering)
        dicOfGenesWithClusters=joinGenesWithClusters(kmeansCluster.labels_,Initial_Clusters,dicOfGenesIndices)
        clusterScore=clusterScoreOfGenes(dicOfGenesWithClusters,Initial_Clusters)
        filteredClusters=filterClustersWithThreshold(clusterScore,Threshold)
        GeneDataForClustering,combineGenes,dicOfGenesIndices=combinedGenesUsingClusters(filteredClusters,dicOfGenesWithClusters)
        GeneDataForClustering=GeneDataForClustering.transpose()
        Initial_Clusters=int(Initial_Clusters-Decreate_Rate*Initial_Clusters)
        samples = GeneDataForClustering.shape[1]
        firstDim = GeneDataForClustering.shape[0]
        dic_gene_clusters[Initial_Clusters].append(firstDim)
        count+=1
        print("___________________________________________________________")
    filteredGenesList=getFinalPruncedGenes(pdGeneNamesList,list(dicOfGenesIndices.values()))
    svmClassifier=SVC()
    GeneDataForClusteringReshaped=GeneDataForClustering.transpose()
    kFoldCrossValidator=KFold(n_splits=10,shuffle=True)
    averageAccuracy=[]
    for i in range(3):
        for train_index,test_index in kFoldCrossValidator.split(GeneDataForClusteringReshaped,y):
            FinalX_train,FinalX_test=GeneDataForClusteringReshaped[train_index],GeneDataForClusteringReshaped[test_index]
            FinalY_Train,FinalY_test=y[train_index],y[test_index]
            svmClassifier.fit(FinalX_train,FinalY_Train)
            averageAccuracy.append(svmClassifier.score(FinalX_test,FinalY_test))
    if np.mean(averageAccuracy)>=0.9:
        dic_Accuracy_genes[mainLoopControl]=str(np.mean(averageAccuracy))+","+str(firstDim)
        mainLoopControl+=1
        print("Passed......",mainLoopControl)
        print("Accuracy:",np.mean(averageAccuracy))
        dicOfGenes=dicOfGenesToIndex(pdGeneNames)
        geneIndices=convertGeneNameToIndex(dicOfGenes,filteredGenesList)
        pdTestDataNormalized=stats.zscore(pdTestFileValues)
        testDataPruned=pdTestDataNormalized[:,geneIndices]
        print(le.inverse_transform(svmClassifier.predict(testDataPruned)))
        geneName="/TrainedModels/GeneNames/geneNames_"+str(mainLoopControl)+".txt"
        geneIndex="/TrainedModels/GeneIndices/geneIndices_"+str(mainLoopControl)+".txt"
        geneAccuracy=str(firstDim)+","+str(np.mean(averageAccuracy))
        genesNamePath=  rootPath+geneName
        genesIndexPath =rootPath+geneIndex
        genesExtracted=saveSeletedGenesToFile(geneIndices,genesNamePath,pdGeneNames)
        for gene in genesExtracted:
            fullGeneList.append(gene)
        saveSelectedGenesIndicesToFile(geneIndices,genesIndexPath,pdGeneNames)


    Initial_Clusters=100
    Final_Clusters=10
    Decreate_Rate=0.1
    GeneDataForClustering=kBestFeaturesData.transpose()
    Threshold=0.7
    dicOfGenesIndices=convertGeneKeyScoreToGeneIndexDic(dictOfKBestFeatures)
    count=1
    samples=kBestFeaturesData.shape[0]
    firstDim = GeneDataForClustering.shape[0]
    #Getting Test Data

fileAccuracy.write(str(dic_Accuracy_genes))
fileAccuracy.close()

saveFullGeneList(fullGeneList,rootPath + "/TrainedModels/FullGeneList/FullGenes.txt")
saveGeneClusterCountToFile(dic_gene_clusters,rootPath + "/TrainedModels/Accuracy/Cluster_Genes.txt")

fullGenePath=rootPath+"/TrainedModels/FullGeneList/FullGenes.txt"
pdAllSelectedGeneSet=pd.read_csv(fullGenePath,sep=" ",header=None)


FinalGenesCount=25
freqDistOfSelectedGenes,TopGenes=getFreqDistAndTopNValues((list(pdAllSelectedGeneSet[0])),FinalGenesCount)
geneIndices=getIndicesOfGenes(TopGenes,pdGeneNames)
X_Selected=X[:,geneIndices]


svmRCEClassifer=SVC()
finalAccuracy=[]
for train_index, test_index in kFoldCrossValidator.split(X_Selected, y):
    FinalX_train, FinalX_test = GeneDataForClusteringReshaped[train_index], GeneDataForClusteringReshaped[test_index]
    FinalY_Train, FinalY_test = y[train_index], y[test_index]
    svmRCEClassifer.fit(FinalX_train, FinalY_Train)
    finalAccuracy.append(svmRCEClassifer.score(FinalX_test, FinalY_test))
print(np.mean(finalAccuracy))

saveTrainedModel(svmRCEClassifer,rootPath+"/TrainedModels/Model/svm-rce.pkl")

mergings=linkage(X_Selected.transpose(),method="complete")
dendrogram(mergings,labels=TopGenes)
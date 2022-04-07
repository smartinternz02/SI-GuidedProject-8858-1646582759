#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import pickle
import operator


# In[14]:


def distance(instance1, instance2, k ):
    distance =0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
#Method to calculate distance between two instances.
    distance = np.trace (np.dot (np. linalg.inv(cm2), cm1))
    distance+=(np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)),mm2-mm1))
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance


# In[15]:


def getNeighbors (trainingSet, instance, k):
    distances = []
    for x in range (len (trainingSet)):
        dist = distance (trainingSet [x], instance, k )+ distance (instance, trainingSet[x], k) 
        distances.append( (trainingSet [x] [2], dist)) 
    distances.sort (key=operator.itemgetter(1))
    neighbors = []
    for x in range (k):
        neighbors.append (distances [x][0])
    print("top",k, "neighbors are ",neighbors)
    return neighbors


# In[16]:


def nearestClass(neighbors): 
    classVote = {}
    for x in range (len (neighbors)):
        response = neighbors [x]
        if response in classVote:
            classVote[response]+=1
        else:
            classVote[response]=1
    sorter = sorted (classVote.items (), key = operator.itemgetter(1), reverse=True) 
    return sorter [0][0]


# In[17]:


directory = r"C:\Users\lasya\OneDrive\Desktop\pt-2\Music Genres"


# In[18]:


# Extract features from the data (audio files) and dump these features #into a binary .dat file "my.dat": 
f = open("my.dat", 'wb') 
i = 0
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+'/'+folder):
        try:
            (rate,sig) = wav.read(directory+"/"+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        except Exception as e:
            print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
f.close()


# In[19]:


#Loading the created dataset into a python readable object (list) 
dataset = []
def loadDataset(filename):
    with open(r"C:\Users\lasya\OneDrive\Desktop\pt-2\my.dat", 'rb') as f: 
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError: 
                f.close()
                break
loadDataset (r"C:\Users\lasya\OneDrive\Desktop\pt-2\my.dat")
#we have to convert the dataset from a list to np array.
dataset = np.array(dataset)
#type(dataset) ##uncomment this line to check the type of (dataset),


# In[20]:


#Train and test split on the dataset: 
#as the dataset contains features for all the audio files, 
#we have to split that manually into train and test data 
from sklearn.model_selection import train_test_split 
x_train,x_test = train_test_split(dataset, test_size=0.15)
#Make prediction using KNN and get the accuracy on test data: 
leng = len(x_test) 
predictions = []
for x in range (leng): 
    predictions.append(nearestClass(getNeighbors(x_train,x_test[x], 8)))


# In[21]:


def getAccuracy (testSet, predictions): 
    #this is a variable to count total number of correct predictions.
    correct = 0
    for x in range (len (testSet)) :
        if testSet[x] [-1] ==predictions [x]: 
            correct+=1
    return 1.0*correct/len(testSet)
#Print accuracy using defined function
accuracy1 = getAccuracy (x_test, predictions) 
print(accuracy1)


# In[ ]:





# In[ ]:





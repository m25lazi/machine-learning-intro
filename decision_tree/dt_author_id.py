#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# What's the number of features in your data? (Hint: the data is organized into a numpy array where the number of rows is the number of data points and the number of columns is the number of features.)
noOfFeatures = len(features_train[0])
print "Number of features : ", noOfFeatures
# With selector = SelectPercentile(f_classif, percentile=10) in email_preprocess.py - 3785
# selector = SelectPercentile(f_classif, percentile=1) - 379



#########################################################
### your code goes here ###
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy;

# With 10 percentile features
# Training time: 66.607 s
# Prediction time: 0.037 s
# 0.97781

# With 1 percentile features
# Training time: 4.283 s
# Prediction time: 0.002 s
# 0.9664

#########################################################



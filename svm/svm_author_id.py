#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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




#########################################################
### your code goes here ###

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

from sklearn.svm import SVC
clf = SVC(C=10000.0, kernel='rbf') #(kernel='linear')
print clf

t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy;

# kernel='linear'
# Training time: 158.786 s
# Prediction time: 15.507 s
# 0.984072810011

# After slicing the training set down to 1% of original

# kernel='linear'
# Training time: 0.088 s
# Prediction time: 1.04 s
# 0.884527872582

# kernel='rbf', C= 1.0
# Training time: 0.1 s
# Prediction time: 1.016 s
# 0.616040955631

# kernel='rbf', C= 10.0
# Training time: 0.101 s
# Prediction time: 1.067 s
# 0.616040955631

# kernel='rbf', C= 100.0
# Training time: 0.099 s
# Prediction time: 1.037 s
# 0.616040955631

# kernel='rbf', C= 1000.0
# Training time: 0.102 s
# Prediction time: 1.042 s
# 0.821387940842

# kernel='rbf', C= 10000.0
# Training time: 0.096 s
# Prediction time: 0.866 s
# 0.892491467577

# kernel='rbf', C= 10000.0, With Full set of training data
# Training time: 113.658 s
# Prediction time: 10.483 s
# 0.990898748578

#########################################################



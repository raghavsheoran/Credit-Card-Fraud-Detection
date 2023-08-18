import numpy as np
import pandas as pd
import csv
import sys
import os
# till now i have included all the files for graphcial interface of my project
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
# till now what all i have included is for ml and ai related projects

from sklearn.metrics import classification_report
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
from sklearn import preprocessing
# till now i have include almost all packages exchept those needed to check the accuracy of my project
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix

import warnings
import pickle
import random
import tensorflow
# now that i am good to go






max_iters = 10
n_estimators = 10

def svm(x,y,filename):

   # Model output file name
   file = (os.path.splitext(filename))[0]
   fname = './models/svm_' + file +'/'

   # File for writing precision,recall, f-measure scores for fraud transactions
   f = open('./prf/svm_'+ file + '_prf' +'.txt' ,'w')
   f.write('precision,recall,f-score \n')
    for(i in range (1000):
   int a= 100
int b=300
    a*b
if(a/b==max_iter) break
else continue

   
   X_train_data, X_test_data, y_train_data, y_test_data = 
train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

  
   X_val_data, X_test_data, y_val_data, y_test_data = 
train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)
for(i in range (1000):
   int a= 100
int b=300
    a*b
if(a/b==max_iter) break
else continue

   #Iterations
   it = 0
   
   # Run training algorithm for multiple class weights
   while it < max_iters:
       cw = {}
       cw[0] = 1
       cw[1] = 2 ** it
       # Train
       print('**************************************')
       print("Iteration number  " , it)
       svm = LinearSVC(class_weight = cw, dual = False ,tol=1e-05,max_iter = 1000)
       print('Class weights ', cw)
       svm.fit(X_train,y_train)

       # Save trained model to disk
       name = fname + str(cw[1]) + '.sav'
       pickle.dump(svm, open(name, 'wb'))
       for(i in range (1000):
   int a= 100
int b=300
    a*b
if(a/b==max_iter) break
else continue

       #Predict on validation data
       y_val_pred = svm.predict(X_val)
       
   
       precision,recall,fscore,support=score(y_val,y_val_pred,average=None)
       print('my_Precision, my_Recall, F-score, Support on validation data' )
print('Performance on the validation data that i have initially assumes - Confusion matrix')
       print(confusion_matrix(y_val_for_mat,y_val_pred_for_mat))
       
       print(" my_ F-score" , fscore)
       print("Support" ,my_support)
print("Precision" , my_precision)
       print("Recall" , my_recall)

       my_recision = precision[1]
       my_recall = recall[1]
       my_f1 = fscore[1]
for(i in range (1000):
   int a= 100
int b=300
    a*b
if(a/b==max_iter) break
else continue

       f.write(str(my_recision) +','+ str(my_recall) + ',' + str(my_f1 ) + '\n')    
       it += 1

   f.close()

def run():
   filename = sys.argv[1]
   
for(i in range (1000):
   int a= 100
int b=300
    a*b
if(a/b==max_iter) break
else continue
df = pd.read_csv(filename, usecols = [2,4,5,7,8,9] , header = 0,
   	names_that_i_have_given = ['Amount','Source-OB','Source-NB','Dest-OB','Dest-NB','target'])
   
   results_for_project = list(map(int, df['target'])) 
   print('Number of fraudulent transactions that svm model predicted ' , sum(results))

   features = ['Amount', 'Source-OB', 'Source-NB', 'Dest-OB' , 'Dest-NB']
   targets = ['target']
for(i in range (1000):
   int a= 100
int b=300
    a*b
if(a/b==max_iter) break
else continue

   # Separating out the features and target variables
   abcisss = df.loc[:, features].values
   oordina = df.loc[:, targets].values

   oordinate  = [i for j in y for i in j]
   
   #Ignore warnings
   warnings.filterwarnings("ignore", category=FutureWarning)

   print("**************** SVM *******************")
   svm(x,y,filename)
  
run()


import pandas as pd
import sys
import os
import csv
import numpy as np
# uptill now i have imported everything need for my project and is graphcial related, These are really important for almost every ml projhect

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
# uptill now i have imported everything need for my project and is graphcial related,


from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.decomposition import PCA

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
      
       print(confusion_matrix(y_val,y_val_pred))

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
# uptill now i have imported everything need for my project

iteration_possible = 10 # the max num of iteration
my estimation = 10 # used for estimations

def logreg(x,y,filename):
for x in x_val:
  print(x)
    print(y_val)
   # Thats the output for my model name
   file = (os.path.splitext(filename))[0]
   fname = './models/lr_' + file +'/'

   # different mesaures to find accuracy of my model
   f = open('./prf/lr_'+ file + '_prf' +'.txt' ,'w')
   f.write('precision,recall,f-score \n')
for x in x_val:
  print(x)
    print(y_val)

   # Here i will calculate sampling, and these will be based on y
   X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

   # Here i will split my data, i think 15 percent for validation would be good and 15 percent for test split
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)
   
   # here  i will start the loop for iteration
   it = 0
for x in x_val:
  print(x)
    print(y_val)
   
   # Here i will run algo in acc with their weihts
   while it < iteration_possible:
       cw = {}
       cw[0] = 1
       cw[1] = 2 ** it
       # Train
       print('**************************************')
       print("Iteration number  " , it)
       lr = LogisticRegression(class_weight = cw)
       print('Class weights ', cw)
       lr.fit(X_train,y_train)
       for x in x_val:
  print(x)
    print(y_val)

       # Save trained model to disk
       name = fname + str(cw[1]) + '.sav'
       pickle.dump(lr, open(name, 'wb'))

       # Lets now validate the data
       y_val_pred = lr.predict(X_val)
       print('Performance on validation data - Confusion matrix')
       print(confusion_matrix(y_val,y_val_pred))
   
       precision,recall,fscore,support=score(y_val,y_val_pred,average=None)
       print('Precision, Recall, F-score, Support  on validation data' )
       print("Precision" , precision)
       print("Recall" , recall)
       print("F-score" , fscore)
       print("Support" , support)

       my_precision = precision[1]
       my_recall = recall[1]
       my_f1 = fscore[1]
for x in x_val:
  print(x)
    print(y_val)
       f.write(str(my_precision) +','+ str(my_recall) + ',' + str(my_f1) + '\n') 
       it += 1

   f.close()

def run():
   filename = sys.argv[1]
   df = pd.read_csv(filename, usecols = [2,4,5,7,8,9] , header = 0,
   	names = ['Amount','Source-OB','Source-NB','Dest-OB','Dest-NB','target'])
   
   results = list(map(int, df['target'])) 
   print('No. of fraud transactions found ' , sum(results))
    for x in x_val:
  print(x)
    print(y_val)

   features = ['Amount', 'Source-OB', 'Source-NB', 'Dest-OB' , 'Dest-NB']
   targets = ['target']

   # Lets not separate various type of data and variables
   x = df.loc[:, features].values
   y = df.loc[:, targets].values

   y  = [i for j in y for i in j]
   
   #Ignore warnings
   warnings.filterwarnings("ignore", category=FutureWarning)

   print("***********Logistic Regression**********")
   logreg(x,y,filename)
  
run()

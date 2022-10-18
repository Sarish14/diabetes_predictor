import pickle
import statistics
import pandas as pd
import app
import numpy as np

import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


data = pd.read_csv("diabetes.csv")

cols = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for col in cols:
    data[col]=data[col].replace(0,int(round(data[col].mean())))

l1 = ['Pregnancies','BloodPressure','SkinThickness','Age']
l2 = ['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

data['Glucose_Group'] = [ int(i / 5) for i in data['Glucose']]
data['Insulin_Group'] = [ int(i / 50) for i in data['Insulin']]
data['BMI_Group'] = [ int(i / 5) for i in data['BMI']]
data['DiabetesPedigreeFunction_Group'] = [ int((i*100) / 5) for i in data['DiabetesPedigreeFunction']]

#global preg
'''global pedigree
global age
global blood_pressure
global glucose
global bmi
global insulin
'''
preg = [x for x in data['Pregnancies']]
pedigree = [x for x in data['DiabetesPedigreeFunction']]
age = [x for x in data['Age']]
blood_pressure = [x for x in data['BloodPressure']]
glucose = [x for x in data['Glucose']]
bmi = [x for x in data['BMI']]
insulin = [x for x in data['Insulin']]


from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler()
ss = StandardScaler()
df1 = data.copy(deep = True)
df1.drop(columns = ['Glucose_Group','Insulin_Group','BMI_Group','DiabetesPedigreeFunction_Group'],inplace =True)
df1['Pregnancies'] = mms.fit_transform(df1[['Pregnancies']])
df1['Insulin'] = ss.fit_transform(df1[['Insulin']])
df1['DiabetesPedigreeFunction'] = ss.fit_transform(df1[['DiabetesPedigreeFunction']])
df1['Age'] = ss.fit_transform(df1[['Age']])
df1['BloodPressure'] = ss.fit_transform(df1[['BloodPressure']])
df1['SkinThickness'] = ss.fit_transform(df1[['SkinThickness']])
df1['Glucose'] = ss.fit_transform(df1[['Glucose']])
df1['BMI'] = ss.fit_transform(df1[['BMI']])

df1.drop(columns=['SkinThickness'],inplace = True)
df2= df1.copy(deep = True)

f1 = df1.iloc[:,:7].values
t1 = df1.iloc[:,7].values


over = SMOTE()
f2 = df1.iloc[:,:7].values
t2 = df1.iloc[:,7].values
f2, t2 = over.fit_resample(f2, t2)

x_train2,x_test2,y_train2,y_test2 = train_test_split(f2,t2,test_size=0.25,random_state=1)

# Instantiate the model
classifier = XGBClassifier(learning_rate=0.01,max_depth=9,n_estimators=950,nthread =4,seed =42)

# Fit the model
classifier.fit(x_train2,y_train2)

# Make a pickle file of the model
pickle.dump(classifier,open("model.pkl","wb"))

























from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pickle
url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df=pd.read_csv(url, sep=',')

df=df.drop(columns=['Name'])
df['Sex']=df['Sex'].map({'male':1,'female':0})
df=df.drop(columns=['Ticket'])
df=df.drop(columns=['Cabin'])
df['Embarked']=df['Embarked'].map({'S':2,'C':1,'Q':0})
df['Age'][np.isnan(df['Age'])]=df['Age'].mean()
df['Embarked'][np.isnan(df['Embarked'])]=2

X=df.drop(columns=['Survived'])
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=520, test_size=0.2)
modelo_final=xgb.XGBClassifier(colsample_bytree= 0.9,eta= 0.15,gamma= 0.5,max_depth= 2,min_child_weight= 3)
modelo_final.fit(X_train, y_train)
y_train_final=modelo_final.predict(X_train)
y_test_final=modelo_final.predict(X_test)
pickle.dump(modelo_final, open('../models/XGboost.pickle', 'wb'))
git remote set-url origin 
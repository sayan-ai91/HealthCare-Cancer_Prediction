#!/usr/bin/env python
# coding: utf-8

#  # CANCER PREDICTION

# In[175]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.inspection import permutation_importance
import sweetviz
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


# In[176]:


data=pd.read_csv("C:/Users/Sayan Mondal/Desktop/wbcd.csv")


# In[177]:


data.head()


# In[178]:


data.drop(['id'], axis=1,inplace=True)  ## Droping unnecessary columns


# In[179]:


data.isnull().sum()  ## There is no missing value in the dataset


# In[ ]:


#data.loc[data.diagnosis=="B","diagnosis"]=1
#data.loc[data.diagnosis=="M","diagnosis"]=0


# In[180]:


# Import label encoder 
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'diagnosis'. 
data['diagnosis']= label_encoder.fit_transform(data['diagnosis']) 


# In[7]:


data.head(18)


# In[181]:


X=data.drop(["diagnosis"],axis=1)
y=data["diagnosis"]


# In[182]:


X.shape,y.shape


# CHECKING DATA STRUCTURES

# In[15]:


data.hist()
plt.rcParams['figure.figsize'] = (16.0, 17.0)
plt.show()


# In[154]:


## checking percentage wise distribution of Diagnosis..##
data['diagnosis'].value_counts(normalize=True)*100


# In[16]:


## Q-Q PLOT
from statsmodels.graphics.gofplots import qqplot
# q-q plot
qqplot(X,line='s')
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.show()


# # Plotting Heatmap

# In[164]:


data2=data.corr()
plt.rcParams['figure.figsize'] = (16.0, 10.0)
sns.heatmap(data2,cmap='RdYlGn') 


# In[19]:


import sweetviz ## EDA in one line
my_report = sweetviz.analyze([data, "data"],target_feat='diagnosis')



# In[162]:


## EDA report in html file...
my_report.show_html('Cancer_report.html')


# # Indentification of top 10 features

# In[127]:


best_features=SelectKBest(score_func=chi2,k=15)
fit=best_features.fit(X,y)


# In[129]:


dataf_scores=pd.DataFrame(fit.scores_)
dataf_columns=pd.DataFrame(X.columns)


# In[130]:


## Concat two Dataframe for better vizualization
featuresScores=pd.concat([dataf_columns,dataf_scores],axis=1)
featuresScores.columns=['Inde_variable','Score']


# In[167]:


## Sorting the score value in descending order
featuresScores.sort_values(by=['Score'], inplace=True, ascending=False)
featuresScores


# In[163]:


print(featuresScores.nlargest(10,'Score')) ## Finding out the top 10 features


# In[21]:


## Train Test Split..
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=10)


# In[22]:


## Scale the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[23]:


### Fit only to the training data
scaler.fit(X_train)


# In[24]:


## Now apply transformation to the data
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# ###  LOGISTIC REGRESSION ####

# In[226]:


from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(C=1.0)  ## Regularization parameter C
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=10)

model=logreg.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[227]:


## Accuracy
accuracy_score(y_test, y_pred)


# In[228]:


# confusion Matrix...##
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[229]:


## Kappa Score
cohen_kappa_score(y_test,y_pred)


# In[185]:


## Defining a Learning Curve Function

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# In[187]:


#Learning curve for LR
plot_learning_curve(logreg, 'Learning Curve For Logistic Model', X, y, (0.85,1.05), 10)
plt.rcParams['figure.figsize'] = (8.0, 5.0)
plt.savefig('5')
plt.show()


# # Support Vector Machine

# In[222]:


#### appling SVM Model....###
from sklearn import svm
from sklearn.exceptions import NotFittedError
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier=svm.SVC(kernel='linear',gamma='auto', C=2)
model=classifier.fit(X_train,y_train)

y_pred=model.predict(X_test)


# In[223]:



accuracy_score(y_test, y_pred)


# In[224]:


# confusion Matrix...##
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[225]:


## Kappa Score
cohen_kappa_score(y_test,y_pred)


# In[192]:


#Learning curve for SVM
plot_learning_curve(classifier,'Learning Curve For SVM', X, y, (0.85,1.05), 10)
plt.rcParams['figure.figsize'] =(5.0, 4.0)
plt.savefig('5')
plt.show()


# # Naive Bayes Classifier

# In[212]:


## Using Naive Bays Classifier...##
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 42)
model=nb.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[213]:


## Accuracy
accuracy_score(y_test, y_pred)


# In[214]:


# confusion Matrix...##
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[215]:


## Kappa Score
cohen_kappa_score(y_test,y_pred)


# In[197]:


#Learning curve for NB
plot_learning_curve(nb, 'Learning Curve For NB', X, y, (0.85,1.05), 10)
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.savefig('5')
plt.show()


# # Random Forest Classifier

# In[198]:


from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =10)

rf= RandomForestClassifier(n_estimators=50, criterion='gini')

model=rf.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[199]:


## Accuracy
accuracy_score(y_test, y_pred)


# In[200]:


##Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[43]:


## Kappa Score
cohen_kappa_score(y_test,y_pred)


# In[201]:


#Learning curve for RF
plot_learning_curve(rf, 'Learning Curve For RF', X, y, (0.85,1.05), 10)
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.savefig('4')
plt.show()


# # CatBoost Classifier

# In[202]:


### catboost..##
from catboost import CatBoostClassifier
cb=CatBoostClassifier(iterations=400,learning_rate=0.03,depth=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3,random_state =10)

model=cb.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[203]:


## Accuracy
accuracy_score(y_test, y_pred)


# In[204]:


## Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[205]:


## Kappa Score
cohen_kappa_score(y_test,y_pred)


# In[207]:


#Learning curve for CatBoost
plot_learning_curve(cb, 'Learning Curve For CatBoost', X, y, (0.85,1.05), 10)
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.savefig('4')
plt.show()


# Kappa Score:: 0.9640(CatBoost)>0.9518(RF)>0.8929(SVM)>0.8840(NB)>0.8555(LR)>

# So from Kappa Score Final Selected Model is CatBoost.

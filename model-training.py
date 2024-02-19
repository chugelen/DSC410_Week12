#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Dense
import joblib
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[2]:


# Read csv and create a dataframe
df = pd.read_csv('HR Analytics/Employee-Attrition.csv')


# ## Remove Irrelevant Columns
# 
# - 'EmployeeCount': It appears to represent the count of individual employees.
# - 'EmployeeNumber': This variable is not significant for the model.
# - 'Over18': Since it only has one value, it should not be considered significant for predictions.
# - 'StandardHours': This variable is not significant for prediction as it has a constant value of 40 hours for all records.

# In[3]:


#remove irrevelant columns
columns = ['EmployeeCount','EmployeeNumber','Over18', 'StandardHours']

df = df.drop(columns, axis=1)


# # Dataset Information
# Display general information of the dataset, such as the shape, dataypes and null values.
# 
# This section will be used to learn what type of data there is and to see what kind of feature engineering and data cleaning that will be needed.

# In[4]:


df.head()


# In[5]:


#display dataframe shape
df.shape


# In[6]:


#display datatypes of columns
df.dtypes


# In[7]:


cat = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
       'JobRole', 'MaritalStatus', 'OverTime']
num = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
       'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']
target = ['Attrition']


# ## Encoding categorical features

# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to each categorical column
for column in cat:
    df[column] = le.fit_transform(df[column])
    
# Apply LabelEncoder to Attrition column
df['Attrition'] = le.fit_transform(df['Attrition'])


# ## EDA

# Because there are so many columns, I'm going to use the correlation analysis, to see if reduce the size of the columns, to hopefully have a better outcome of the model.

# In[9]:


# Correlation Analysis
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')
plt.title('Correlation Matrix')
plt.show()


# The following columns have high correlations, and I will remove some of them from the dataset to reduce redundancy, and improve model performance.
# 
# - Age and TotalWorkingYears
# - JobLevel and MonthlyIncome
# - JobLevel and TotalWorkingYears
# - MonthlyIncome and TotalWorkingYears
# - PercentSalaryHike and PerformanceRating
# - YearsAtCompany and YearsInCurrentRole
# - YearsAtCompany and YearsWithCurrManager
# - YearsatCompany and YearsSinceLastPromotion
# - YearsInCurrentRole and YearsWithCurrManager
# 
# 
# I will remove the following features:
# 
# - `JobLevel` because it can be determined that it a persons higher joblevel would lead to higher `MonthlyIncome` and `TotalWorkingYears`, so this will most likely not effect the model.
# 
# - `TotalWorkingYears` similar to above if the `TotalWorkingYears` is higher than the `MonthlyIncome` would most likely be higher. This also correlates highly with `Age` its obvious the higher the `TotalWorkingYears` is the `Age` would likely also be high. 
# 
# - `PercentSalaryHike` assuming this is based off the `PerformanceRating` only one field should be needed. 
# 
# - `YearsinCurrentRole` this will be redunct to `YearsAtCompany` and that would show the over tenure.
# 
# - `YearsSinceLastPromotion` this has a high correlation with both `YearsAtCompany` and `MonthlyIncome` 
# 
# - `YearsWithCurrManager` similar to above I believe `YearsAtCompany` would show overall tenure. 

# In[10]:


#remove high correlations columns
columns = ['JobLevel', 'TotalWorkingYears', 'PercentSalaryHike', 'YearsInCurrentRole', 
          'YearsSinceLastPromotion','YearsWithCurrManager']

df = df.drop(columns, axis=1)


# # Balance Dataset 
# 
# To prevent biased towards predicting the majority of the target `Attrition`

# In[11]:


#Check Attrition distribution
df['Attrition'].value_counts()


# In[12]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Separate features and target variable
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Apply RandomUnderSampler to balance distribution
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Display distribution after resampling
y_resampled.value_counts()


# # Model Training & Testing

# ## Split and Standardize Dataset

# In[13]:


X = X_resampled
y = y_resampled


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5/30, random_state=101)

#Create an instance of the class
scaler = StandardScaler()

#use fit method
scaler.fit(X_train)

#transform method to perform the transformation
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Random Forest

# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators=99,
                            max_features=6,
                            max_depth=6,
                            min_samples_split=100,
                            random_state=85)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[16]:


accuracy_rf = accuracy_score(y_true=y_test, y_pred=y_pred_rf)
accuracy_rf


# In[17]:


from sklearn.metrics import confusion_matrix
def CM(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    out = pd.DataFrame(M, index=["Actual No Attrition", "Actual Attrition"], columns=["Predicted No Attrition", "Predicted Attrition"])
    return out

threshold = 0.5
y_pred_prob = rf.predict_proba(X_test)[:,1]
y_pred = (y_pred_prob > threshold).astype(int)

CM(y_test, y_pred)


# In[18]:


accuracy_rf = accuracy_score(y_true=y_test, y_pred=y_pred_rf)
accuracy_rf


# # Logistic Regression

# In[19]:


# Create and fit the logistic regression model
simple_log_reg = LogisticRegression(C=1e6)
simple_log_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred_logreg = simple_log_reg.predict(X_test)

# Calculate accuracy on the test set
accuracy_logreg = accuracy_score(y_true=y_test, y_pred=y_pred_logreg)
accuracy_logreg


# # GridSearch

# In[20]:


from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators":[25,100,200,400],
              "max_features":[4,10,19],
              "max_depth":[4,8,16,20]}

# Create the RandomForestClassifier
rf = RandomForestClassifier(random_state=17)

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy',
                           n_jobs=-1)

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_rf = grid_search.best_estimator_

# Predictions on the test set using the best model
y_pred_rf_tuned = best_rf.predict(X_test)

# Calculate accuracy on the test set
accuracy_rf_tuned = accuracy_score(y_true=y_test, y_pred=y_pred_rf_tuned)
print("Improved Random Forest Accuracy:", accuracy_rf_tuned)


# In[21]:


# Building the neural network
n_input = X.shape[1]
n_hidden1 = 32
n_hidden2 = 16
n_hidden3 = 8

nn_cls = Sequential()
nn_cls.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,)))
nn_cls.add(Dense(units=n_hidden2, activation='relu'))
nn_cls.add(Dense(units=n_hidden3, activation='relu'))
# output layer for binary classification with 'sigmoid' activation
nn_cls.add(Dense(units=1, activation='sigmoid'))

# Training the neural network for binary classification
batch_size = 32
n_epochs = 40
nn_cls.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_cls.fit(X, y, epochs=n_epochs, batch_size=batch_size)


# In[23]:


## Serializing:
# Scaler
joblib.dump(scaler, './Model/scaler.joblib')

# Trained model
nn_cls.save("./Model/attrition_prediction_model.h5")


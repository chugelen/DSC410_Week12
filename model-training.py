#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import Sequential
from keras.layers import Dense
import joblib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the dataset into dataframe
df = pd.read_csv('data/travel_insurance.csv')


# # Dataset Information
# 
# Display general information of the dataset, such as the shape, dataypes and null values.
# 
# This section will be used to learn what type of data there is and to see what kind of feature engineering and data cleaning that will be needed. 

# In[3]:


df.head()


# In[4]:


#display shape of dataframe
df.shape


# In[5]:


#display datatypes of the dataframe
df.dtypes


# In[6]:


#check for null values
df.isnull().sum()


# Because there is no many null values in the gender field, I will remove this from the dataset. With the provided values, I don't think it would be able to determine what the gender should be, so it would be better to remove it. 

# In[7]:


#drop the 'Gender' column
df = df.drop('Gender', axis=1)


# # Renaming Columns for Convenience
# 
# In the following code, I have renamed some of the columns in the DataFrame to make the code more concise and easier to read. This is purely for convenience and does not alter the underlying data.
# 
# The original column names and their corresponding new names are as follows:
# 
# - `Agency Type` is renamed to `Type`
# - `Distribution Channel` is renamed to `Dist_Channel`
# - `Product Name` is renamed to `Product`
# - `Net Sales` is renamed to `Sales`
# - `Commision (in value)` is renamed to `Commission`
# 
# This will help streamline the coding process and make it more straightforward to reference specific columns in subsequent analyses.
# 

# In[8]:


#renaming columns 
df.rename(columns={
    'Agency Type': 'Type',
    'Distribution Channel': 'Dist_Channel',
    'Product Name': 'Product',
    'Net Sales': 'Sales',
    'Commision (in value)': 'Commission'
}, inplace=True)


# In[9]:


#create a list of containing the groups of features of categorical and numerical values. 
cat = ['Agency','Type','Dist_Channel', 'Product', 'Destination']
num =['Duration', 'Sales','Commission', 'Age']
target = ['Claim']


# In[10]:


# Display unique values in each column of the categorial columns
for column in df[cat].columns:
    unique_values = df[column].unique()
    print(f"\nUnique values in {column}: {unique_values}")


# # Encoding categorical features

# ### Grouping & Encoding Destinations Column
# Group similar destinations by reigon, to help reduce the number of unique values. Then encode those reigons into numbers so they can be used in the prediction model. 
# 
# The orginal destination with its assigned region is below in destination_mapping. 
# Each region will than be encoded with a numerical value, and the value of that is below in region_encoded.
# 
# Once each is grouped and encdoed the destination field will be updated to the region encoded value.

# In[11]:


destination_mapping = {
    'MALAYSIA': 'Asia',
    'AUSTRALIA': 'Oceania',
    'ITALY': 'Europe',
    'UNITED STATES': 'North America',
    'THAILAND': 'Asia',
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": 'Asia',
    'NORWAY': 'Europe',
    'VIET NAM': 'Asia',
    'DENMARK': 'Europe',
    'SINGAPORE': 'Asia',
    'JAPAN': 'Asia',
    'UNITED KINGDOM': 'Europe',
    'INDONESIA': 'Asia',
    'INDIA': 'Asia',
    'CHINA': 'Asia',
    'FRANCE': 'Europe',
    'TAIWAN, PROVINCE OF CHINA': 'Asia',
    'PHILIPPINES': 'Asia',
    'MYANMAR': 'Asia',
    'HONG KONG': 'Asia',
    'KOREA, REPUBLIC OF': 'Asia',
    'UNITED ARAB EMIRATES': 'Middle East',
    'NAMIBIA': 'Africa',
    'NEW ZEALAND': 'Oceania',
    'COSTA RICA': 'North America',
    'BRUNEI DARUSSALAM': 'Asia',
    'POLAND': 'Europe',
    'SPAIN': 'Europe',
    'CZECH REPUBLIC': 'Europe',
    'GERMANY': 'Europe',
    'SRI LANKA': 'Asia',
    'CAMBODIA': 'Asia',
    'AUSTRIA': 'Europe',
    'SOUTH AFRICA': 'Africa',
    'TANZANIA, UNITED REPUBLIC OF': 'Africa',
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": 'Asia',
    'NEPAL': 'Asia',
    'NETHERLANDS': 'Europe',
    'MACAO': 'Asia',
    'CROATIA': 'Europe',
    'FINLAND': 'Europe',
    'CANADA': 'North America',
    'TUNISIA': 'Africa',
    'RUSSIAN FEDERATION': 'Europe',
    'GREECE': 'Europe',
    'BELGIUM': 'Europe',
    'IRELAND': 'Europe',
    'SWITZERLAND': 'Europe',
    'CHILE': 'South America',
    'ISRAEL': 'Middle East',
    'BANGLADESH': 'Asia',
    'ICELAND': 'Europe',
    'PORTUGAL': 'Europe',
    'ROMANIA': 'Europe',
    'KENYA': 'Africa',
    'GEORGIA': 'Asia',
    'TURKEY': 'Middle East',
    'SWEDEN': 'Europe',
    'MALDIVES': 'Asia',
    'ESTONIA': 'Europe',
    'SAUDI ARABIA': 'Middle East',
    'PAKISTAN': 'Asia',
    'QATAR': 'Middle East',
    'PERU': 'South America',
    'LUXEMBOURG': 'Europe',
    'MONGOLIA': 'Asia',
    'ARGENTINA': 'South America',
    'CYPRUS': 'Europe',
    'FIJI': 'Oceania',
    'BARBADOS': 'North America',
    'TRINIDAD AND TOBAGO': 'North America',
    'ETHIOPIA': 'Africa',
    'PAPUA NEW GUINEA': 'Oceania',
    'SERBIA': 'Europe',
    'JORDAN': 'Middle East',
    'ECUADOR': 'South America',
    'BENIN': 'Africa',
    'OMAN': 'Middle East',
    'BAHRAIN': 'Middle East',
    'UGANDA': 'Africa',
    'BRAZIL': 'South America',
    'MEXICO': 'North America',
    'HUNGARY': 'Europe',
    'AZERBAIJAN': 'Asia',
    'MOROCCO': 'Africa',
    'URUGUAY': 'South America',
    'MAURITIUS': 'Africa',
    'JAMAICA': 'North America',
    'KAZAKHSTAN': 'Asia',
    'GHANA': 'Africa',
    'UZBEKISTAN': 'Asia',
    'SLOVENIA': 'Europe',
    'KUWAIT': 'Middle East',
    'GUAM': 'Oceania',
    'BULGARIA': 'Europe',
    'LITHUANIA': 'Europe',
    'NEW CALEDONIA': 'Oceania',
    'EGYPT': 'Africa',
    'ARMENIA': 'Asia',
    'BOLIVIA': 'South America',
    'VIRGIN ISLANDS, U.S.': 'North America',
    'PANAMA': 'North America',
    'SIERRA LEONE': 'Africa',
    'COLOMBIA': 'South America',
    'PUERTO RICO': 'North America',
    'UKRAINE': 'Europe',
    'GUINEA': 'Africa',
    'GUADELOUPE': 'North America',
    'MOLDOVA, REPUBLIC OF': 'Europe',
    'GUYANA': 'South America',
    'LATVIA': 'Europe',
    'ZIMBABWE': 'Africa',
    'VANUATU': 'Oceania',
    'VENEZUELA': 'South America',
    'BOTSWANA': 'Africa',
    'BERMUDA': 'North America',
    'MALI': 'Africa',
    'KYRGYZSTAN': 'Asia',
    'CAYMAN ISLANDS': 'North America',
    'MALTA': 'Europe',
    'LEBANON': 'Middle East',
    'REUNION': 'Africa',
    'SEYCHELLES': 'Africa',
    'ZAMBIA': 'Africa',
    'SAMOA': 'Oceania',
    'NORTHERN MARIANA ISLANDS': 'Oceania',
    'NIGERIA': 'Africa',
    'DOMINICAN REPUBLIC': 'North America',
    'TAJIKISTAN': 'Asia',
    'ALBANIA': 'Europe',
    'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF': 'Europe',
    'LIBYAN ARAB JAMAHIRIYA': 'Africa',
    'ANGOLA': 'Africa',
    'BELARUS': 'Europe',
    'TURKS AND CAICOS ISLANDS': 'North America',
    'FAROE ISLANDS': 'Europe',
    'TURKMENISTAN': 'Asia',
    'GUINEA-BISSAU': 'Africa',
    'CAMEROON': 'Africa',
    'BHUTAN': 'Asia',
    'RWANDA': 'Africa',
    'SOLOMON ISLANDS': 'Oceania',
    'IRAN, ISLAMIC REPUBLIC OF': 'Middle East',
    'GUATEMALA': 'North America',
    'FRENCH POLYNESIA': 'Oceania',
    'TIBET': 'Asia',
    'SENEGAL': 'Africa',
    'REPUBLIC OF MONTENEGRO': 'Europe',
    'BOSNIA AND HERZEGOVINA': 'Europe',
}

# Encode regions of destination for better model processing with numerical fields.
region_encoded = {
    'Africa': 0,
    'Asia': 1,
    'Europe': 2,
    'Middle East': 3,
    'North America': 4,
    'Oceania': 5,
    'South America': 6,
}

#Update 'Destination' column based on the mapping
df['Destination'] = df['Destination'].map(destination_mapping).map(region_encoded).astype(int)


# ### Grouping & Encoding Product Column
# 
# Group similar product types, to help reduce the number of unique values. These will then be encoded into numbers so they can be used in the prediction model.
# 
# Each orginal value is below, with its corresponding encoded number.

# In[12]:


plan_mapping = {
    # Group: Comprehensive Plans
    'Comprehensive Plans': [
        '1 way Comprehensive Plan',
        '2 way Comprehensive Plan',
        'Comprehensive Plan',                        
        'Spouse or Parents Comprehensive Plan',
        'Child Comprehensive Plan',
    ],

    # Group: Leveled Plans
    'Leveled Plans': [
        'Basic Plan',
        'Bronze Plan',
        'Gold Plan',
        'Silver Plan',
        'Premier Plan',  
        'Value Plan',
    ],

    # Group: Annual Plans
    'Annual Plans': [
        'Annual Gold Plan',
        'Annual Silver Plan',
        'Annual Travel Protect Gold',
        'Annual Travel Protect Platinum',
        'Annual Travel Protect Silver',
    ],

    # Group: Miscellaneous Plans
    'Miscellaneous Plans': [
        '24 Protect',
        'Cancellation Plan',        
        'Ticket Protector',
        'Rental Vehicle Excess Insurance',
        'Travel Cruise Protect',
        'Travel Cruise Protect Family',
    ],

    # Group: Individual Plans
    'Individual Plans': [
        'Individual Comprehensive Plan',
        'Single Trip Travel Protect Gold',
        'Single Trip Travel Protect Platinum',
        'Single Trip Travel Protect Silver',
    ],
}

# Encode Product for better model processing with numerical fields.
product_encoded = {
    'Annual Plans': 0,
    'Comprehensive Plans': 1,
    'Individual Plans': 2,
    'Leveled Plans': 3,
    'Miscellaneous Plans': 4,
}

#Update 'Product' field based on the mapping
df['Product'] = df['Product'].map({plan: group for group, plans in plan_mapping.items() for plan in plans}).map(product_encoded).astype(int)


# ### Encoding for Agency

# In[13]:


agency_encoding = {
    'CBH': 0,
    'CWT': 1,
    'JZI': 2,
    'KML': 3,
    'EPX': 4,
    'C2B': 5,
    'JWT': 6,
    'RAB': 7,
    'SSI': 8,
    'ART': 9,
    'CSR': 10,
    'CCR': 11,
    'ADM': 12,
    'LWC': 13,
    'TTW': 14,
    'TST': 15
}
df['Agency'] = df['Agency'].map(agency_encoding)


# ### Encoding for Type

# In[14]:


type_encoding = {
    'Travel Agency': 0,
    'Airlines': 1
}

df['Type'] = df['Type'].map(type_encoding)


# ### Distrubution Enconding

# In[15]:


dist_encoding = {
    'Online': 0,
    'Offline': 1
}
df['Dist_Channel'] = df['Dist_Channel'].map(dist_encoding)


# ### Claim Encoding

# In[16]:


claim_encoding = {
    'No': 0,
    'Yes': 1
}
df['Claim'] = df['Claim'].map(claim_encoding)


# # Exploratory Data Analysis

# In[17]:


df[num].describe()


# In[18]:


# Correlation Analysis
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()


# After looking at this correlation matrix, I can see that Sales and Commission have a strong postive correlation, as well as Duration and Sales. While Duration and Commission have moderate postive correlation. So therefore I believe removing Commission will be okay, because I think the Sales and Commission are similar enough that it won't change the prediction much.

# In[19]:


df.drop('Commission',axis=1, inplace=True)


# In[20]:


#update list containing the groups of features
num = num =['Duration', 'Sales','Age']


# While working on this project, I was getting a model that would not predict claims, after re-reading and research I discovered that 'Claim' was imbalanced with a majority of the dataset being "No Claim". I will attempt to fix this by reducing the number of "No Claims" to help balance the class distrubution.

# In[21]:


# Move 'Claim' to the end of the DataFrame
df = df[['Agency', 'Type', 'Dist_Channel', 'Product', 'Duration', 'Destination', 'Sales', 'Age', 'Claim']]


# In[22]:


# Check Claim distribution
df['Claim'].value_counts()


# In[23]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Separate features and target variable
X = df[cat + num]
y = df['Claim']

# Apply RandomUnderSampler to balance distribution
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Display distribution after resampling
print('Claim distribution before resampling:', Counter(y))
print("Claim distribution after resampling:", Counter(y_resampled))


# ## Split and Standardize Dataset

# In[24]:


X = X_resampled
y = y_resampled


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5/30, random_state=101)

#Create an instance of the class
scaler = StandardScaler()

#use fit method
scaler.fit(X_train)

#transform method to perform the transformation
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Random Forests

# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators=99,
                            max_features=6,
                            max_depth=6,
                            min_samples_split=100,
                            random_state=85)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[27]:


accuracy_rf = accuracy_score(y_true=y_test, y_pred=y_pred_rf)
accuracy_rf


# Before resampling I was getting 0.98. But when I did confustion matrix it was showing that it was not predicting any claims only "No Claims".  -- I was worried that I have done something wrong since the accuracy is showing so close to 1. I dont think that it should be that high. I know its a good thing when its that accurate but I feel like that is a false accuracy, however I'm not sure where I went wrong. I've gone over my code to see what I should change and I can not figure it out. I also did the logistic regression model below, to see if it was the same and it was. I'm not sure how to fix where I went wrong.

# In[28]:


from sklearn.metrics import confusion_matrix
def CM(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    out = pd.DataFrame(M, index=["Actual No Claim", "Actual Claim"], columns=["Predicted No Claim", "Predicted Claim"])
    return out

threshold = 0.5
y_pred_prob = rf.predict_proba(X_test)[:,1]
y_pred = (y_pred_prob > threshold).astype(int)

CM(y_test, y_pred)


# ## Logistic Regression

# In[29]:


from sklearn.linear_model import LogisticRegression

# Create and fit the logistic regression model
simple_log_reg = LogisticRegression(C=1e6)
simple_log_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred_logreg = simple_log_reg.predict(X_test)

# Calculate accuracy on the test set
accuracy_logreg = accuracy_score(y_true=y_test, y_pred=y_pred_logreg)
accuracy_logreg


# ## Create Neural Network

# In[30]:


## Building the neural network
n_input = X.shape[1]
n_hidden1 = 32
n_hidden2 = 16
n_hidden3 = 8

nn_reg = Sequential()
nn_reg.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,)))
nn_reg.add(Dense(units=n_hidden2, activation='relu'))
nn_reg.add(Dense(units=n_hidden3, activation='relu'))
# output layer
nn_reg.add(Dense(units=1, activation=None))

## Training the neural network
batch_size = 32
n_epochs = 40
nn_reg.compile(loss='mean_absolute_error', optimizer='adam')
nn_reg.fit(X, y, epochs=n_epochs, batch_size=batch_size)


# In[31]:


## Serializing:
# Scaler
joblib.dump(scaler, './Model/scaler.joblib')

# Trained model
nn_reg.save("./Model/travel_insurance_claim_model.h5")


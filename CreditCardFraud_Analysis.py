#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries required in the DATA Modeling and visulisations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#Iporting the data set from local laptop 
credit_card_data = pd.read_csv(r"/Users/mayankmahajan/Downloads/DME Project/archive/fraudTest.csv")
credit_card_data 


# In[3]:


#Reviewing First 5 row of the Dataset
credit_card_data.head()


# In[4]:


#Reviewing Last 5 row of the Dataset
credit_card_data.tail()


# In[5]:


#Looking at the information/datatype of the dataset
credit_card_data.info()


# In[6]:


#Looking for the missing Value in the dataset
credit_card_data.isnull().sum()


# In[7]:


#Looking for the distribution (number of fraud and legit) in the dataset
credit_card_data['is_fraud'].value_counts()


# In[8]:


#Showing the distribution of legit and fraud transaction as bar graph

count_classes = pd.value_counts(credit_card_data['is_fraud'], sort=True)

LABELS = ["Normal", "Fraud"]  #Defining the labels on X-axis as Normal and fraud

count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")

plt.show()  #this command is used to show the graph


# In[9]:


#Labeling Legit Transaction as 0 and fraud as 1
legit = credit_card_data[credit_card_data.is_fraud == 0]
fraud = credit_card_data[credit_card_data.is_fraud == 1]

#Fraud is the class of interest


# In[10]:


#looking for number of legit and fraud transaction in the total no. of columns
print(legit.shape)
print(fraud.shape)


# In[11]:


#this will show the Statistical data of Legit transaction in the dataframe 
legit.amt.describe()


# In[12]:


#this will show the Statistical data of Fraud transaction in the dataframe 
fraud.amt.describe()


# In[3]:


#looking for Skewness & Kurtosis in the dataset
print("Skewness: %f" % credit_card_data['is_fraud'].skew())
print("Kurtosis: %f" % credit_card_data['is_fraud'].kurt())


# In[14]:


#drawing the histogram for amount per transaction in each class
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.amt, bins = bins)
ax1.set_title('Fraud')
ax2.hist(legit.amt, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[15]:


#Showing the distribution of legit and fraud transaction in Pie Chart
plt.figure(figsize = [4,4])
plot_var = credit_card_data['is_fraud'].value_counts(normalize = True)
plt.pie(plot_var,
        autopct='%1.1f%%',
        labels = ['non_fraud','fraud'], 
        explode = [0.2, 0], 
        shadow = True)
plt.title('Distribution of the Target');


# In[16]:


#Importing Seaborn 
import seaborn as sns


# Selecting only numeric columns for correlation calculation
numeric_cols = credit_card_data.select_dtypes(include=[np.number])

#Finding correlation of each feature in the dataFrame 
corrmat = numeric_cols.corr()
top_corr_features = corrmat.index

# ploting the heat map
plt.figure(figsize=(20,20))
g = sns.heatmap(numeric_cols[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()


# In[17]:


#Listing columns in the dataset 
list(credit_card_data.columns)


# In[18]:


#droping the columns
drop_cols = ["Unnamed: 0","zip","first","last","merchant","trans_num","street","city"]
credit_card_data.drop(drop_cols, axis =1, inplace = True)
#Listing columns in the dataset 
list(credit_card_data.columns)


# In[19]:


#Ensuring That the column "trans_date_trans_time" is in datetime format
credit_card_data['trans_date_trans_time'] = pd.to_datetime(credit_card_data['trans_date_trans_time'])

#Extracting the Hour component
credit_card_data['transaction_hour'] = credit_card_data['trans_date_trans_time'].dt.hour


# In[20]:


#Ensuring That the column "transaction_day" is day name
credit_card_data['transaction_day'] = credit_card_data['trans_date_trans_time'].dt.day_name()


# In[21]:


#Ensuring That the column "transaction_month" is Month name
credit_card_data['transaction_month'] = credit_card_data['trans_date_trans_time'].dt.month


# In[22]:


#Ensuring That the column "transaction_year" is Year
credit_card_data['transaction_year'] = credit_card_data['trans_date_trans_time'].dt.year


# In[23]:


#Ensuring That the column "trans_date_trans_time" &'dob' are in datetime format
credit_card_data['trans_date_trans_time'] = pd.to_datetime(credit_card_data['trans_date_trans_time'])
credit_card_data['dob'] = pd.to_datetime(credit_card_data['dob'])

# coverting days difference into years
credit_card_data['age'] = (credit_card_data['trans_date_trans_time'] - credit_card_data['dob']) / np.timedelta64(1, 'D') / 365.25

#rounding off age to convert into whole number
credit_card_data['age'] = credit_card_data['age'].apply(np.floor)

#Displaying first 5 rows in order to confirm if trasks were accurately done
credit_card_data['age'].head()



# In[24]:


#this will show the Statistical data of cc_num in the dataframe 
credit_card_data.groupby(['cc_num'])['cc_num'].count().sort_values(ascending = False).describe().astype(int)


# In[25]:


#Sorting the data in assending order acc. to the time on which transaction was made
credit_card_data.sort_values(by = ['cc_num','unix_time'], ascending = True, inplace = True)
credit_card_data['unix_time_prev_trans'] = credit_card_data.groupby(by = ['cc_num'])['unix_time'].shift(1)
credit_card_data['unix_time_prev_trans'].fillna(credit_card_data['unix_time'] - 86400, inplace = True)
credit_card_data['timedelta_last_trans'] = (credit_card_data['unix_time'] - credit_card_data['unix_time_prev_trans'])//60


# In[26]:


#Listing columns in the dataset
list(credit_card_data.columns)


# In[27]:


#Calculating the Absolute Latitude Distance
credit_card_data['lat_dist_cust_merch'] = (credit_card_data['lat'] -credit_card_data['merch_lat']).abs()
#Storing the Result in a New Column & Displaying the First Three Entries.
credit_card_data['lat_dist_cust_merch'].head(3)


# In[28]:


#Calculating the Absolute Longitude Distance
credit_card_data['long_dist_cust_merch'] = (credit_card_data['long'] -credit_card_data['merch_long']).abs()

#Storing the Result in a New Column & Displaying the First Three Entries.
credit_card_data['long_dist_cust_merch'].head(3)


# In[29]:


#Grouping Data by Credit Card Number & Shifting the Merchant's Latitude and Longitude
credit_card_data['prev_merch_lat'] = credit_card_data.groupby(by = ['cc_num'])['merch_lat'].shift(1) 

#Storing the Previous Merchant Coordinates in New Columns
credit_card_data['prev_merch_long'] = credit_card_data.groupby(by = ['cc_num'])['merch_long'].shift(1)


# In[30]:


credit_card_data['prev_merch_lat'].fillna(credit_card_data['merch_lat'], inplace = True)

credit_card_data['prev_merch_long'].fillna(credit_card_data['merch_long'], inplace = True)


# In[31]:


#Subtracting Previous Merchant Latitude from Current Merchant Latitude
credit_card_data['lat_dist_prev_merch'] = (credit_card_data['merch_lat'] - credit_card_data['prev_merch_lat']).abs() 

#Storing the Result in a New Column
credit_card_data['lat_dist_prev_merch'].head(3)


# In[32]:


#Subtracting Previous Merchant Longitude from Current Merchant Longitude
credit_card_data['long_dist_prev_merch'] = (credit_card_data['merch_long'] -credit_card_data['prev_merch_long']).abs() 

#Storing the Result in a New Column
credit_card_data['long_dist_prev_merch'].head(3)


# In[33]:


#Listing columns in the dataset
list(credit_card_data.columns)


# In[34]:


#Showing top 5 rows
credit_card_data.head()


# In[4]:


#Identifying Categorical Columns & Standardizing Textual Data
cat_cols = credit_card_data.select_dtypes(include = 'object').columns 

#Analyzing Unique Entries
for col in cat_cols:
    credit_card_data[col] = credit_card_data[col].str.lower().str.strip() 

credit_card_data[cat_cols].nunique().sort_values()


# In[36]:


# Selecting only numeric columns in the dataframe
numeric_data = credit_card_data.select_dtypes(include=[np.number])

# Calculating the correlation matrix for numeric columns
cormat = numeric_data.corr()

cormat


# In[37]:


#Droping columns
drop_cols = ['trans_date_trans_time','transaction_year', 'unix_time','unix_time_prev_trans','dob','lat','long','merch_lat','merch_long','prev_merch_lat','prev_merch_long']
credit_card_data.drop(drop_cols, axis =1, inplace = True)
credit_card_data.head()


# In[38]:


#Listing the Columns
list(credit_card_data.columns)


# In[39]:


#Creating a Copy of the DataFrame & Mapping is_fraud to Descriptive Labels
all = credit_card_data.copy()
all['class'] = all['is_fraud'].map({1:'Fraud',0:'Non_Fraud'})

#Filtering Non-Fraudulent and Fraudulent Transactions
normal = all[credit_card_data['is_fraud'] == 0] 
fraud = all[credit_card_data['is_fraud'] == 1]


# In[40]:


#Calculating Normalized Value Counts for Non-Fraudulent Transactions
def stats(variable):
    n = (normal[variable].value_counts(normalize = True)*100).round(2).rename('normal')
    f = (fraud[variable].value_counts(normalize = True)*100).round(2).rename('fraud')
    return pd.concat([n,f], axis = 1).transpose()


# In[41]:


#Grouping Data by Transaction Class & Aggregating Statistics
def stats_by_class(variable):
    stat_grid = all.groupby('class')[variable].agg([np.min,np.max,np.mean,np.median])
    stat_grid = stat_grid.transpose().round(2)
    return stat_grid


# In[42]:


#Ploting Line graph
plt.figure(figsize = [20,7])

trans_hour_distribution = all.groupby('class')['age'].value_counts(normalize = True).rename('distribution').reset_index()

sns.lineplot(data = trans_hour_distribution, x = 'age', y = 'distribution', hue = 'class') 

plt.xticks(np.arange(10,100,5));

stats_by_class('age')


# In[43]:


#Ploting Box plot
def plot_box (data, x, y, title , width = 6, height = 4):
    plt.figure(figsize = [width,height])
    sns.boxplot(data = data, x = x, y = y)
    plt.title(title)


# In[44]:


#Side by side Box Plot comparison of amount and fraud transaction
plot_box(all,'is_fraud','amt','Distribution of Amount vs Class')

stats_by_class('amt')


# In[45]:


#Side by side histogram comparison of amount of tansactions
fig, ax = plt.subplots(1,3,figsize=(20,5))
ax[0].hist(credit_card_data[credit_card_data.amt<=1500].amt, bins=50)
ax[1].hist(credit_card_data[(credit_card_data.is_fraud==0) & (credit_card_data.amt<=1500)].amt, bins=50)
ax[2].hist(credit_card_data[(credit_card_data.is_fraud==1) & (credit_card_data.amt<=1500)].amt, bins=50)

ax[0].set_title('Overall Amt Distribution')
ax[1].set_title('Non Fraud Amt Distribution')
ax[2].set_title('Fraud Amt Distribution')

ax[0].set_xlabel('Transaction Amount')
ax[0].set_ylabel('#.of Transactions')

ax[1].set_xlabel('Transaction Amount')
ax[2].set_xlabel('Transaction Amount')
plt.show()


# In[46]:


#Ploting Line graph
plt.figure(figsize = [10,5])

trans_hour_distribution = all.groupby('class')['transaction_hour'].value_counts(normalize = True).rename('distribution').reset_index()

sns.lineplot(data = trans_hour_distribution, x = 'transaction_hour', y = 'distribution', hue = 'class')
plt.xticks(np.arange(0,24,1))

plt.show()


# In[47]:


#Setting Up Plot Dimensions & Calculating Normalized Distributions
def normalize_count_by_class(variable, width = 20, height = 7):
    plt.figure(figsize = [width,height])
    normalized_normal = (normal.groupby('class')[variable].value_counts(normalize = True)*100).rename('value').reset_index() # calculate the normalized value for normal transactions 
    normalized_fraud = (fraud.groupby('class')[variable].value_counts(normalize = True)*100).rename('value').reset_index() # calculate the normalized valued for the fraud transactions
    plot_table = pd.concat([normalized_normal.set_index(variable)[['class','value']],
                             normalized_fraud.set_index(variable)[['class','value']]], axis = 0).reset_index()
    sns.barplot(data = plot_table, x = variable, y = 'value', hue = 'class')
    plt.title('\nNormalized frequency of the varible < '+variable+' > on both classes\n')
    plt.xticks(rotation = 30);
    summary_table = pd.concat([normalized_normal.set_index(variable)['value'],
                             normalized_fraud.set_index(variable)['value']],
                            axis = 1).reset_index() 
    summary_table.columns = [variable, 'normal', 'fraud'] 
    summary_table['diff in %'] = (summary_table['fraud'] - summary_table['normal'])
    summary_table.sort_values(by = 'diff in %', ascending = True, inplace = True)
    del normalized_normal,normalized_fraud,plot_table
    print('\nNormalized frequency of < '+variable+' > on both classes and the percentage diffrence\n')
    return summary_table


# In[48]:


normalize_count_by_class('transaction_day')


# In[49]:


#Listing the Columns
list(credit_card_data.columns)


# In[50]:


number_collunms = credit_card_data[['amt', 'city_pop', 'transaction_hour', 'transaction_month', 'timedelta_last_trans',
       'age', 'lat_dist_cust_merch', 'long_dist_cust_merch',
       'lat_dist_prev_merch', 'long_dist_prev_merch']]

number_collunms


# In[51]:


#Side by side box plox 
plt.figure(figsize = [25,10])

for ind,col_name in enumerate(number_collunms):
    plot_var = credit_card_data[col_name]
    plt.subplot(3,4,ind+1)
    sns.boxplot(plot_var)
    plt.title(col_name)
    plt.axis(False)


# In[52]:


#Getting upper and lower cap on amount, population and transaction

from feature_engine.outliers import Winsorizer
variables = ['amt', 'city_pop', 'timedelta_last_trans'] # outlier handaling variables 

capper_iqr = Winsorizer(capping_method = 'iqr',tail = 'both', fold = 1.5, variables = variables) # selecting IQR method 
capper_iqr.fit(credit_card_data)
print('upper capping value : ',capper_iqr.right_tail_caps_)

print('lower capping value : ',capper_iqr.left_tail_caps_)
train_symmetric_X = capper_iqr.transform(credit_card_data) 

df = capper_iqr.transform(credit_card_data) 


# In[53]:


## Take some sample of the data

#credit_card_data1 = credit_card_data.sample(frac = 0.1,random_state=1)

#credit_card_data1.shape


# In[54]:


#credit_card_data.shape


# In[55]:


#Determine the number of fraud and valid transactions in the dataset

#Fraud = credit_card_data1[credit_card_data1['is_fraud']==1]

#Valid = credit_card_data1[credit_card_data1['is_fraud']==0]

#outlier_fraction = len(Fraud)/float(len(Valid))


# In[56]:


#print(outlier_fraction)

#print("Fraud Cases : {}".format(len(Fraud)))

#print("Valid Cases : {}".format(len(Valid)))


# In[57]:


list(credit_card_data.columns)


# In[58]:


cat_cols = credit_card_data.select_dtypes(include = 'object').columns 

for col in cat_cols:
    credit_card_data[col] = credit_card_data[col].str.lower().str.strip() 

credit_card_data[cat_cols].nunique().sort_values()


# In[59]:


from feature_engine.encoding import OneHotEncoder

#using onehotencoder to improve the dataset
variable = ["category","gender"]
onehot_encod = OneHotEncoder(variables = variable, drop_last = True)
onehot_encod.fit(credit_card_data)


# In[60]:


credit_card_data = onehot_encod.transform(credit_card_data) 
credit_card_data.head()


# In[61]:


from feature_engine.encoding import MeanEncoder

#using MeanEncoder to improve the dataset
variables = ['state','transaction_day','job']
mean_encod = MeanEncoder(variables = variables)
mean_encod.fit(credit_card_data,y = credit_card_data["is_fraud"])


# In[62]:


mean_encod.encoder_dict_


# In[63]:


df = mean_encod.transform(credit_card_data)
df.head()


# In[64]:


# separating the data for analysis
true = df[df.is_fraud == 0]
Fraud = df[df.is_fraud == 1]


# In[65]:


print(true.shape)
print(Fraud.shape)


# In[66]:


#Applying oversampling
True_sample = true.sample(n=2145)


# In[67]:


new_dataset = pd.concat([True_sample, Fraud], axis=0)


# In[68]:


new_dataset.head()


# In[69]:


new_dataset.tail()


# In[70]:


new_dataset['is_fraud'].value_counts()


# In[71]:


print(new_dataset.dtypes)


# In[72]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[73]:


model = LogisticRegression()

# selecting RFE features for data fitting 
rfe = RFE(model, n_features_to_select=23)
rfe.fit(new_dataset,new_dataset['is_fraud'])


# Printing the selected features
print("Selected Features: ")
print(new_dataset.columns[rfe.support_])


# In[74]:


#Droping CC_num Columns
drop_cols = ['cc_num']
new_dataset.drop(drop_cols, axis =1, inplace = True)
new_dataset.head()


# In[75]:


#Dividing the dataset into training and validation set
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(new_dataset.drop(["is_fraud"], axis =1),new_dataset["is_fraud"],test_size=0.30,random_state=42)


# In[76]:


#shape of training and validation set
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[77]:


#Using MinMaxScaler function for getting max. and Min. data 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
scaler.data_max_
scaler.data_min_

X_train = pd.DataFrame(data = scaler.transform(X_train), columns = X_train.columns) 

X_valid = pd.DataFrame(data = scaler.transform(X_valid), columns = X_valid.columns)


# In[78]:


X_train.head()


# In[79]:


#Below is the code for finding F1 Score, AUC,Roc.
from sklearn.neighbors import KNeighborsClassifier
def confusion(X_train, y_train, X_valid, y_valid, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    
    cm = confusion_matrix(y_valid, pred)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    print(classification_report(y_valid, pred), '\n')
    
    RocCurveDisplay.from_estimator(estimator = model, X = X_valid, y = y_valid)
    
    f_1 = dict([("f1_score_binary", f1_score(y_valid, pred, average="binary")),
                 ("f1_score_micro", f1_score(y_valid, pred, average="micro")),
                ("f1_score_macro", f1_score(y_valid, pred, average="macro")),
                ("f1_score_weighted", f1_score(y_valid, pred, average="weighted"))
                 ])
    mcc = matthews_corrcoef(y_valid, pred)
    
    tn, fp, fn, tp = confusion_matrix(y_valid, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    
    try:
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1])
        skplt.metrics.plot_cumulative_gain(y_valid, model.predict_proba(X_valid))
    except AttributeError:
        auc = None
        print("predict_proba is not available when probability=False")
    except Exception as e:
        auc = None
        print(e)
    
    plt.show()
    return f_1, mcc, auc


# In[80]:


#Importing Lib. for ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import discriminant_analysis


# In[89]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import classification_report
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score



# In[86]:


#Running the algorithms
algorithms = [RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(), MultinomialNB(), discriminant_analysis.LinearDiscriminantAnalysis(), MLPClassifier(), SVC(), KNeighborsClassifier()]

algo = []
for algorithm in algorithms:
    print(type(algorithm).__name__)

    f_score, MCC, AUC = confusion(X_train, y_train, X_valid, y_valid,model= algorithm)
    algo.append([type(algorithm).__name__, f_score, MCC, AUC])
print(algo)


# In[87]:


#Below is the code for finding F1 Score, AUC,Roc.

from sklearn.neighbors import KNeighborsClassifier
def confusion(X_resampled, y_resampled, X_valid, y_valid, model):
    
    model.fit(X_resampled, y_resampled)
    pred = model.predict(X_valid)
    
    cm = confusion_matrix(y_valid, pred)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    print(classification_report(y_valid, pred), '\n')
    
    RocCurveDisplay.from_estimator(estimator = model, X = X_valid, y = y_valid)
    
    f_1 = dict([("f1_score_binary", f1_score(y_valid, pred, average="binary")),
                 ("f1_score_micro", f1_score(y_valid, pred, average="micro")),
                ("f1_score_macro", f1_score(y_valid, pred, average="macro")),
                ("f1_score_weighted", f1_score(y_valid, pred, average="weighted"))
                 ])
    mcc = matthews_corrcoef(y_valid, pred)
    
    tn, fp, fn, tp = confusion_matrix(y_valid, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    
    try:
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1])
        skplt.metrics.plot_cumulative_gain(y_valid, model.predict_proba(X_valid))
    except AttributeError:
        auc = None
        print("predict_proba is not available when probability=False")
    except Exception as e:
        auc = None
        print(e)
        
    plt.show()
    
    return f_1, mcc, auc


# In[92]:


from imblearn.over_sampling import SMOTE

# Resmapeling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Running the loop in Resmapeling
algo = []
for algorithm in algorithms:
    print(type(algorithm).__name__)
    f_score, MCC, AUC = confusion(X_resampled, y_resampled, X_valid, y_valid, model=algorithm)
    algo.append([type(algorithm).__name__, f_score, MCC, AUC])

print(algo)


# In[ ]:





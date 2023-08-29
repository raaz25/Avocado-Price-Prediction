#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df =pd.read_csv(r"D:\Documents\Avocado\avocado.csv")


# In[4]:


#top 5 rows of the dataset
df.head()


# In[5]:


df.shape


# In[6]:


# Some relevant columns in the dataset:

# Date - The date of the observation

# AveragePrice - the average price of a single avocado

# type - conventional or organic

# year - the year

# Region - the city or region of the observation

# Total Volume - Total number of avocados sold

# 4046 - Total number of avocados with PLU 4046 sold  (Small Hass)
# 4225 - Total number of avocados with PLU 4225 sold  (Large Hass)
# 4770 - Total number of avocados with PLU 4770 sold  (XLarge Hass)


# In[7]:


# Weekly 2018 retail scan data for National retail volume (units) and price.

# Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. 

# The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. 

# The Product Lookup codes (PLU’s) in the table are only for Hass avocados.

# Other varieties of avocados (e.g. greenskins) are not included in this table.


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


# To summarise the dataset we see;

# 14 columns (variables) and 18249 rows (observations)

# There isn't any NULL variable

# data types: float64(9), int64(2), object(3)

# there are some unnamed/undefined columns

# 'region','type' and 'date' columns are in object format


# In[11]:


# Target of this project is to predict the future price of avocados depending on those variables we have; 

# * Type     *Bags(4 units) vs Bundle(one unit)     *Region      *Volume      *Size     *Years


# # Data Preprocessing

# In[12]:


#dropping unnamed column and renaming the columns
df.drop(['Unnamed: 0'],axis=1,inplace=True)

df.rename(columns={'4046':"Small Hass",'4225':'Large Hass','4770':'XLarge Hass'},inplace=True)


# In[13]:


#convert Date column's format
df['Date']=pd.to_datetime(df['Date'])

df.sort_values(by=['Date'],inplace=True,ascending=True)

df.head()


# In[14]:


# Average price of Conventional Avocados over time

fig=plt.figure(figsize=(26,7))
plt.scatter(df[df['type']=='conventional']['Date'],df[df['type']=='conventional']['AveragePrice'],cmap='plasma',c=df[df['type']=='conventional']['AveragePrice'])
plt.title('Average Price of Conventional Avocados Over Time',fontsize=25)
plt.xlabel('Date',fontsize=16)
plt.ylabel('Average Price',fontsize=16)
plt.show()


# In[15]:


# Average price of Organic Avocados over time

fig=plt.figure(figsize=(26,7))
plt.scatter(df[df['type']=='organic']['Date'],df[df['type']=='organic']['AveragePrice'],cmap='plasma',c=df[df['type']=='organic']['AveragePrice'])
plt.title('Average Price of Organic Avocados Over Time',fontsize=25)
plt.xlabel('Date',fontsize=16)
plt.ylabel('Average Price',fontsize=16)
plt.show()


# In[16]:


# Dropping the Date column (date format is not suitable for next level analysis (i.e. OHE))

df=df.drop(['Date'],axis=1)


# In[17]:


# Checking if the sample is balanced

df.groupby('region').size()  # Approximately, there are 338 observations from each region, sample seems balanced


# In[18]:


len(df['region'].unique())


# In[19]:


df['region'].unique()


# In[20]:


# basically we can remove states and work on cities rather than analysing both (to prevent multicollinerarity)

regionsToRemove = ['California', 'GreatLakes', 'Midsouth', 'NewYork', 'Northeast', 'SouthCarolina', 'Plains', 'SouthCentral', 'Southeast', 'TotalUS', 'West']
df=df[~df['region'].isin(regionsToRemove)]
len(df['region'].unique())


# In[21]:


# The average prices by regions

plt.figure(figsize=(10,15))
plt.title("Average Price of Avocado by Region")
sns.barplot(x=df['AveragePrice'],y=df['region'])


# In[22]:


type_counts=df.groupby('type').size()
type_counts

# Types of avocados are also balanced since the ratio is almost 0.5


# In[23]:


# The average prices of avocados by types; organic or not

plt.figure(figsize=(4,5))
plt.title("Average prices of avocados by types")
sns.barplot(x=df['type'],y=df['AveragePrice'])


# In[24]:


df[['Small Bags', 'Large Bags', 'XLarge Bags','Small Hass', 'Large Hass','XLarge Hass','Total Volume','Total Bags']].corr()


# In[25]:


plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# In[26]:


# There is a high correlation between those pairs: 
# small hass & total volume  (0.89)      
# total bags & total volume  (0.87)      
# small bags & total bags    (0.96)      

# Small Hass avocados are the most preferred/sold type in the US and customers tend to buy those avocados as bulk, not bag.
# Retailers want to increase the sales of bagged avocados instead of bulks. They think this is more advantageous for them.
# Total Bags variable has a very high correlation with Total Volume (Total Sales) and Small Bags, so we can say that most of the bagged sales comes from the small bags.


# In[27]:


df_V = df.drop(['AveragePrice', 'Total Volume', 'Total Bags'], axis = 1).groupby('year').sum()
df_V


# In[28]:


#pie chart for 2015 Volume Distribution
labels=['Small Hass','Large Hass','XLarge Hass','Small Bags','Large Bags','XLarge Bags']
plt.pie(df_V.loc[2015].tolist(),labels=labels,colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'],autopct='%1.1f%%')
plt.title('2015 Volume Distribution')
plt.show()


# In[29]:


#pie chart for 2016 Volume Distribution
labels=['Small Hass','Large Hass','XLarge Hass','Small Bags','Large Bags','XLarge Bags']
plt.pie(df_V.loc[2016].tolist(),labels=labels,colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'],autopct='%1.1f%%')
plt.title('2016 Volume Distribution')
plt.show()


# In[30]:


#pie chart for 2017 Volume Distribution
labels=['Small Hass','Large Hass','XLarge Hass','Small Bags','Large Bags','XLarge Bags']
plt.pie(df_V.loc[2017].tolist(),labels=labels,colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'],autopct='%1.1f%%')
plt.title('2017 Volume Distribution')
plt.show()


# In[31]:


#pie chart for 2018 Volume Distribution
labels=['Small Hass','Large Hass','XLarge Hass','Small Bags','Large Bags','XLarge Bags']
plt.pie(df_V.loc[2018].tolist(),labels=labels,colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'],autopct='%1.1f%%')
plt.title('2018 Volume Distribution')
plt.show()


# In[32]:


# Total Bags = Small Bags + Large Bags + XLarge Bags
df=df.drop(['Total Bags'],axis=1)


# In[33]:


# Total Volume = Small Hass +Large Hass +XLarge Hass + Total Bags , to avoid multicollinearity I also drop Total Volume column.


df = df.drop(['Total Volume'], axis = 1)


# In[34]:


df.info()


# In[35]:


# Standardizing (scaling) the variables

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df.loc[:,'Small Hass':'XLarge Bags']= scaler.fit_transform(df.loc[:,'Small Hass':'XLarge Bags']) 
df.head()


# In[36]:


df.round(3).head()


# In[37]:


# Specifying dependent and independent variables

X=df.drop('AveragePrice',axis=1)
y=df['AveragePrice']
y=np.log1p(y)


# In[38]:


# Labeling the categorical variables

Xcat=pd.get_dummies(X[['type','region']],drop_first=True)


# In[39]:


Xnum=X[["Small Hass","Large Hass","XLarge Hass","Small Bags","Large Bags","XLarge Bags"]]


# In[40]:


# Concatenate dummy categorcal variables and numeric variables
X=pd.concat([Xcat,Xnum],axis=1)
X.shape


# In[41]:


F_DF=pd.concat([y,X],axis=1)
F_DF.head(3)


# In[42]:


# Just before the regression analysis, I want to visualise the highly correlated Variables with the Average Prices;

sns.set(color_codes=True)
sns.jointplot(x="Small Hass", y="AveragePrice", data=F_DF, kind="reg")
sns.jointplot(x="Small Bags", y="AveragePrice", data=F_DF, kind="reg")
sns.jointplot(x="Large Bags", y="AveragePrice", data=F_DF, kind="reg")


# In[43]:


sns.lmplot(x="Small Hass", y="AveragePrice", col="type_organic", data=F_DF, col_wrap=2)

# Graphs depict that organic avocados have less elasticity to the price, compared to conventional ones.


# In[44]:


# TRAIN and TEST SPLIT

# Since the data is a time series data (gives weekly avocado prices between Jan 2015 and Apr 2018)
# I sort it by Date and then split it due to date manually (not randomly), to preserve the 'times series effect' on it.
# I determined the split ratio as 0.30, so train and test data are just as follows;


X_train=X[0:10172]
y_train=y[0:10172]
X_test=X[10172:]
y_test=y[10172:]


# In[45]:


#implementing Machine Learning Models


# In[46]:


#Multiple Linear Regression

from sklearn.linear_model import LinearRegression
LinReg=LinearRegression()
LinReg.fit(X_train,y_train)

print("R2 of Linear Regression:",LinReg.score(X_train,y_train))


# In[47]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[48]:


print('MAE: ',mean_absolute_error(y_test,LinReg.predict(X_test)))
print('MSE: ',mean_squared_error(y_test,LinReg.predict(X_test)))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,LinReg.predict(X_test))))


# In[49]:


# Creating a Histogram of Residuals

plt.figure(figsize=(5,4))
sns.distplot(y_test-LinReg.predict(X_test))
plt.title('Distribution of Residuals')


# In[50]:


#Scatter plot of y_test and y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_test,LinReg.predict(X_test))


# In[51]:


# we can confirm the R2 value (moreover, get the R2 Adj.value) of the model by statsmodels library of python

import statsmodels.api as sm
X_train=sm.add_constant(X_train)
model=sm.OLS(y_train,X_train).fit()
print(model.summary())


# In[52]:


X_train=X[0:10172]
y_train=y[0:10172]
X_test=X[10172:]
y_test=y[10172:]


# In[53]:


# LASSO and RIDGE Regressions

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-5,3,20)

clf = GridSearchCV(estimator=linear_model.Ridge(), param_grid=dict(alpha=alphas), cv=10)
clf.fit(X_train, y_train)
optlamGSCV_R = clf.best_estimator_.alpha
print('Optimum regularization parameter (Ridge):', optlamGSCV_R)

clf = GridSearchCV(estimator=linear_model.Lasso(), param_grid=dict(alpha=alphas), cv=10)
clf.fit(X_train, y_train)
optlamGSCV_L= clf.best_estimator_.alpha
print('Optimum regularization parameter (Lasso):', optlamGSCV_L)


# In[55]:


from sklearn.metrics import mean_squared_error

ridge = linear_model.Ridge(alpha = optlamGSCV_R) 
ridge.fit(X_train, y_train)
print('RMSE value of the Ridge Model is: ',np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))))


# In[56]:


ridge.score(X_train, y_train) #Returns the coefficient of determination (R2) of the prediction.


# In[57]:


# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_test - ridge.predict(X_test))
plt.title('Distribution of residuals')


# In[59]:


lasso = linear_model.Lasso(alpha = optlamGSCV_L)
lasso.fit(X_train, y_train)
print('RMSE value of the Lasso Model is: ',np.sqrt(mean_squared_error(y_test, lasso.predict(X_test))))


# In[60]:


lasso.score(X_train, y_train) #Returns the coefficient of determination R^2 of the prediction.


# In[61]:


# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_test - lasso.predict(X_test))
plt.title('Distribution of residuals')


# In[65]:


coef = pd.Series(lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  
      str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values()]) #plot all
plt.rcParams['figure.figsize'] = (7.0, 30.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# In[66]:


# According to the RMSE results, Ridge works best compared to linear regression and lasso.

# Let's see the other ML Models' RMSE values;


# In[67]:


# KNN Regressor


# In[68]:


from sklearn.neighbors import KNeighborsRegressor
from math import sqrt

knn=KNeighborsRegressor()
knn.fit(X_train,y_train)


# In[69]:


mse=sqrt(mean_squared_error(y_test,knn.predict(X_test)))
print('RSME of KNN model is: ',mse)


# In[70]:


print('R2 of KNN model is: ',knn.score(X_train,y_train))


# In[71]:


# SVR Regressor

# First, let's choose which kernel is the best for our data

from sklearn.svm import SVR

for k in ['linear','poly','rbf','sigmoid']:
    clf=SVR(kernel=k)
    clf.fit(X_train,y_train)
    confidence=clf.score(X_train,y_train)
    print(k,confidence)


# In[72]:


Svr=SVR(kernel='rbf', C=1, gamma= 0.5)   # Parameter Tuning to get the best accuracy

# Intuitively, the gamma defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’.
# The C parameter trades off correct classification of training examples against maximization of the decision function’s margin. 
# For larger values of C, a smaller margin will be accepted if the decision function is better at classifying all training points correctly. 
# A lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. 
# In other words C behaves as a regularization parameter in the SVM.

Svr.fit(X_train,y_train)
print(Svr.score(X_train,y_train))


# In[73]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,Svr.predict(X_test))
rsme=sqrt(mean_squared_error(y_test,Svr.predict(X_test)))
print("MSE calculated is: ",mse)
print("RSME calculated is: ",rsme)


# In[74]:


Svr.predict(X_test)[0:5]


# In[75]:


#Decision Tree Regressor


# In[76]:


# Determining the best depth
from sklearn.tree import DecisionTreeRegressor

minDepth = 100
minRMSE = 100000


for depth in range(2,10):
    tree_reg = DecisionTreeRegressor(max_depth=depth)
    tree_reg.fit(X_train, y_train)
    y_pred = tree_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Depth:",depth,", MSE:", mse)
    print("Depth:",depth, ",RMSE:", rmse)
  
    if rmse < minRMSE:
        minRMSE = rmse
        minDepth = depth
    
      
print("MinDepth:", minDepth)
print("MinRMSE:", minRMSE)


# In[77]:


DTree=DecisionTreeRegressor(max_depth=minDepth)
DTree.fit(X_train,y_train)
print(DTree.score(X_train,y_train))  


# In[78]:


from sklearn.metrics import mean_absolute_error
print('MAE:',mean_absolute_error(y_test, DTree.predict(X_test)))
print('MSE:',mean_squared_error(y_test, DTree.predict(X_test)))
print('RMSE:', np.sqrt(mean_squared_error(y_test, DTree.predict(X_test))))


# In[79]:


#RandomForest Regressor


# In[80]:


from sklearn.ensemble import RandomForestRegressor
RForest = RandomForestRegressor()
RForest.fit(X_train,y_train)
print(RForest.score(X_train,y_train))  


# In[81]:


from sklearn.metrics import mean_absolute_error

print('MAE:',mean_absolute_error(y_test, RForest.predict(X_test)))
print('MSE:',mean_squared_error(y_test, RForest.predict(X_test)))
print('RMSE:', np.sqrt(mean_squared_error(y_test, RForest.predict(X_test))))


# In[82]:


# CONCLUSION 

# Comparing The RMSE Values Of The Models


# In[83]:


# Linear Regression RMSE : 
print('RMSE value of the Linear Regr : ',round(np.sqrt(mean_squared_error(y_test, LinReg.predict(X_test))),4))

# KNN RMSE               : 
print('RMSE value of the KNN Model   : ',round(np.sqrt(mean_squared_error(y_test, knn.predict(X_test))),4))

# SVR RMSE               : 
print('RMSE value of the SVR Model   : ',round(np.sqrt(mean_squared_error(y_test, Svr.predict(X_test))),4))

# Decision Tree RMSE     : 
print('RMSE value of the Decis Tree  : ',round(np.sqrt(mean_squared_error(y_test, DTree.predict(X_test))),4))

# Random Forest RMSE     : 
print('RMSE value of the Rnd Forest  : ',round(np.sqrt(mean_squared_error(y_test, RForest.predict(X_test))),4))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





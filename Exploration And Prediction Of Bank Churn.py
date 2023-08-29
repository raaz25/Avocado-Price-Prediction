#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo


# # Data Loading

# In[2]:


c_data=pd.read_csv(r"C:\Users\Raj Aryan\Downloads\BankChurners.csv")
c_data=c_data[c_data.columns[:-2]]
c_data.head(10)


# # EDA

# In[3]:


fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(x=c_data['Customer_Age'], name='Age Box Plot',boxmean=True)
tr2 = go.Histogram(x=c_data['Customer_Age'], name='Age Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="<b>Distribution of Customer Ages<b>")
fig.show()


# #### We can see that the distribution of customer ages in our dataset follows a fairly normal distribution; thus, further use of the age feature can be done with the normality assumption.

# In[4]:


fig = make_subplots(
    rows=2, cols=2,subplot_titles=('','<b>Platinum Card Holders','<b>Blue Card Holders<b>','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 2}       ,{"type": "pie"}] ,
           [None                               ,{"type": "pie"}]            ,                                      
          ]
)

fig.add_trace(
    go.Pie(values=c_data.Gender.value_counts().values,labels=['<b>Female<b>','<b>Male<b>'],hole=0.3,pull=[0,0.3]),
    row=1, col=1
)

fig.add_trace(
    go.Pie(
        labels=['Female Platinum Card Holders','Male Platinum Card Holders'],
        values=c_data.query('Card_Category=="Platinum"').Gender.value_counts().values,
        pull=[0,0.05,0.5],
        hole=0.3
        
    ),
    row=1, col=2
)

fig.add_trace(
    go.Pie(
        labels=['Female Blue Card Holders','Male Blue Card Holders'],
        values=c_data.query('Card_Category=="Blue"').Gender.value_counts().values,
        pull=[0,0.2,0.5],
        hole=0.3
    ),
    row=2, col=2
)



fig.update_layout(
    height=800,
    showlegend=True,
    title_text="<b>Distribution Of Gender And Different Card Statuses<b>",
)

fig.show()


# #### More samples of females in our dataset are compared to males, but the percentage of difference is not that significant, so we can say that genders are uniformly distributed.

# In[5]:


fig=make_subplots(rows=2,cols=1)

tr1=go.Box(x=c_data['Dependent_count'],boxmean=True,name='Dependent Count Boxplot')
tr2=go.Histogram(x=c_data['Dependent_count'],name='Dependent Count Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700,width=1200,title_text='Distribution of Dependent Count(close family size)')
fig.show()


# #### The distribution of Dependent counts is fairly normally distributed with a slight right skew.

# In[6]:


ex.pie(c_data,names='Education_Level',title='Proportion of Education Levels',hole=0.33)


# #### If most of the customers with unknown education status lack any education, we can state that more than 70% of the customers have a formal education level. About 35% have a higher level of education.

# In[7]:


ex.pie(c_data,names='Marital_Status',title='Proportion of Different Marital Statuses',hole=0.33)


# #### Almost half of the bank customers are married, and interestingly enough, almost the entire other half are single customers. only about 7% of the customers are divorced, which is surprising considering the worldwide divorce rate statistics! (let me know where the bank is located and sign me up!)

# In[8]:


ex.pie(c_data,names='Income_Category',title='Proportion of Different Income Levels',hole=0.33)


# In[9]:


ex.pie(c_data,names='Card_Category',title='Distribution of Different Card Categories')


# In[10]:


fig=make_subplots(rows=2,cols=1)

tr1=go.Box(x=c_data['Months_on_book'],name='Months On Book Box Plot',boxmean=True)
tr2=go.Histogram(x=c_data['Months_on_book'],name='Months On Book Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700,width=1200,title_text='Distribution of months the customer is part of the bank')
fig.show()


# In[11]:


print(f"Kurtosis of months on book features is:{c_data['Months_on_book'].kurt()}")


# #### We have a low kurtosis value pointing to a very flat shaped distribution (as shown in the plots above as well), meaning we cannot assume normality of the feature.

# In[12]:


fig=make_subplots(rows=2,cols=1)

tr1=go.Box(x=c_data['Total_Relationship_Count'],boxmean=True,name='Total number of Customers Box Plot')
tr2=go.Histogram(x=c_data['Total_Relationship_Count'],name='Total number of Customers Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700,width=1200,title_text='Distribution of Total no. of products held by the customer')
fig.show()


# #### The distribution of the total number of products held by the customer seems closer to a uniform distribution and may appear useless as a predictor for churn status.

# In[13]:


fig=make_subplots(rows=2,cols=1)

tr1=go.Box(x=c_data['Months_Inactive_12_mon'],boxmean=True,name='Number of Customers Inactive Boxplot')
tr2=go.Histogram(x=c_data['Months_Inactive_12_mon'],name='Number of Customers Inactice Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700,width=1200,title_text='Distribution of the number of months inactive in the last 12 months')
fig.show()


# In[14]:


fig=make_subplots(rows=2,cols=1)

tr1=go.Box(x=c_data['Credit_Limit'],boxmean=True,name='Credit Limit Boxplot')
tr2=go.Histogram(x=c_data['Credit_Limit'],name='Credit Limit Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700,width=1200,title_text='Distribution of Credit Limit')
fig.show()


# In[15]:


fig=make_subplots(rows=2,cols=1)

tr1=go.Box(x=c_data['Total_Trans_Amt'],boxmean=True,name='Total Transaction Amt Boxplot')
tr2=go.Histogram(x=c_data['Total_Trans_Amt'],name='Total Transaction Amt Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700,width=1200,title_text='Distribution of Total Transaction Amount(Lst 12 Months)')
fig.show()


# #### We see that the distribution of the total transactions (Last 12 months) displays a multimodal distribution, meaning we have some underlying groups in our data; it can be an interesting experiment to try and cluster the different groups and view the similarities between them and what describes best the different groups which create the different modes in our distribution.

# In[16]:


ex.pie(c_data,names='Attrition_Flag',title='Proportion of churn vs not churn customers',hole=0.33)


# #### As we can see, only 16% of the data samples represent churn customers; in the following steps, I will use SMOTE to upsample the churn samples to match them with the regular customer sample size to give the later selected models a better chance of catching on small details which will almost definitely be missed out with such a size difference. 

# # Data Preprocessing

# In[17]:


c_data['Attrition_Flag']=c_data['Attrition_Flag'].replace({'Attrited Customer':1,'Existing Customer':0})
c_data['Gender']=c_data['Gender'].replace({'M':1,'F':0})
c_data=pd.concat([c_data,pd.get_dummies(c_data['Education_Level']).drop(columns=['Unknown'])],axis=1)
c_data=pd.concat([c_data,pd.get_dummies(c_data['Income_Category']).drop(columns=['Unknown'])],axis=1)
c_data=pd.concat([c_data,pd.get_dummies(c_data['Marital_Status']).drop(columns=['Unknown'])],axis=1)
c_data=pd.concat([c_data,pd.get_dummies(c_data['Card_Category']).drop(columns=['Platinum'])],axis=1)
c_data.drop(columns=['Education_Level','Income_Category','Card_Category','Marital_Status','CLIENTNUM'],inplace=True)


# #### Here we one hot encode all the categorical features describing different statuses of a customer.

# In[18]:


fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Pearson Correlation','Spearman Correlation'])
colorscale=     [[1.0              , "rgb(165,0,38)"],
                [0.8888888888888888, "rgb(215,48,39)"],
                [0.7777777777777778, "rgb(244,109,67)"],
                [0.6666666666666666, "rgb(253,174,97)"],
                [0.5555555555555556, "rgb(254,224,144)"],
                [0.4444444444444444, "rgb(224,243,248)"],
                [0.3333333333333333, "rgb(171,217,233)"],
                [0.2222222222222222, "rgb(116,173,209)"],
                [0.1111111111111111, "rgb(69,117,180)"],
                [0.0               , "rgb(49,54,149)"]]



s_val=c_data.corr('pearson')
s_idx=s_val.index
s_col=s_val.columns
s_val=s_val.values
fig.add_trace(go.Heatmap(x=s_col,y=s_idx,z=s_val,showscale=False,xgap=0.7,ygap=0.7,colorscale=colorscale),row=1,col=1)

s_val=c_data.corr('spearman')
s_idx=s_val.index
s_col=s_val.columns
s_val=s_val.values
fig.add_trace(go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=0.7,ygap=0.7,colorscale=colorscale),row=2,col=1)

fig.update_layout(hoverlabel=dict(bgcolor='White',font_size=16,font_family='Rockwell'))

fig.update_layout(height=700,width=1200,title='Numerical correlation')

fig.show()


# # Data Upsampling Using SMOTE

# In[19]:


from imblearn.over_sampling import SMOTE


# In[20]:


oversample=SMOTE()
X,y = oversample.fit_resample(c_data[c_data.columns[1:]],c_data[c_data.columns[0]])

usampled_df=X.assign(Churn=y)


# In[21]:


ohe_data=usampled_df[usampled_df.columns[15:-1]].copy()

usampled_df=usampled_df.drop(columns=usampled_df.columns[15:-1])


# In[22]:


fig = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion',  'Spearman Correaltion'))
colorscale=     [[1.0              , "rgb(165,0,38)"],
                [0.8888888888888888, "rgb(215,48,39)"],
                [0.7777777777777778, "rgb(244,109,67)"],
                [0.6666666666666666, "rgb(253,174,97)"],
                [0.5555555555555556, "rgb(254,224,144)"],
                [0.4444444444444444, "rgb(224,243,248)"],
                [0.3333333333333333, "rgb(171,217,233)"],
                [0.2222222222222222, "rgb(116,173,209)"],
                [0.1111111111111111, "rgb(69,117,180)"],
                [0.0               , "rgb(49,54,149)"]]

s_val =usampled_df.corr('pearson')
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=1,ygap=1,colorscale=colorscale),
    row=1, col=1
)


s_val =usampled_df.corr('spearman')
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale),
    row=2, col=1
)
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)
fig.update_layout(height=700, width=900, title_text="Upsmapled Correlations")
fig.show()


# In[23]:


#https://www.canva.com/design/DAFrPKEiGDQ/7GzoZmkJkGdCfKnpPci0yg/edit


# # Principal Component Analysis Of One Hot Encoded Data

# #### We will use principal component analysis to reduce the dimensionality of the one-hot encoded categorical variables losing some of the variances, but simultaneously, using a couple of principal components instead of tens of one-hot encoded features will help me construct a better model.

# In[24]:


ohe_data


# In[25]:


from sklearn.decomposition import PCA


# In[26]:


N_COMPONENTS=4

pca_model=PCA(n_components=N_COMPONENTS)
pca_matrix=pca_model.fit_transform(ohe_data)

evr=pca_model.explained_variance_ratio_

total_var=evr.sum()*100

cumsum_evr=np.cumsum(evr)


trace1={'name':'Individual Explained Variance',
       'type':'bar',
       'y':evr}
trace2={'name':"Cumulative Explained Variance",
       'type':'scatter',
       'y':cumsum_evr}

data=[trace1,trace2]

layout={"xaxis":{'title':'Principal Components'},
       "yaxis":{'title':'Explained Variance Ratio'}}

fig=go.Figure(data=data,layout=layout)
fig.update_layout(title="Explained Variance Using 4 Dimensions")
fig.show()


# In[27]:


usampled_df_with_pcs=pd.concat([usampled_df,pd.DataFrame(pca_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)


# In[28]:


fig=ex.scatter_matrix(usampled_df_with_pcs[['PC-{}'.format(i) for i in range(0,N_COMPONENTS)]].values,color=usampled_df_with_pcs['Credit_Limit'],
                     dimensions=range(N_COMPONENTS),labels={str(i):'PC-{}'.format(i) for i in range(0,N_COMPONENTS)},
                      title=f'Total Variance:{total_var:.2f}%')

fig.update_traces(diagonal_visible=False)
fig.update_layout(coloraxis_colorbar=dict(title='Credit_Limit'))
fig.show()


# In[29]:


fig = make_subplots(rows=2,cols=1, shared_xaxes=True,subplot_titles=('Pearson_Correlation','Spearman_Correlation'))

s_val=usampled_df_with_pcs.corr('pearson')
s_idx=s_val.index
s_col=s_val.columns
s_val=s_val.values

fig.add_trace(go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale,name='pearson',showscale=False),row=1,col=1)


s_val=usampled_df_with_pcs.corr('spearman')
s_idx=s_val.index
s_col=s_val.columns
s_val=s_val.values

fig.add_trace(go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale,showscale=True,name='spearman'),row=2,col=1)

fig.update_layout(hoverlabel=dict(bgcolor='white',font_family='Rockwell',font_size=16,))

fig.update_layout(height=900,width=700,title="Upsmapled Correlations With PC\'s")
fig.show()


# # Model Selection And Evaluation

# In[30]:


usampled_df_with_pcs


# In[31]:


X_features=['Total_Trans_Ct','PC-3','PC-1','PC-0','PC-2','Total_Ct_Chng_Q4_Q1','Total_Relationship_Count']

X=usampled_df_with_pcs[X_features]
y=usampled_df_with_pcs['Churn']


# In[32]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)


# # Cross Validation

# In[33]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# In[34]:


rf_pipe=Pipeline(steps=[('scale',StandardScaler()),('Rf',RandomForestClassifier(random_state=42))])
ada_pipe=Pipeline(steps=[('scale',StandardScaler()),("ada",AdaBoostClassifier(random_state=42,learning_rate=0.7))])
svm_pipe=Pipeline(steps=[('scale',StandardScaler()),('svm',SVC(random_state=42,kernel='rbf'))])
#Pipeline-constructor for creating a scikit-learn pipeline so to chain multiple processing steps together

rf_cross_val_score=cross_val_score(rf_pipe,X_train,y_train,cv=5,scoring='f1')
ada_cross_val_score=cross_val_score(ada_pipe,X_train,y_train,cv=5,scoring='f1')
svm_cross_val_score=cross_val_score(svm_pipe,X_train,y_train,cv=5,scoring='f1')


# In[35]:


fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=('RandomForest Cross Val Score','AdaBoost Cross Val Score','SupportVector Cross Val Score'))

fig.add_trace(go.Scatter(x=list(range(0,len(rf_cross_val_score))),y=rf_cross_val_score,name='Random Forest'),row=1,col=1)
fig.add_trace(go.Scatter(x=list(range(0,len(ada_cross_val_score))),y=ada_cross_val_score,name='AdaBoost Classifier'),row=2,col=1)
fig.add_trace(go.Scatter(x=list(range(0,len(svm_cross_val_score))),y=svm_cross_val_score,name='Support Vector Classifier'),row=3,col=1)


fig.update_layout(height=700,width=900,title='Different Model 5 Fold Cross Validation')
fig.update_xaxes(title='Fold #')
fig.update_yaxes(title='F1-score')
fig.show()


# # Model Evaluation

# In[36]:


rf_pipe.fit(X_train,y_train)
rf_prediction=rf_pipe.predict(X_test)

ada_pipe.fit(X_train,y_train)
ada_prediction=ada_pipe.predict(X_test)

svm_pipe.fit(X_train,y_train)
svm_prediction=svm_pipe.predict(X_test)


# In[37]:


from sklearn.metrics import f1_score as f1


# In[38]:


fig=go.Figure(data=[go.Table(header=dict(values=['<b>Model<b>','<b>F1-score on Test Data<b>'],line_color='darkslategray',
                                        fill_color='whitesmoke',font=dict(size=18,color='black'), align=['center','center'],height=40),
                             
                                        cells=dict(values=[['<b>Random Forest<b>','<b>AdaBoost<b>','<b>SVM<b>'],
                                                           [np.round(f1(rf_prediction,y_test),2),
                                                           np.round(f1(ada_prediction,y_test),2),
                                                           np.round(f1(svm_prediction,y_test),2)]]))])

fig.update_layout(title='Model Results On Test Data')
fig.show()


# # Model Evaluation On Original Data (Before Upsampling)

# In[39]:


ohe_data=c_data[c_data.columns[16:]].copy()
pc_matrix=pca_model.fit_transform(ohe_data)
original_df_with_pcs=pd.concat([c_data,pd.DataFrame(pc_matrix,columns=["PC-{}".format(i) for i in range(0,N_COMPONENTS)])],axis=1)

usampled_data_prediction_rf=rf_pipe.predict(original_df_with_pcs[X_features])
usampled_data_prediction_ada=ada_pipe.predict(original_df_with_pcs[X_features])
usampled_data_prediction_svm=svm_pipe.predict(original_df_with_pcs[X_features])


# In[40]:


fig=go.Figure(data=go.Table(header=dict(values=['<b>Model<b>','<b>F1-score on original data<b>'], line_color='darkslategray',
                                       fill_color='whitesmoke',font=dict(size=18,color='black'),align=['center','center'],height=40),
                           cells=dict(values=[['<b>Random Forest<b>','<b>AdaBoost<b>','<b>SVM<b>'],
                                             [np.round(f1(usampled_data_prediction_rf,original_df_with_pcs['Attrition_Flag']),2),
                                             np.round(f1(usampled_data_prediction_ada,original_df_with_pcs['Attrition_Flag']),2),
                                             np.round(f1(usampled_data_prediction_svm,original_df_with_pcs['Attrition_Flag']),2)]])))

fig.update_layout(title='Model Result On Original Data (Without Upsampling)')
fig.show()


# # Results

# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


z=confusion_matrix(usampled_data_prediction_rf,original_df_with_pcs['Attrition_Flag'])
fig=ff.create_annotated_heatmap(z,x=['Not Churn','Churn'],y=['Predicted Not Churn','Predicted Churn'],colorscale='Fall',xgap=3,ygap=3)
fig['data'][0]['showscale']=True

fig.update_layout(title='Prediction On Original Data With Random Forest Model Confusion Matrix')
fig.show()


# In[43]:


import scikitplot as skplt


# In[44]:


usampled_data_prediction_RF=rf_pipe.predict_proba(original_df_with_pcs[X_features])
skplt.metrics.plot_precision_recall(original_df_with_pcs['Attrition_Flag'],usampled_data_prediction_RF)

plt.legend(prop={'size':8})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





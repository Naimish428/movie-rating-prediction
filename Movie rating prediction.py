#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


#Read the data 
df=pd.read_csv("D:/codsoft/IMDb Movies India.csv",encoding='ISO-8859-1')


# In[18]:


#Show the data
df.head(10)


# In[20]:


#The shape of the data
df.shape


# In[22]:


#the column of the data 
df.columns


# In[24]:


#information about the data
df.info()


# In[26]:


#check the null values
df.isnull().sum()


# In[28]:


#some statistics about the data
df.describe()


# In[30]:


#eliminate missing values
df.dropna(inplace=True)


# In[32]:


#recheck for missing values
df.isna().sum()


# In[34]:


df.head()


# In[36]:


df.shape


# In[45]:


#converting the year into integer
df['Year'] = df['Year'].str.strip('()').astype(int)


# In[48]:


#remove the string from the column
df['Duration']=df['Duration'].str.strip('min')


# In[50]:


df.info()


# In[52]:


df.head()


# In[54]:


#visulaze rating 
sns.histplot(data=df,x='Rating',kde=True)
plt.title('Distribution of ratings')
plt.show()


# In[56]:


#visualize the years
sns.histplot(data=df,x='Year',kde=True)
plt.title('Distribution of years')
plt.show()


# In[58]:


#visualze the relation between the year and rating
sns.scatterplot(data=df,x='Year',y='Rating')
plt.title("The relatiob between Year and Rating")
plt.show()


# In[60]:


#the distibution of duratiion over years
sns.lineplot(data=df.head(15),x='Year',y='Duration')
plt.title('The distibution of duration over years')
plt.show()


# In[62]:


#visualze genre, before tat let's get all the genres
movies_genre = df['Genre'].str.split(', ',expand=True).stack().value_counts()
labels = movies_genre.keys()
count = movies_genre.values
plt.figure(figsize=(10,6))
sns.barplot(x=labels,y=count)
plt.xticks(rotation=90)
plt.title('The frequency of each genre in the data')
plt.xlabel('Genre')
plt.ylabel('Counts')
plt.show()


# In[64]:


#Encod the column to use them in the model:
encoder = LabelEncoder()
df['Actor 1'] = encoder.fit_transform(df['Actor 1'])
df['Actor 2'] = encoder.fit_transform(df['Actor 2'])
df['Actor 3'] = encoder.fit_transform(df['Actor 3'])
df['Genre'] = encoder.fit_transform(df['Genre'])
df['Director'] = encoder.fit_transform(df['Director'])
df.head()


# In[67]:


df2 = df.drop('Name',axis=1)


# In[69]:


#Heatmap Showing the correlation between columns
sns.heatmap(df2.corr(),annot=True)
plt.show()


# In[71]:


#split the data 
X = df2.drop('Rating',axis=1)
y= df2['Rating']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[73]:


#initialoze the model
model = LinearRegression()
#Fiting the training data
model.fit(X_train,y_train)


# In[75]:


#predict values
y_pred = model.predict(X_test)
y_pred


# In[77]:


#calculating some metrics
print(f"Mean Absolute Error : {mean_absolute_error(y_test,y_pred)}")
print(f"Mean Squared Error : {mean_squared_error(y_test,y_pred)}")


# In[78]:


#calculating the r2 score
print(f"R2 score : {r2_score(y_test,y_pred)}")


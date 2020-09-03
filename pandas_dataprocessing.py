#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

df = pd.read_csv("http://rcs.bu.edu/examples/python/data_analysis/Salaries.csv")


# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.describe()


# In[6]:


df.max()


# In[7]:


df.min()


# In[8]:


df.mean()


# In[9]:


df.median()


# In[10]:


df.std()


# In[11]:


df.sample(10)


# In[12]:


df.dropna()


# In[13]:


#exercise
# summary for numeric dataset
df.describe()


# In[14]:


#standard deviation
df.std()


# In[18]:


#mean value of 1st 50 records
x=df.head(50)
x.mean()


# In[24]:


#column name as subset
df['rank']


# In[25]:


# coumn name as attributes
df.rank


# In[43]:


# basic statistics for phd column
df.phd.mean()
df.phd.median()
df.phd.std()
df.phd.sample(10)
df.phd.describe()
df.phd.count()
df.phd.mean()


# In[41]:


# group by 
df.groupby(['rank']).mean()


# In[44]:


df.groupby('sex')[['salary']].mean()


# In[45]:


df.groupby('sex')[['salary','phd']].mean()


# In[46]:


# data frame filtering
df.salary>120000


# In[47]:


df[df.salary>120000]


# In[48]:


df[df.sex=='male']


# In[ ]:


# iloc method
# df.iloc[0]     # 1st row of a data frame
# df.iloc[i]     # ith row
# df.iloc[-1]    # last row
# df.iloc[:, 0]  # 1st column
# df.iloc[:,-1]  # last column
# df.iloc[0:7]   # first 7 rows
# df.iloc[:,0:2] # first 2 columns
# df.iloc[1:3,0:2] # 1st & 2nd rows and first 2 columns
# df.iloc[[0,5],[1,3]] # 1st & 6th rows and 2nd & 4th columns


# In[19]:


df.iloc[[0,5],[1,3]]


# In[22]:


df.head(7)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





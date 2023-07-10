
import pandas as pd
import numpy as np
import pickle


# In[2]:


df = pd.read_csv('AUTO_Mpg_Reg.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[9]:


df.horsepower = df.horsepower.fillna(df.horsepower.median())


# In[10]:


df.isnull().sum()


# In[11]:


y = df.mpg


# In[13]:


X = df.drop('mpg',axis = 1)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


reg = LinearRegression()


# In[16]:


reg.fit(X,y)


# In[17]:


reg.score(X,y)


# In[18]:


regpred = reg.predict(X)


# In[19]:


from sklearn.metrics import mean_squared_error


# In[20]:


np.sqrt(mean_squared_error(y,regpred))

pickle.dump(reg,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))




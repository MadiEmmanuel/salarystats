
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('salary.csv')

df.to_csv('salary.csv')

df = pd.read_csv('salary.csv')
df.head()


# In[5]:


#How many responders are there? Are there any missing values in any of the variables?
import pandas as pd
df = pd.read_csv('salary.csv')
print(df['salary'].count())


df.isnull().sum().sum()
null_value_info = df[df['salary'].isnull()]

#Drop null value row

df = df.drop(208)


# In[24]:


#What is the lowest salary and highest salary in the group?
import pandas as pd
df = pd.read_csv('salary.csv')

list = df['salary']

max_value = max(list)
print('Max value is:' ,max_value)

min_value = min(list)
print('Min value is:' ,min_value)


# In[18]:


#What is the mean salary for the sample? Include the standard error of the mean.
from statistics import mean, median
import scipy
from scipy import stats

import pandas as pd
df = pd.read_csv('salary.csv')
df = df.drop(208)

somelist =  df['salary']
#avg_value = stats.sem(somelist)
#print(avg_value)

salary_mean = somelist.mean()
print("Salary mean: ",salary_mean)
std_mean_error = stats.sem(somelist)
print("Mean Standard Error: " ,std_mean_error)


# In[5]:


#What is the standard deviation for the years worked?
import numpy 

import pandas as pd
df = pd.read_csv('salary.csv')

somelist = df['yearsworked']

standd = (somelist)

print("Standard deviation: ", numpy.std(standd))


# In[20]:


#What is the median salary for the sample?

import numpy
import pandas as pd

df = pd.read_csv('salary.csv')
df = df.drop(208)

somelist = df['salary']

standd = (somelist)

print("Mean salary $: ", numpy.median(standd))


# In[23]:


#What is the interquartile range for salary in the sample?
import scipy
import numpy as np
from scipy.stats import iqr
import pandas as pd

df = pd.read_csv('salary.csv')
df = df.drop(208)

somelist = (df['salary'])

x = np.array(somelist)

standd = (x)

print("Interquatile range: ", iqr(x))


# In[59]:


#How many men are there in the sample? How many women are there in the sample? Present this information in a table.
import matplotlib.pyplot as plt

df = pd.read_csv('salary.csv')
df = df.drop(208)

# Hide axes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)



salary_df['male'].value_counts()

d = {'Male': [385], 'Female': [128]}
df = pd.DataFrame(data=d)
df





# In[34]:


#How many women are executives compared to men?

salary_df['male'] =  salary_df['male'].map({0: 'female', 1: 'male'})
salary_df.groupby(['position', 'male']).size()


# In[68]:


#Create a histogram for the variable Salary.
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
sns.set(style="whitegrid", color_codes=True)

data = salary_df['salary'].plot(kind='hist')
plt.xlabel('salary')
#palette_col = sns.color_palette("Green_d")
#sns.barplot(x=data.index, y=data, palette=palette_col())
plt.show()


# In[74]:


#Examine the histogram and describe the distribution for Salary.

avg_salary = salary_df.groupby('male').mean()
avg_salary['salary'].plot(kind='bar')
plt.xlabel('Gender')
plt.show()


# In[39]:


#Create a scatterplot showing the relationship between Years Worked and Salary
import seaborn as sns

sns.regplot(salary_df['yearsworked'], salary_df['salary'])


# In[50]:


import scipy
from scipy import stats

stats.pearsonr(df['yearsworked'], df['salary'])


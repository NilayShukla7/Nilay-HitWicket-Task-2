#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Nilay Shukla
import pandas as pd

# Creating the DataFrame
data = {
    'Class': [4, 4, 5, 1, 1, 2, 5, 2],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male'],
    'Values': [10, 3, 1, 5, 7, 2, 5, 10]
}

df = pd.DataFrame(data)
print(df)


# In[2]:


# Nilay Shukla
sorted_df = df.sort_values(by=['Class', 'Values'], ascending=[True, False])
print(sorted_df)


# In[3]:


# Nilay Shukla
grouped_df = df.groupby(['Class', 'Gender']).agg({'Values': 'nunique'}).reset_index()
print(grouped_df)


# In[4]:


#Nilay Shukla
import matplotlib.pyplot as plt

# Creating the scatter plot
plt.figure(figsize=(8, 6))
for gender in df['Gender'].unique():
    subset = df[df['Gender'] == gender]
    plt.scatter(subset['Class'], subset['Values'], label=gender)

plt.xlabel('Class')
plt.ylabel('Values')
plt.title('Scatter Plot of Class vs Values with Gender Differentiation')
plt.legend(title='Gender')
plt.show()


# In[5]:


# Nilay Shukla
pivot_table = df.pivot_table(index='Class', columns='Gender', values='Values', aggfunc='sum').reset_index()
print(pivot_table)


# In[6]:


#Nilay Shukla

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

np.random.seed(42)
table_to_plot = pd.DataFrame({
    'Category': np.random.randint(1, 10, 50),
    'x_variable': np.random.randint(800, 2001, 50),
    'y_variable': np.random.randint(800, 2001, 50)
})

g = sns.FacetGrid(table_to_plot, col="Category", col_wrap=3, height=4)
g.map(plt.scatter, 'x_variable', 'y_variable')
plt.show()


# In[7]:


#Nilay Shukla
labels = ['Low', 'Medium', 'High']

table_to_plot['x_variable_label'] = pd.cut(table_to_plot['x_variable'], bins=3, labels=labels)
table_to_plot['y_variable_label'] = pd.cut(table_to_plot['y_variable'], bins=3, labels=labels)

print(table_to_plot)


# In[ ]:





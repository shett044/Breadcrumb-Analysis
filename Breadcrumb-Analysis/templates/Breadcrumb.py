
# coding: utf-8

# In[212]:

import pandas as pd
import numpy as np


# In[213]:

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')


data = pd.read_csv('Sequence_reports.csv',  parse_dates=[7], date_parser=dateparse, sep='|', header=None, 
                   names=['title','seq1','seq2','seq3','seq4','seq5','seq6','date'])


# <h3> Getting Fact table from the seq columns

# Getting factnames into the seq column

# In[214]:


data.iloc[:,data.columns.str.contains('seq')] = data.iloc[:,data.columns.str.contains('seq')].applymap(lambda x: x.split('%%%')[0] if type(x)==str else x) 


# In[216]:

data.info()


# Creating a sequence with "end" string at the end 

# In[217]:

data["seq"]= data.iloc[:,data.columns.str.contains('seq')].apply(lambda x: '-'.join(list(x.dropna())+['end']), axis =1)


# In[218]:

data.groupby(['title','seq'],as_index=False).agg({'seq1':'count'}).rename(columns={"seq1":"count"}).to_csv('visit_seq.csv',index= False)


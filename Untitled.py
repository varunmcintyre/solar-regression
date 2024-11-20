#!/usr/bin/env python
# coding: utf-8

# U.S. electricity generation by source per month from 1950-2020 (in million Kilowatt hours)
# Cleaned down to just renewable energy, between 2010 and 2020 (most recent)

# https://www.eia.gov/totalenergy/data/browser/index.php?tbl=T07.02A#/?f=M

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

# In[39]:


# Each row title corresponds to a different type of electricity generation source
MSN_DICT = {"CLETPUS":"Coal", "PAETPUS":"Petroleum", "NGETPUS":"Natural Gas", 
          "OJETPUS":"Other Gases", "NUETPUS":"Nuclear", "HPETPUS":"Hydroelectric Pump", 
          "HVETPUS":"Hydroelectric","WDETPUS":"Wood", "WSETPUS":"Waste", 
          "GEETPUS":"Geothermal", "SOETPUS":"Solar", "WYETPUS":"Wind", "ELETPUS":"Total"}

# The sources we care about
args = sys.argv
args.pop(0)
SOURCES = args

# In[40]:


def clean_data(df):
    ''' Function: prepares dataframe for analysis
        Parameters: dataframe
        Returns: dataframe
    '''
    # create new column with just the year
    df["Year"] = df["YYYYMM"] / 100
    df["Year"] = df["Year"].apply(lambda x: int(x))
    
    # create new column with just the month
    df["Month"] = df["YYYYMM"] % 100
    
    # translate MSN to corresponding energy source
    df["Source"] = df["MSN"]
    df["Source"] = df["MSN"].apply(lambda x: MSN_DICT[x])
    
    # take out year totals (Month = 13), and only the last 10 years
    df = df.loc[(df["Month"] != 13)]
    df = df.loc[(df["Year"] >= 2010)]
    
    # remove unnecessary columns
    df = df.drop(["MSN", "YYYYMM", "Unit", "Description", "Column_Order", "Month"], 1)

    # make sure all values in Value are floats
    df["Value"] = df["Value"].apply(lambda x: float(x))
    
    return df


# In[41]:


def split_by_source(df):
    ''' Function: make a list of data frames grouped by electricity source
        Parameters: dataframe
        Returns: a list of dataframes
    '''
    dfs = []
    # for each renewable source
    for source in SOURCES:
        # make a new dataframe with just that source
        grouped_df = df.loc[(df["Source"] == source)]
        # add it to the list of dataframes
        dfs.append(grouped_df)
    return dfs


# In[42]:


def plot_sources(dfs):
    ''' Function: plots a graph for each dataframe in a list
        Parameters: list of dataframes
        Returns: prints graphs
    '''
    for i in range(len(dfs)):
        sns.regplot(x = dfs[i]["Year"], y = dfs[i]["Value"])
        plt.title(SOURCES[i]+" Electricity Generation 2010-2020 (U.S.)")
        plt.ylabel("Million Kilowatt Hours")
        plt.show()


# In[43]:


def lin_reg(dfs, year):
    ''' Function: makes a pie chart based on the linear regression line
                  for each source
        Parameters: list of dataframes, year we want to predict
        Returns: prints graphs
    '''
    data = []
    for i in range(len(dfs)):
        # calculates the linear regression line for each dataframe
        lr = stats.linregress(x = dfs[i]["Year"], y = dfs[i]["Value"])
        # finds the point on the line that corresponds the year we want
        estimate = round((lr.slope * year) + lr.intercept, 3)
        # add to list of estimates, one for each source
        data.append(estimate)
    return data


# In[44]:


def pie_chart(data, year):
    ''' Function: makes a pie chart based on the linear regression prediction
                  for each source
        Parameters: list of predictions (floats), year
        Returns: prints graphs
    '''
    colors = sns.color_palette('bright')[0:5]
    plt.pie(data, labels = SOURCES, colors = colors, autopct='%.0f%%')
    plt.title(str(year)+" U.S. Renewable Electricity Generation Distribution")
    plt.show()


# In[45]:


df = pd.read_csv("electricity_gen_copy.csv")
df = clean_data(df)
dfs = split_by_source(df)
plot_sources(dfs)
'''
data2020 = lin_reg(dfs, 2020)
pie_chart(data2020, 2020)
data2030 = lin_reg(dfs, 2030)
pie_chart(data2030, 2030)
'''


# In[ ]:





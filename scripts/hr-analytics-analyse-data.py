
# coding: utf-8

# In[83]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# #### Import data

# In[84]:

data = pd.read_csv('../data/raw_data.csv')
data.head(10)


# ### Analyze correlations

# In[85]:

sns.set(style="white")

# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# ### Analyze features

# In[86]:

sns.set(style="white")
f, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=data.satisfaction_level,y=data.left,orient="h", ax=ax)


# In[87]:

sns.set(style="darkgrid")
g = sns.FacetGrid(data, row="department", col="left", margin_titles=True)
bins = np.linspace(0, 1, 13)
g.map(plt.hist, "satisfaction_level", color="steelblue", bins=bins, lw=0)


# ### Leavers analysis

# In[119]:

sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9,7))
sns.despine(left=True)

#people that left
leavers = data.loc[data['left'] == 1]

# Plot a simple histogram with binsize determined automatically
sns.distplot(leavers['satisfaction_level'], kde=False, color="b", ax=axes[0,0])
sns.distplot(leavers['salary_level'], bins=3, kde=False, color="b", ax=axes[0, 1])
sns.distplot(leavers['average_monthly_hours'], kde=False, color="b", ax=axes[0, 2])
sns.distplot(leavers['number_projects'], kde=False, color="b", ax=axes[1,0])
sns.distplot(leavers['last_evaluation'], kde=False, color="b", ax=axes[1, 1])
sns.distplot(leavers['time_spent_company'], kde=False, bins=5, color="b", ax=axes[1, 2])
sns.distplot(leavers['promotion_last_5_years'],bins=10, kde=False, color="b", ax=axes[2,0])
sns.distplot(leavers['work_accident'], bins=10,kde=False, color="b", ax=axes[2, 1])


plt.tight_layout()


# ### Count key employees

# In[134]:

#all key employees
key_employees = data.loc[data['last_evaluation'] > 0.7].loc[data['time_spent_company'] >= 3]
key_employees.describe()


# In[135]:

#lost key employees
lost_key_employees = key_employees.loc[data['left']==1]
lost_key_employees.describe()


# In[151]:

print "Number of key employees: ", len(key_employees)
print "Number of lost key employees: ", len(lost_key_employees)
print "Percentage of lost key employees: ", round((float(len(lost_key_employees))/float(len(key_employees))*100),2),"%"


# In[152]:

#save key employees data as csv
key_employees.to_csv('../data/key_employees.csv')


# ### Why do performing emploees leave ?

# In[90]:

#filter out people with a good last evaluation
leaving_performers = leavers.loc[leavers['last_evaluation'] > 0.7]


# In[91]:

sns.set(style="white")

# Compute the correlation matrix
corr = leaving_performers.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# ### Why do satisifed employees leave ?

# In[97]:

#filter out people with a good last evaluation
satisfied_employees = data.loc[data['satisfaction_level'] > 0.7]


# In[99]:

sns.set(style="white")

# Compute the correlation matrix
corr = satisfied_employees.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


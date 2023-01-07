#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Effects of characterestic of fordgobike-tripdata on there duration


# In[19]:


#Investigation Overview

#In this investigation
#I wanted to look at the characteristics of  fordgobike-tripdata on there duration 
#that could be used to predict their duration.
#The main focus was on the four Cs of diamonds: carat (weight), cut grade, color grade, and clarity grade.

#Dataset Overview

#The data consisted There are 183412 fordgobike trips in the dataset with 16 specifications
##Out of 16 specifications 9 are numerical, 
##2 are datetime, 4 are object type and 1 is boolean type
#8460 data points were removed from the analysis due to inconsistencies or missing information.
#to inconsistencies or missing information.


# In[20]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# In[22]:


# load in the dataset into a pandas datafra
df= pd.read_csv('201902-fordgobike-tripdata.csv')


# In[23]:


#Distribution of Trip Durations
# from arount 8000 values at 0 to 12500 values at around 600 
#but then starts to fall and raching below 2000 values under 2000 sec.
#Plotted on a logarithmic scale.
#the distribution of trip duration is skewed to right.


# In[24]:


log_binsize = 0.05
log_bins = 10 ** np.arange(2.4, np.log10(df['duration_sec'].max()) + log_binsize, log_binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df, x = 'duration_sec', bins = log_bins)
plt.xscale('log')
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (sec)')
plt.ylabel('Number of Trips')
plt.xscale('log')
plt.xticks([500, 1e3, 2e3, 5e3, 1e4], [500, '1k', '2k', '5k', '10k'])
plt.axis([0, 10000, 0, 15000])
plt.show()


# In[25]:


#Distribution of age derived from member's birth year.
#A large proportion of age take on birth day member to 
#with gradually decreasing frequencies from age 45.


# In[26]:


# Plotting age distribution derived from member's birth year.
binsize = 1
bins = np.arange(0, df['member_birth_year'].astype(float).max()+binsize, binsize)
plt.figure(figsize=[8, 5])
plt.hist(data = df.dropna(), x = 'member_birth_year', bins = bins)
plt.axis([1939, 2009, 0, 12000])
plt.xticks([1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009], [(2019-1939), (2019-1949), (2019-1959), (2019-1969), (2019-1979), (2019-1989), (2019-1999), (2019-2009)])
plt.gca().invert_xaxis()
plt.title('Distribution of User Age')
plt.xlabel('Age (years)')
plt.ylabel('Number of Users')
plt.show()


# In[27]:


#Duration vs. member_birth_year
#We can see that the distribution is more concentrated between 20 to 40 years old.
#Higher duration is younger members.
#start less frequent users of bikes are aged after 45.


# In[28]:


plt.figure(figsize=[12,5])
plt.subplot(1, 2, 1)
plt.scatter((2019 - df['member_birth_year']), df['duration_sec'], alpha = 0.25, marker = '.' )
plt.axis([-5, 85, 500, 6500])
plt.xlabel('Age (years)')
plt.ylabel('Duaration (sec)')

plt.subplot(1, 2, 2)
bins_y = np.arange(500, 6500+1, 1000)
bins_x = np.arange(-5, 85+1, 10)
plt.hist2d((2019 - df['member_birth_year']), df['duration_sec'],
           bins = [bins_x, bins_y])
plt.colorbar(ticks=[10000, 20000, 30000, 40000]);
plt.show()


# In[29]:


#Duration by gender and age
#Here we observed that though the number of higher duration trip is higher for male 
#but percentage is higher for women and other,
#also other has two peak the first one at the age 30 the scond one at nearly the age of 60 years for higher duration time.
#women and male have one peak.


# In[51]:


df['age'] = (2019 - df['member_birth_year'])
genders = sb.FacetGrid(data = df, col = 'member_gender', col_wrap = 2, size = 5,
                 xlim = [10, 80], ylim = [-500, 9000])
genders.map(plt.scatter, 'age', 'duration_sec', alpha=0.25)
genders.set_xlabels('Age (year)')
genders.set_ylabels('Duration (sec)')

plt.show()


# In[ ]:


#duration by user_type and Age
#Both Customer and Subscriber are showing similer trends for age and trip duration. 
#But the higher age is highest intensity for subscribers.


# In[52]:


user_types = sb.FacetGrid(data = df, col = 'user_type', col_wrap = 2, size = 5,
                 xlim = [10, 80], ylim = [-500, 9000])
user_types.map(plt.scatter, 'age', 'duration_sec', alpha=0.25)
user_types.set_xlabels('Age (year)')
user_types.set_ylabels('Duration (sec)')
plt.show()


# In[ ]:





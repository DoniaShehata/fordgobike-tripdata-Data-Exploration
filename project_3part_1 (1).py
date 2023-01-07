#!/usr/bin/env python
# coding: utf-8

# In[1]:


# fordgobike-tripdata Data Exploration

## Preliminary Wrangling

#This document explores a dataset containing the trip data of the fordgo bike. for approximately 174952 .


# In[2]:


# import all packages and set plots to be embedded inline
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
# load in the dataset into a pandas dataframe, print statistics
df = pd.read_csv('201902-fordgobike-tripdata.csv')
df.head()


# In[3]:


df.info()


# In[4]:


#clean the data from missing value
df.dropna(inplace=True)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# high-level overview of data shape and composition
print(df.shape)
print(df.dtypes)
print(df.head(10))


# In[8]:


# descriptive statistics for numeric variables
print(df.describe())


# In[9]:


#What is the structure of your dataset?
##There are 183412 fordgobike trips in the dataset with 16 specifications
##(duration_sec, start_time, end_time, start_station_id, start_station_name, start_station_latitude, start_station_longitude, end_station_id, end_station_name, end_station_latitude ,end_station_longitude, bike_id, user_type, member_birth_year, member_gender, bike_share_for_all_trip). 
##Out of 16 specifications 9 are numerical, 
##2 are datetime, 4 are object type and 1 is boolean type

#What is/are the main feature(s) of interest in your dataset?
#I'm most interested in figuring out how trip duration is dependent on other specifications from the dataset.

#What features in the dataset do you think will help support your investigation into your feature(s) of interest?
#I expect that trip duration is highly dependent on the start stations and end stations,
#more crowded places should receive more rides so some stations should be logging more duration sec.
#I also think user_type, member_birthyear and member_gender should also effect trip duration.

#Univariate Exploration
#I'll start by looking at the distribution of the main variable of interest: duration_sec.


# In[10]:


binsize = 500
bins = np.arange(0, df['duration_sec'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df, x = 'duration_sec', bins = bins)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (sec)')
plt.ylabel('Number of Trips')
plt.axis([-500, 10000, 0, 90000])
plt.show()


# In[11]:


#There is a long tail in the distribution so lets put it on log scale


# In[12]:


log_binsize = 0.05
log_bins = 10 ** np.arange(2.4, np.log10(df['duration_sec'].max()) + log_binsize, log_binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df, x = 'duration_sec', bins = log_bins)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (sec)')
plt.ylabel('Number of Trips')
plt.xscale('log')
plt.xticks([500, 1e3, 2e3, 5e3, 1e4], [500, '1k', '2k', '5k', '10k'])
plt.axis([0, 10000, 0, 15000])
plt.show()


# In[13]:


#Trip duration is mostly concentrated on the lower spectrum. Most of the values are less than 2000 seconds 
#with peak around 600 seconds. Trip duration values first increases starting 
#from arount 8000 values at 0 to 12500 values at around 600 
#but then starts to fall and raching below 2000 values under 2000 sec.
#Now lets look at other factors like start and end station id and birth year


# In[14]:


# Plotting start station id distribution.
binsize = 1
bins = np.arange(0, df['start_station_id'].astype(float).max()+binsize, binsize)

plt.figure(figsize=[20, 8])
plt.xticks(range(0, 401, 10))
plt.hist(data = df.dropna(), x = 'start_station_id', bins = bins)
plt.title('Distribution of Start Stations')
plt.xlabel('Start Station')
plt.ylabel('Number of Stations')
plt.show()


# In[15]:


# Plotting end station id distribution.
binsize = 1
bins = np.arange(0, df['end_station_id'].astype(float).max()+binsize, binsize)

plt.figure(figsize=[20, 8])
plt.xticks(range(0, 401, 10))
plt.hist(data = df.dropna(), x = 'end_station_id', bins = bins)
plt.title('Distribution of End Stations')
plt.xlabel('End Station')
plt.ylabel('Number of Stations')
plt.show()


# In[16]:


#We can see that same stations are more frequent as start stations and end stations


# In[17]:


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


# In[18]:


#We can see that the distribution is more concentrated between 20 to 40 years old.


# In[19]:


# plotting types of users on bar.
plt.figure(figsize=[8,5])
plt.bar(x = df.user_type.value_counts().keys(), height = df.user_type.value_counts() )
plt.xlabel('User Type')
plt.ylabel('Number of Users')
plt.show()


# In[20]:


# plotting genders on bar.
plt.figure(figsize=[8,5])
plt.bar(x = df.member_gender.value_counts().keys(), height = df.member_gender.value_counts() )
plt.xlabel('Gender')
plt.ylabel('Number of Users')
plt.show()


# In[21]:


#Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? 
#Did you need to perform any transformations?
#The trip duration takes a large amount of values and is concentrated to a tail so 
#I looked at it in log transform and found that peak occurs at 600 seconds starting from 0 and 
#then distribution starts to dip and does not regain any more peak value.

#Of the features you investigated, were there any unusual distributions? 
#Did you perform any operations on the data to tidy, adjust, or change the form of the data? 
#If so, why did you do this?

#Birth year is converted by substracting the year 
#from cur rent year so this gives us a distibution 
#for age, this action is performed as age gives a better perception regarding trip duration dependency.
#Also start station and end station is plotted in a larger plot
#because it is better show about regarding traffic of bikes at certain stations.

#Bivariate Exploration
#Let's first have a look at the correlation between trip duration and age.


# In[49]:


plt.figure(figsize=[8,5])
plt.scatter((2019 - df['member_birth_year']), df['duration_sec'], alpha = 0.25, marker = '.' )
plt.axis([-5, 145, 500, 10500])
plt.xlabel('Age (years)')
plt.ylabel('Duaration (sec)')
plt.show()


# In[23]:


#As most of the durations are below 6000 and age is below 80, lets crop the plot till those values.


# In[24]:


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


# In[25]:


#By looking at these olys we can say that most frequent users of bikes are aged between 20 and 45.
#By looking at these oly we can say that  start less frequent users of bikes are aged after 45.
#Higher duration is  younger members.
#Now lets look into the duration dependency on start station and end station.


# In[26]:


sorted(df.start_station_id.unique())


# In[27]:


#Now lets look into the duration dependency on start station and end station.
t = []
all_start_station_ids = sorted(df.start_station_id.unique())
for x in all_start_station_ids :
    t.append(df[df.start_station_id == x].duration_sec.sum()) 
total_duration = pd.Series(t)
plt.figure(figsize = [20, 8])
sb.lineplot(x = df['start_station_id'], y = total_duration)
plt.xticks(range(0, 401, 10))
plt.xlabel('Start Station')
plt.ylabel('Total Duration')
plt.show()


# In[28]:


t = []

all_end_station_ids = sorted(df.end_station_id.unique())
for x in all_end_station_ids :
    t.append(df[df.end_station_id == x].duration_sec.sum()) 
total_duration = pd.Series(t)


# In[29]:


plt.figure(figsize = [20, 8])
sb.lineplot(x = df['start_station_id'], y = total_duration)
plt.xticks(range(0, 401, 10))
plt.xlabel('End Station')
plt.ylabel('Total Duration')
plt.show()


# In[30]:


#By looking at these plots you can see that trip duration for some station as start station is higher 
#and for some stations as end station is higher.
#By this we can see that what stations result in starting of longer trips and what stations comes end of longer trips.
#Now lets look into the dependency of trip durations on gender and on member type.


# In[31]:


plt.figure(figsize = [8, 5])
base_color = sb.color_palette()[1]
sb.boxplot(data = df, x = 'member_gender', y = 'duration_sec', color = base_color)
plt.xlabel('Gender')
plt.ylabel('Duration (sec)')
plt.show()


# In[32]:


#As we can see, values are vey widespread to see a box plot, so lets trim 
#duration to max 1500 sec to get clearer picture.


# In[56]:



plt.figure(figsize = [8, 5])
base_color = sb.color_palette()[1]
sb.boxplot(data = df, x = 'member_gender', y = 'duration_sec', color = base_color)
plt.ylim([-10, 2500])
plt.xlabel('Gender')
plt.ylabel('Duration (sec)')
plt.show()


# In[57]:


plt.figure(figsize = [8, 5])
base_color = sb.color_palette()[0]
sb.violinplot(data = df, x = 'member_gender', y = 'duration_sec',
                  color = base_color)
plt.xticks(rotation = 15)
plt.ylim([-10, 2500])
plt.xlabel('Gender')
plt.ylabel('Duration (sec)')
plt.show()


# In[34]:


#Though quantity of male riders are very high then other and female but we can see that higher percentage of female 
#and other rides longer trips then males.


# In[35]:


plt.figure(figsize = [8, 5])
base_color = sb.color_palette()[1]
sb.boxplot(data = df, x = 'user_type', y = 'duration_sec', color = base_color)
plt.xlabel('User Type')
plt.ylabel('Duration (sec)')
plt.show()


# In[36]:


#As we can see, values are vey widespread to see a box plot
#so lets trim duration to max 3000 sec get clearer picture


# In[59]:


plt.figure(figsize = [8, 5])
base_color = sb.color_palette()[1]
sb.boxplot(data = df, x = 'user_type', y = 'duration_sec', color = base_color)
plt.ylim([-10, 3000])
plt.xlabel('User Type')
plt.ylabel('Duration (sec)')
plt.show()


# In[38]:


#Here we can see that higher percentage of customers are taking longer trips then compared to subscribers.


# In[39]:


#Talk about some of the relationships you observed in this part of the investigation.
#How did the feature(s) of interest vary with other features in the dataset?
#Trip Duration is very dependendable on the age of the member, I havent expected the=at much dependency.
#On the other hand Start station and end station does not much determine the trip duration.
#It only suggests that some starting stations are having higher trip durations as starting point 
#and some end stations are having higher trip durations as ending point.

#Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?
#I have expected that categorial variable's like user type and gender values having higher value
#to be having higher trip duration but it is the other way round like for gender,
#value of male members is very high but the percentage of female members to take longer trips is higher.

#Multivariate Exploration
#The main thing I want to explore in this part of the analysis is how the two  categorical measures gender 
#and user type play into the relationship between trip duration and age.


# In[40]:


gender_markers = [['Male', 's'],['Female', 'o'],['Other', 'v']]

for gender, marker in gender_markers:
    df_gender = df[df['member_gender'] == gender]
    plt.scatter((2019 - df_gender['member_birth_year']), df_gender['duration_sec'], marker = marker, alpha=0.25)
plt.legend(['Male','Female','Other'])
plt.axis([10, 80, -500, 9000 ])
plt.xlabel('Age (year)')
plt.ylabel('Duration (sec)')
plt.show()


# In[41]:


#This plot does not show quit a clear picture, lets seperate all three genders into different graphs.


# In[42]:


df['age'] = (2019 - df['member_birth_year'])
genders = sb.FacetGrid(data = df, col = 'member_gender', col_wrap = 2, size = 5,
                 xlim = [10, 80], ylim = [-500, 9000])
genders.map(plt.scatter, 'age', 'duration_sec', alpha=0.25)
genders.set_xlabels('Age (year)')
genders.set_ylabels('Duration (sec)')

plt.show()


# In[43]:


#Here we are seeing a jump in duration for others at an older age (around 60 years)


# In[44]:


user_type_markers = [['Customer', 's'],['Subscriber', 'o']]

for utype, marker in user_type_markers:
    df_utype = df[df['user_type'] == utype]
    plt.scatter((2019 - df_utype['member_birth_year']), df_utype['duration_sec'], marker = marker, alpha=0.25)
plt.legend(['Customer','Subscriber'])
plt.axis([10, 80, -500, 9000 ])
plt.xlabel('Age (year)')
plt.ylabel('Duration (sec)')
plt.show()


# In[45]:


user_types = sb.FacetGrid(data = df, col = 'user_type', col_wrap = 2, size = 5,
                 xlim = [10, 80], ylim = [-500, 9000])
user_types.map(plt.scatter, 'age', 'duration_sec', alpha=0.25)
user_types.set_xlabels('Age (year)')
user_types.set_ylabels('Duration (sec)')

plt.show()


# In[46]:


#In this case both Customer and Subscriber are showing similer trends for age and trip duration. 
#But there is slight tilt to higher age for subscribers.


# In[47]:


##Talk about some of the relationships you observed in this part of the investigation.
#Were there features that strengthened each other in terms of looking at your feature(s) of interest?

#Here we observed that though the number of higher duration trip is higher for male 
#but percentage is higher for women and other,
#also other has one more peak at nearly the age of 60 years for higher duration time.
#For different user types both are showing similer trends for age and trip duration.
#But there is slight tilt to higher age for subscribers having better trip duration.

#Were there any interesting or surprising interactions between features?
#A second peak for other gender at an older is a surprise.


# In[ ]:





# In[ ]:





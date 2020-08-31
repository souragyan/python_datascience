# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:50:17 2020

@author: neeraj.shukla
"""

#install and then import matplotlib
import matplotlib.pyplot as plt

#matplotlib to render plots in the notebook
%matplotlib inline

x = [-3, 5, 7]
y = [10, 2, 5]

fig = plt.figure(figsize=(15,3))

plt.plot(x, y)
plt.xlim(-4, 10)
plt.ylim(0, 12)
plt.xlabel('X Axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Sales Comaprison', size=20, y=1.03)

#################
fig.get_size_inches()
fig.set_size_inches(14, 4)
fig, axs = plt.subplots(nrows=1, ncols=1)
fig


# More than one Axes with plt.subplots, then the second item in the tuple is a NumPy array containing all the Axes
fig, axs = plt.subplots(2, 4)

#### Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



mtcars = pd.read_csv("mtcars.csv")
mtcars.columns
mtcars.head()

# table 
pd.crosstab(mtcars.gear,mtcars.cyl)

# bar plot between 2 different categories 
pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar")
mtcars["gear"].value_counts()

mtcars["gear"].value_counts()
mtcars.gear.value_counts().plot(kind="pie")

import seaborn as sns 
# getting boxplot of mpg with respect to each category of gears 
sns.boxplot(x="gear",y="mpg",data=mtcars)

sns.pairplot(mtcars.iloc[:,0:4]) # histogram of each column and 
# scatter plot of each variable with respect to other columns 

plt.scatter(mtcars.mpg,mtcars.qsec)## scatter plot of two variables


# Graphical Representation of data
#import matplotlib.pyplot as plt
# Histogram
plt.hist(mtcars['mpg']) 

plt.hist(mtcars['mpg'],facecolor ="peru",edgecolor ="blue",bins =5)
#creates histogram with 5bins and colours filled init.

#Boxplot
#help(plt.boxplot)
plt.boxplot(mtcars['mpg'],vert = True)

plt.boxplot(mtcars['mpg'],vert =False);plt.ylabel("MPG");plt.xlabel("Boxplot");plt.title("Boxplot")  # for vertical

#Violin Plot
#help(plt.violinplot)
plt.violinplot(mtcars["mpg"])

#### Visualizing data with pandas
# seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tips =sns.load_dataset('tips')
tips

sns.set_style("darkgrid", {'axes.grid' : True})
sns.lmplot(x= 'total_bill', y='tip', data=tips)
plt.show()



sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex',palette='Set2')
plt.show()sns.lmplot(x='total_bill', y='tip', data=tips, col='sex')
plt.show()

## Univariate → “one variable” data visualization

#strip plot

sns.stripplot(y= 'tip', data=tips,jitter=False)
plt.ylabel('tip ($)')
plt.show()

#Grouping with stripplot()
sns.stripplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

sns.stripplot(x='day', y='tip', data=tips, size=4,jitter=False)
plt.ylabel('tip ($)')
plt.show()

#Swarm plot

sns.swarmplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()
    
sns.swarmplot(x='day', y='tip', data=tips, hue='sex')
plt.ylabel('tip ($)')
plt.show()

#Changing orientation
sns.swarmplot(x='tip', y='day', data=tips, hue='sex',orient='h')
plt.xlabel('tip ($)')
plt.show()

### Box and Violin plot

#plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

#plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

plt.show()

plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

plt.show()

##Combining plots

sns.violinplot(x='day', y='tip', data=tips, inner=None,color='lightgray')
sns.stripplot(x='day', y='tip', data=tips, size=4,jitter=True)
plt.ylabel('tip ($)')
plt.show()

## Bivariate → “two variables” data visualization

#Joint plot
sns.jointplot(x= 'total_bill', y= 'tip', data=tips)
plt.show()

# Density plot
sns.jointplot(x='total_bill', y= 'tip', data=tips,kind='kde')
plt.show()

#Pair plot

sns.pairplot(tips)
plt.show()

sns.pairplot(tips, hue='sex')
plt.show()

tips.corr()
#Covariance heat map of tips data
sns.heatmap(tips.corr(),vmin=-1, vmax=1, cmap='ocean')

# Titanic data visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

### Loading dataset

data = pd.read_csv("train.csv")
data.head()
### Proportion of target (Survived

data.groupby('Survived')['PassengerId'].count()
#This dataset has a decent proportion of target class and it is not skewed to any one.
### Visual Exploration

f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x='Survived', y="Age",  data=data);
f, ax = plt.subplots(figsize=(7,7))
sns.barplot(x='Sex', y="Survived",  data=data);
f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x="Sex", y="Age", hue="Survived", data=data);

sns.barplot(x="Pclass", y="Survived", data=data);
sns.barplot(x="Pclass", y="Survived",hue="Sex", data=data);

sns.barplot(x="SibSp", y="Survived", data=data);
sns.barplot(x="Parch", y="Survived", data=data);

survived = data.loc[data['Survived']==1,"Age"].dropna()
sns.distplot(survived)
plt.title("Survived");

not_survived = data.loc[data['Survived']==0,"Age"].dropna()
sns.distplot(not_survived)
plt.title("Not Survived");

#Infants had high survival rate and elderly passengers above 65+ were less likely to survive


sns.pairplot(data.dropna());

# Pclass vs Survive
grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.4, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5,ci=None)
grid.add_legend()

grid = sns.FacetGrid(data, row='Pclass', col='Sex', )
grid.map(plt.hist, 'Age', alpha=.5, bins=40)
grid.add_legend()

sns.distplot(a=data[data['Embarked']=='C']['Survived'],bins=3,kde=False)
plt.title("Cherbourg")
plt.xticks([0,1])
plt.show()
plt.title("QueensTown")
sns.distplot(a=data[data['Embarked']=='Q']['Survived'],bins=3,kde=False)
plt.xticks([0,1])

plt.show()
plt.title("Southampton")
sns.distplot(a=data[data['Embarked']=='S']['Survived'],bins=3,kde=False)
plt.xticks([0,1])

plt.show()
#Most of the Passengers embarked from Southampton


#Plotting correlation by using heatmap
sns.heatmap(data.corr(),cmap='CMRmap')
plt.legend()


figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
data.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
data.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
data.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
data.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
data.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=data,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=data,ax=axesbi[1,2])








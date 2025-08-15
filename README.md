<H3>ENTER YOUR NAME : Vikamuhan reddy</H3>
<H3>ENTER YOUR REGISTER NO. : 212223240181</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 15/08/25</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Read the dataset from drive
df = pd.read_csv("/Users/apple/neuralNetworkExps/EX-1-NN/Churn_Modelling.csv")
df = df.drop(["Surname","RowNumber","CustomerId"],axis=1)
df.head()

# Finding Missing Values
print(df.isnull().sum())

#Check for Duplicates
df.duplicated().sum()

# converting string to numerical values(Label Encoding)
encode = LabelEncoder()
df['Gender'] = encode.fit_transform(df['Gender'])
df['Geography'] = encode.fit_transform(df['Geography'])

# Spliting the data
X = df.iloc[: , :-1]
y = df.iloc[: , -1]

# Normalize the data
scaler = StandardScaler()
X_transform = scaler.fit_transform(X)
X_transform

#split the dataset into input and output
x_train,x_test,y_train,y_test = train_test_split(X_transform,y,test_size=0.2)

#Print the training data and testing data
print("The training set of transformed X :\n",x_train)
print("The testing set of transformed X :\n ",x_test)
print("The training set of y : \n",y_train)
print("The testing set of y : \n",y_test)
```


## OUTPUT:
!['Output](./Screen%20Shot%201947-05-24%20at%2012.45.10.png)
!['Output'](./Screen%20Shot%201947-05-24%20at%2012.44.58.png)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
## 1.Importing the libraries
## 2.Importing the dataset
## 3.Taking care of missing data
## 4.Encoding categorical data
## 5.Normalizing the data
## 6.Splitting the data into test and train

## PROGRAM:
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

df.head()
df.info()

x = df.iloc[:,:-1].values
y= df.iloc[:,1].values
x
y

df.describe()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))



df1.describe()


X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))
## OUTPUT:
## HEAD:
![263438828-29bc9335-b460-48f0-a2f2-47c95f82ce63](https://github.com/SYEDADILBASHA1/Ex.No.1---Data-Preprocessing/assets/134796157/f9068916-d55d-4e2c-8e18-d7e1179acff7)
## INFO:
![263439148-bd4e699f-9aef-472f-8d9a-69b7f28f0f2d](https://github.com/SYEDADILBASHA1/Ex.No.1---Data-Preprocessing/assets/134796157/f6627736-75c7-4737-9c0b-ec97e11582ff)
## DESCRIBE BEFORE NORMALIZING:
![263439313-465a0119-2567-4667-bd3b-9ed8b2b5393c](https://github.com/SYEDADILBASHA1/Ex.No.1---Data-Preprocessing/assets/134796157/780e0375-0147-4b53-a8a3-37a899358489)
## DESCRIBE AFTER NORMALIZING:
![263439526-8e887571-f499-4a37-9500-996d0dac577c](https://github.com/SYEDADILBASHA1/Ex.No.1---Data-Preprocessing/assets/134796157/7edd95b6-e836-4813-9bc8-cbab023117a8)
## SIZE OF XTRAIN:
![263440105-5fefe79a-7ad5-4342-ae2e-fe84820cd7d7](https://github.com/SYEDADILBASHA1/Ex.No.1---Data-Preprocessing/assets/134796157/eced63ed-6929-43ff-8732-7a601fc9012d)
## SIZE OF XTEST
![263440143-8847fc37-c5de-4120-bb51-af0f1cdce126](https://github.com/SYEDADILBASHA1/Ex.No.1---Data-Preprocessing/assets/134796157/5d20baff-5f5a-4a65-9764-6b36e0517fe9)

## RESULT
Hence The  Data preprocessing is performed in the given dataset.

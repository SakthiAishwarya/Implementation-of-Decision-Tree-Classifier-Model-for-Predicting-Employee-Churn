# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy score of the model.
6. Check the trained model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SAKTHI AISHWARYA.S
RegisterNumber:  212219040132
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project",
"average_montly_hours","time_spend_company","Work_accident",
"promotion_last_5years","salary"]]
x.head()
y=df["left"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
acc
dt.predict([[.5,.8,9,260,6,0,1,2]])
```

## Output:
### Initial Dataset:
![image](https://user-images.githubusercontent.com/67967960/175043650-0cec10d3-1248-4c1d-8e73-1abb39fcc26a.png)
### Dataset Information:
![image](https://user-images.githubusercontent.com/67967960/175044119-b59d56cc-d3d0-487d-a13b-6fa8e5f0b3c0.png)
### Null dataset:
![image](https://user-images.githubusercontent.com/67967960/175044180-d8c259e6-723a-45e4-887c-47e404d4c2dd.png)
### Value counts in left column:
![image](https://user-images.githubusercontent.com/67967960/175044250-b7ae7dfc-3a7f-4064-81a7-9a536d4167da.png)
### Encoded dataset:
![image](https://user-images.githubusercontent.com/67967960/175044314-743005ce-9220-4763-b2e2-030e96aa2030.png)
### x set:
![image](https://user-images.githubusercontent.com/67967960/175044391-58b9c239-6fb0-4991-a8a0-a91d9060c984.png)
### y values:
![image](https://user-images.githubusercontent.com/67967960/175044503-a3063353-1e3c-4039-9f8f-e3aa97fb5ff9.png)
### Accuracy Score:
![image](https://user-images.githubusercontent.com/67967960/175044573-2f108bd9-3291-42e0-8340-8dc0874d748e.png)
### Dataset Prediction:
![image](https://user-images.githubusercontent.com/67967960/175044643-e46a834f-2b1f-47e2-b357-6f30a6b40b57.png)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

# Implementation-of-SVM-For-Spam-Mail-Detection
<br><br>
## AIM:
To write a program to implement the SVM For Spam Mail Detection.
<br><br>
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
<br><br>
## Algorithm
step1: Start.

step2: Import the necessary python packages using import statements.

step3: Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

step4: Split the dataset using train_test_split.

step5: Calculate Y_Pred and accuracy.

step6: Print all the outputs.

step7: End.
<br><br>
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
x.shape
y.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
con
```
<br><br>
## Output:
<br><br>
data.head():

![image](https://github.com/user-attachments/assets/f8bf8910-4e73-4a1b-899f-f6c2bbe31d92)
<br><br>
data.info():

![image](https://github.com/user-attachments/assets/05af1adf-e0cb-497e-964c-064021302199)
<br><br>
y_pred:

![image](https://github.com/user-attachments/assets/76560b35-e221-420c-a18e-709f75989bfe)
<br><br>
Accuracy:

![image](https://github.com/user-attachments/assets/201cd1e0-087e-4507-ab1f-66151ea54b32)
<br><br>
Confusion_Matrix:

![image](https://github.com/user-attachments/assets/75363291-6913-4621-8836-9f787f57913b)

<br><br>

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

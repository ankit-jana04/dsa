#Step 1: Import Packages and Display the dataset and perform shape, head, value counts Function
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer(as_frame=True)
dataset


dataset['data'].head()

dataset['data'].shape

dataset['target'].head()

dataset['target'].value_counts()

#Define explanatory and target variables:
#Step 2:Split the dataset into training and testing sets

x = dataset['data']
y = dataset['target']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
ss_train = StandardScaler()
x_train = ss_train.fit_transform(x_train)
ss_test = StandardScaler()
x_test = ss_train.transform(x_test)


#Step 3: Normalize the data for numerical stability
from sklearn.preprocessing import StandardScaler
ss_train = StandardScaler()
x_train = ss_train.fit_transform(x_train)
ss_test = StandardScaler()
x_test = ss_train.transform(x_test)

#Step 4: Fit a logistic regression model to the training data
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train, y_train)


#Step 5: Make predictions on the testing data
y_pred = logistic_classifier.predict(x_test)
print (y_pred[0:5])
print (y_test [0:5])


#Step 6: Calculate the accuracy score by comparing the actual values and predicted values
#We will calculate the confusion matrix to get the necessary parameters:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print ('True Positive (TP) ', TP)
print ('True Negative (TN)',TN)
print ('False Positive(FP) ',FP)
print ('False Negative(FN) ',FN)


from sklearn import metrics
import matplotlib.pyplot as plt
print("Logistic Regression's Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# another way to fing accuracy
accuracy=(TP + TN) / (TP + TN + FP + FN)
print("Logistic Regression's Accuracy: {:0.3f}".format(accuracy))



#Initializing each binary classifier To quickly train each model in loop, we’ll initialize each mode
#and store it by name in a dictionary:
models={}
#Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression']=LogisticRegression()
# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] =LinearSVC()
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

models['Decision Tree']=DecisionTreeClassifier()
#Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest']=RandomForestClassifier()
# Navie Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes']= GaussianNB()
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()
models


#Performance evaluation of each binary classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy, precision, recall={},{},{}
for key in models.keys():
#fit the classifier
    models[key].fit(x_train, y_train)
#Make prediction
    predictions=models [key].predict(x_test)
#Calculate Accuracy
    accuracy[key] = accuracy_score(predictions, y_test)
    precision [key] = precision_score(predictions, y_test)
    recall[key]=recall_score(predictions, y_test)


#With all metrics stored, we can use pandas to view the data as a table:
import pandas as pd
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy']=accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall']=recall.values()
df_model


#Visualization:
ax = df_model.plot.barh()
ax.legend(
ncol=len(models.keys()),
bbox_to_anchor=(0, 1),
loc='lower left',
prop={'size': 14}
)
plt.tight_layout()


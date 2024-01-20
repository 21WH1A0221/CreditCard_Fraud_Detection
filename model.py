# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv("C:\\Users\\pooji\\ml project\\fraudTrain.csv")
#train_data = pd.read_csv("fraudTrain.csv")
train_data = train_data.iloc[:, 1:]
print(train_data.head())
print(train_data.info())
print(train_data.describe())
# Exclude non-numeric columns from the correlation matrix
numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = train_data[numeric_columns].corr()
# Print the correlation matrix
print(correlation_matrix)
#print(train_data.corr())
print(train_data.shape)
# Determine number of fraud cases in dataset
fraud = train_data[train_data['is_fraud'] == 1]
valid = train_data[train_data['is_fraud'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(train_data[train_data['is_fraud'] == 1])))
print('Valid Transactions: {}'.format(len(train_data[train_data['is_fraud'] == 0])))
print('Amount details of the fraudulent transaction')
fraud['amt'].describe()
# dividing the X and the Y from the dataset
X = train_data.drop(['is_fraud'], axis = 1)
Y = train_data["is_fraud"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing
# (its a numpy array with no columns)
x = X.values
y = Y.values
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# convert the 'Date' column to datetime format
fraud['trans_date_trans_time']= pd.to_datetime(fraud['trans_date_trans_time'])
train_data["trans_date_trans_time"] = pd.to_datetime(train_data["trans_date_trans_time"], infer_datetime_format=True)
train_data["dob"] = pd.to_datetime(train_data["dob"], infer_datetime_format=True)
print(train_data.isnull().sum())
if train_data.duplicated().sum() > 0:
    train_data.drop_duplicates(inplace=True)
    print('Duplicates dropped')
else:
    print('No Duplicates Exist')
print(train_data['gender'].value_counts())
train_data['gender'] = pd.Categorical(train_data['gender']).codes
print(train_data)
from datetime import date

def calculate_age(row):
    today = date.today()
    return today.year - row['dob'].year - ((today.month, today.day) < (row['dob'].month, row['dob'].day))
# Train data
train_data['dob'] = pd.to_datetime(train_data['dob'])
train_data['age'] = train_data.apply(lambda row: calculate_age(row), axis=1)
print('Age of train dataset', train_data['age'].head(3))
train_data['age'] = train_data['age'].astype(float)
train_data['city_pop'] = train_data['city_pop'].astype(float)
train_data['gender'] = train_data['gender'].astype(float)
print(train_data)
X = train_data.drop(['is_fraud','cc_num', 'street', 'city', 'state', 'zip', 'lat', 'long','merch_lat','merch_long','trans_num','unix_time','dob','trans_date_trans_time','category','first','last','merchant','job'],axis = 1)
y = train_data['is_fraud']
from sklearn.model_selection import StratifiedShuffleSplit

stratified_split = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

# Apply SMOTE to oversample the minority class
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
print("Data Type of X_train:", X_train.dtypes)
print("Data Type of y_train:", y_train.dtypes)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Train a logistic regression model
lg_model = LogisticRegression(random_state=42)
lg_model.fit(X_train_oversampled, y_train_oversampled)
# Predict probabilities
lg_yproba = lg_model.predict_proba(X_test)[:, 1]
# Set a lower threshold to reduce accuracy
threshold = 0.45  # You can adjust this threshold to achieve your desired accuracy level
# Adjust predictions based on the threshold
lg_ypred = (lg_yproba > threshold).astype(int)
# Evaluate performance
print("Accuracy score is: ", round(accuracy_score(y_test, lg_ypred) * 100, 2), '%')
print("Classification Report:\n", classification_report(y_test, lg_ypred))
print("Confusion Matrix:\n", confusion_matrix(y_test,lg_ypred))
import pickle
#pickle for our model
pickle.dump(lg_model,open("model.pkl","wb"))
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Enable mlflow autologging
mlflow.autolog()

# Load the data
data = pd.read_csv('Titanic.csv')

# Drop unnecessary columns
data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Fill missing values with mean of the corresponding column
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Create the features and target arrays
features = data.drop('Survived', axis=1)
target = data['Survived']

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(features_train, target_train)

# Make predictions on the test set
predictions = rf.predict(features_test)

# Evaluating the model performance
accuracy = accuracy_score(target_test, predictions)
precision = precision_score(target_test, predictions)
recall = recall_score(target_test, predictions)
f1 = f1_score(target_test, predictions)

# Logging the parameters
mlflow.log_params({'n_estimators': 100})
mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1})

# Start the mlflow run
with mlflow.start_run():
    pass

# Updating the parameters of the model
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the updated model
rf.fit(features_train, target_train)

# Start the mlflow run for the second version of the model
with mlflow.start_run():
    # Update the parameters of the model
    mlflow.log_params({'n_estimators': 200})
    
    # Making predictions on the test set and evaluate the model performance
    predictions = rf.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
    precision = precision_score(target_test, predictions)
    recall = recall_score(target_test, predictions)
    f1 = f1_score(target_test, predictions)
    
    # Log the updated metrics
    mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1})
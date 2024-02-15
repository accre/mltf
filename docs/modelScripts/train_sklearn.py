#!/usr/bin/env python3

#Simple scikit-learn random forrest regressor, with MLflow autolog capabilities. 
#This will track and save the model, paramters, metrics, and data on the MLflow server 

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import mlflow
mlflow.autolog()

# Load dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

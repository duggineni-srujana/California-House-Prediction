#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Convert target to binary for logistic regression (above/below median price)
y_median = np.median(y)
y_binary = (y > y_median).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_bin_scaled = scaler.transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_bin_scaled, y_train_bin)

# Save the trained models
with open("model_linear.pkl", "wb") as file:
    pickle.dump((linear_model, scaler), file)

with open("model_logistic.pkl", "wb") as file:
    pickle.dump((logistic_model, scaler), file)


# In[ ]:





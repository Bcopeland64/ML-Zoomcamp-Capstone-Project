import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle

# Load your data
data = pd.read_csv('/home/brandon/ML Zoomcamp/Capstone Project 1/salaries.csv')

# Clean the headings
data.columns = data.columns.str.lower().str.replace(' ', '_')

# Assume you have performed your preprocessing steps here

# Define numerical and categorical features
numerical = ['salary', 'remote_ratio']
categorical = ['work_year', 'experience_level', 'employment_type', 'job_title',
               'salary_currency', 'employee_residence', 'company_location', 'company_size']

# Split the data
df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Assume you have performed your feature engineering and preprocessing here

# Separate target variable
y_train = df_train.salary_in_usd.values
y_val = df_val.salary_in_usd.values

# Create DictVectorizer for feature transformation
dv = DictVectorizer(sparse=False)

# Transform training data
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
x_train = dv.fit_transform(train_dicts)

# Train a Decision Tree model
dt = DecisionTreeRegressor(max_depth=2)
dt.fit(x_train, y_train)

# Save the DictVectorizer, model, and feature names
model_info = {'model': dt, 'num_features': x_train.shape[1], 'dv': dv, 'feature_names': dv.get_feature_names_out()}

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model_info, model_file)

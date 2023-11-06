from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
import pickle
import pandas as pd
import numpy as np

app = Flask('predict')

# Load the trained model along with the DictVectorizer and feature names
with open('model.pkl', 'rb') as model_file:
    model_info = pickle.load(model_file)

model = model_info['model']
num_features = model_info['num_features']
dv = model_info['dv']
feature_names = model_info['feature_names']

# Define categorical and numerical features
categorical = ['work_year', 'experience_level', 'employment_type', 'job_title',
               'salary_currency', 'employee_residence', 'company_location', 'company_size']
numerical = ['remote_ratio']

def preprocess_input(data):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame(data, index=[0])

    # Clean column headings
    input_df.columns = input_df.columns.str.lower().str.replace(' ', '_')

    # Handle missing values if any
    input_df.fillna(0, inplace=True)  # Replace NaN values with 0, you may want to use a different strategy

    # Encode categorical variables using DictVectorizer
    input_dicts = input_df[categorical + numerical].to_dict(orient='records')
    input_data = dv.transform(input_dicts)

    print("Input data shape:", input_data.shape)
    print("Input data content:", input_data)

    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Preprocess the input data
        input_data = preprocess_input(data)

        # Print the number of features expected by DecisionTreeRegressor
        print("Number of features expected by DecisionTreeRegressor:", num_features)

        # Make a prediction
        prediction = model.predict(input_data)

        # Prepare the response
        response = {
            'prediction': prediction.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

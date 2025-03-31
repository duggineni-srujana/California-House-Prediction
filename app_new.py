#!/usr/bin/env python
# coding: utf-8

# In[6]:


from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained models
with open("model_linear.pkl", "rb") as file:
    linear_model, scaler = pickle.load(file)

with open("model_logistic.pkl", "rb") as file:
    logistic_model, _ = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    classification = None
    error = None
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    
    if request.method == 'POST':
        try:
            features = [float(request.form[feature]) for feature in feature_names]
            features_scaled = scaler.transform([np.array(features)])
            prediction = linear_model.predict(features_scaled)[0]
            prediction = f"${prediction * 100000:.2f}"  # Convert to USD
            classification = logistic_model.predict(features_scaled)[0]
            classification = "Above Median Price" if classification == 1 else "Below Median Price"
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template('app_index.html', prediction=prediction, classification=classification, error=error, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('/Users/rahulkumarair/Documents/rahul_vsCode/machine_learning/heart_attack/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Return the result
    if prediction[0] == 1:
        result = "You have a high risk of heart attack."
    else:
        result = "You have a low risk of heart attack."
    
    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

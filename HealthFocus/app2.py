from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='templates')

# Load the datasets
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv", encoding='latin1')
workout = pd.read_csv("workout_df.csv", encoding='latin1')
description = pd.read_csv("description.csv", encoding='latin1')
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Load the trained model
with open('svc.pkl', 'rb') as f:
    svc = pickle.load(f)

# Symptoms dictionary and diseases list
symptoms_dict = {
    'Itching': 0, 'Skin rash': 1, 'Nodal skin eruptions': 2, 'Continuous sneezing': 3, 
    # ... (include all your symptoms from your notebook)
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    # ... (include all your diseases from your notebook)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data['symptoms']
        
        # Create input vector for prediction
        input_vector = np.zeros(len(symptoms_dict))
        for symptom in symptoms:
            if symptom in symptoms_dict:
                input_vector[symptoms_dict[symptom]] = 1
        
        # Make prediction
        prediction = svc.predict([input_vector])[0]
        predicted_disease = diseases_list[prediction]
        
        # Get recommendations
        desc = description[description['Disease'] == predicted_disease]['Description'].values[0]
        
        prec = precautions[precautions['Disease'] == predicted_disease][
            ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        ].values[0].tolist()
        
        meds = medications[medications['Disease'] == predicted_disease]['Medication'].values[0].split(', ')
        
        diet = diets[diets['Disease'] == predicted_disease]['Diet'].values[0].split(', ')
        
        wrkout = workout[workout['disease'] == predicted_disease]['workout'].values[0].split(', ')
        
        # Prepare response
        response = {
            'predicted_disease': predicted_disease,
            'description': desc,
            'precautions': prec,
            'medications': meds,
            'diet': diet,
            'workout': wrkout
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
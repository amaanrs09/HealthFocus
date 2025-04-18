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

# Symptoms dictionary and diseases list (from your notebook)
symptoms_dict = {
    'Itching': 0, 'Skin rash': 1, 'Nodal skin eruptions': 2, 'Continuous sneezing': 3, 
    'Shivering': 4, 'Chills': 5, 'Joint pain': 6, 'Stomach pain': 7, 'Acidity': 8, 
    'Ulcers on tongue': 9, 'Muscle wasting': 10, 'Vomiting': 11, 'Burning micturition': 12, 
    'Spotting urination': 13, 'Fatigue': 14, 'Weight gain': 15, 'Anxiety': 16, 
    'Cold hands and feets': 17, 'Mood swings': 18, 'Weight loss': 19, 'Restlessness': 20, 
    'Lethargy': 21, 'Patches in throat': 22, 'Irregular sugar level': 23, 'Cough': 24, 
    'High fever': 25, 'Sunken eyes': 26, 'Breathlessness': 27, 'Sweating': 28, 
    'Dehydration': 29, 'Indigestion': 30, 'Headache': 31, 'Yellowish skin': 32, 
    'Dark urine': 33, 'Nausea': 34, 'Loss of appetite': 35, 'Pain behind the eyes': 36, 
    'Back pain': 37, 'Constipation': 38, 'Abdominal pain': 39, 'Diarrhoea': 40, 
    'Mild fever': 41, 'Yellow urine': 42, 'Yellowing of eyes': 43, 'Acute liver failure': 44, 
    'Fluid overload': 45, 'Swelling of stomach': 46, 'Swelled lymph nodes': 47, 
    'Malaise': 48, 'Blurred and Distorted vision': 49, 'Phlegm': 50, 'Throat irritation': 51, 
    'Redness of eyes': 52, 'Sinus pressure': 53, 'Runny nose': 54, 'Congestion': 55, 
    'Chest pain': 56, 'Weakness in limbs': 57, 'Fast heart rate': 58, 
    'Pain during Bowel movements': 59, 'Pain in Anal region': 60, 'Bloody stool': 61, 
    'Irritation in Anus': 62, 'Neck pain': 63, 'Dizziness': 64, 'Cramps': 65, 'Bruising': 66, 
    'Obesity': 67, 'Swollen legs': 68, 'Swollen blood vessels': 69, 'Puffy face and eyes': 70, 
    'Enlarged thyroid': 71, 'Brittle nails': 72, 'Swollen extremeties': 73, 
    'Excessive hunger': 74, 'Extra marital contacts': 75, 'Drying and tingling lips': 76, 
    'Slurred speech': 77, 'Knee pain': 78, 'Hip joint pain': 79, 'Muscle weakness': 80, 
    'Stiff neck': 81, 'Swelling joints': 82, 'Movement stiffness': 83, 'Spinning movements': 84, 
    'Loss of balance': 85, 'Unsteadiness': 86, 'Weakness of one body side': 87, 
    'Loss of smell': 88, 'Bladder discomfort': 89, 'Foul smell of urine': 90, 
    'Continuous feel of urine': 91, 'Passage of gases': 92, 'Internal itching': 93, 
    'Toxic look (typhos)': 94, 'Depression': 95, 'Irritability': 96, 'Muscle pain': 97, 
    'Altered sensorium': 98, 'Red spots over body': 99, 'Belly pain': 100, 
    'Abnormal menstruation': 101, 'Dischromic patches': 102, 'Watering from eyes': 103, 
    'Increased appetite': 104, 'Polyuria': 105, 'Family history': 106, 'Mucoid sputum': 107, 
    'Rusty sputum': 108, 'Lack of concentration': 109, 'Visual disturbances': 110, 
    'Receiving blood transfusion': 111, 'Receiving unsterile injections': 112, 'Coma': 113, 
    'Stomach bleeding': 114, 'Distention of abdomen': 115, 'History of alcohol consumption': 116, 
    'Fluid overload.1': 117, 'Blood in sputum': 118, 'Prominent veins on calf': 119, 
    'Palpitations': 120, 'Painful walking': 121, 'Pus filled pimples': 122, 'Blackheads': 123, 
    'Scurring': 124, 'Skin peeling': 125, 'Silver like dusting': 126, 'Small dents in nails': 127, 
    'Inflammatory nails': 128, 'Blister': 129, 'Red sore around nose': 130, 'Yellow crust ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes', 
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine', 
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 
    13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 
    26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 
    5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo', 2: 'Acne', 
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
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
        
        wrkout = workout[workout['disease'] == predicted_disease]['workout'].tolist()
        
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
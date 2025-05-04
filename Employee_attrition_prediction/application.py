from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('grid.pkl', 'rb') as f:
    model = pickle.load(f)

# Define features
features = [
    'DailyRate',
    'BusinessTravel_encoded',
    'EducationField_encoded',
    'Gender_encoded',
    'Department_encoded',
    'JobRole_encoded',
    'MaritalStatus_encoded',
    'Rate_to_income',
    'Income_per_hour',
    'Working_years_per_company',
    'Promotion_frequency',
    'Distance_JobSatisfaction',
    'WorkLife_Satisfaction',
    'YearsInRole_ManagerRatio',
    'JobLevel_Education',
    'JobStability_Score',
    'Engagement_Score',
    'PerformanceToReward',
    'SkillInvestment_Score',
    'Stress_score'
]

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[feature]) for feature in features]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        result = "Likely to Leave the Company" if prediction == 1 else "Not Likely to Leave the Company"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

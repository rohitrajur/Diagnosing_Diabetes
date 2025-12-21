from flask import Flask, request, render_template, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask_cors import CORS
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

# Load model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to evaluate metrics
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
y_pred = model.predict(X)

metrics = {
    'accuracy': f"{accuracy_score(y, y_pred):.4f}",
    'precision': f"{precision_score(y, y_pred):.4f}",
    'recall': f"{recall_score(y, y_pred):.4f}",
    'f1': f"{f1_score(y, y_pred):.4f}"
}

# Health tips
TIPS = [
    "Eat a balanced diet rich in whole grains, vegetables, and lean proteins.",
    "Exercise regularly to improve insulin sensitivity.",
    "Monitor your blood glucose levels consistently.",
    "Avoid sugary drinks and processed foods.",
    "Stay hydrated and manage stress effectively.",
    "Ensure adequate sleep and maintain a healthy weight.",
    "Consult your doctor regularly for checkups.",
    "Take medication as prescribed without skipping doses."
]

# BMI category helper
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Healthy"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Generate bar chart
def generate_risk_chart(file_id, features):
    labels = ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
    values = [features[1], features[2], features[4], features[5], features[7]]
    colors = ['#4ade80', '#60a5fa', '#fbbf24', '#f87171', '#a78bfa']

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.title("Risk Factors Overview")
    plt.ylabel("Value")
    plt.tight_layout()

    chart_path = f"reports/{file_id}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

@app.route('/')
def index():
    return render_template("index.html", metrics=metrics, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patient_name = data.get('Name', 'Unknown').strip()
        patient_file_name = patient_name.replace(" ", "_")

        features = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]

        prediction = model.predict([np.array(features)])[0]
        result = 'Diabetic' if prediction == 1 else 'Not Diabetic'

        # Log prediction
        log_file = "predictions_log.csv"
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Prediction"]
        new_row = features + [result]

        if not os.path.exists(log_file):
            pd.DataFrame([new_row], columns=cols).to_csv(log_file, index=False)
        else:
            pd.DataFrame([new_row], columns=cols).to_csv(log_file, mode='a', header=False, index=False)

        # Generate chart and BMI category
        chart_path = generate_risk_chart(patient_file_name, features)
        bmi_category = get_bmi_category(features[5])

        # PDF generation
        pdf_path = os.path.join("reports", f"{patient_file_name}.pdf")
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 10, "DIABETES MEDICAL REPORT", ln=True, align="C")

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
        pdf.cell(0, 10, f"Gender: {data['Gender']}", ln=True)
        pdf.cell(0, 10, f"BMI Category: {bmi_category}", ln=True)

        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(255, 0, 0) if result == "Diabetic" else pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 10, f"Prediction Result: {result}", ln=True)
        pdf.set_text_color(0, 0, 0)

        # Table
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(60, 10, "Parameter", 1, 0, "C")
        pdf.cell(60, 10, "Value", 1, 1, "C")

        pdf.set_font("Arial", size=12)
        for name, val in zip(cols[:-1], features):
            pdf.cell(60, 10, name, 1, 0)
            pdf.cell(60, 10, str(val), 1, 1)

        # Risk chart
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10,"Risk Analysis Chart", ln=True)
        pdf.image(chart_path, x=30, w=150)

        # Lifestyle Tips
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 10, "Lifestyle Tips", ln=True)

        pdf.set_text_color(0, 0, 0)
        pdf.set_left_margin(20)
        pdf.set_font("Arial", size=11)
        for tip in TIPS:
            pdf.multi_cell(180, 8, f"- {tip}", align='L')
            pdf.ln(1)

        pdf.ln(10)
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "Generated by Smart Diabetes Risk Checker", ln=True, align="C")

        pdf.output(pdf_path, "F")
        return jsonify({'prediction': result, 'report_url': f"/download_report/{patient_file_name}"})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    file_path = os.path.join("reports", f"{filename}.pdf")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "Report not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

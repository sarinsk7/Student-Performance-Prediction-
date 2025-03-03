from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check if dataset exists
DATASET_PATH = 'student_performance_data.csv'
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Preprocessing
FEATURES = ['Attendance (%)', 'Assignment Score', 'Quiz Score', 'Final Exam Score']
TARGET = 'Pass/Fail'

if not all(col in df.columns for col in FEATURES + [TARGET]):
    raise ValueError("Dataset does not contain the required columns.")

X = df[FEATURES]
y = df[TARGET]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN model dynamically based on dataset size
k_value = min(5, len(X_train) // 2)  # Avoid too large k for small datasets
knn = KNeighborsClassifier(n_neighbors=max(1, k_value), metric='euclidean')
knn.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ['attendance', 'assignment_score', 'quiz_score', 'final_exam_score']
        
        # Validate input
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing input fields'}), 400
        
        try:
            input_values = [float(data[field]) for field in required_fields]
        except ValueError:
            return jsonify({'error': 'Invalid input values'}), 400
        
        # Transform input for model
        input_data = scaler.transform([input_values])
        prediction = knn.predict(input_data)[0]
        
        return jsonify({'predicted_result': 'Pass' if prediction == 1 else 'Fail'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import os
import pickle
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the saved Scikit-Learn logistic regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Inline HTML Template with embedded CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Risk Level Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        body {
            background-color: #f4f6f8;
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 650px;
            padding: 30px;
        }
        h2 {
            text-align: center;
            margin-bottom: 24px;
            color: #1a202c;
        }
        .grid-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: #4a5568;
        }
        input, select {
            padding: 10px 12px;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease;
        }
        input:focus, select:focus {
            border-color: #3182ce;
            box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.15);
        }
        button {
            grid-column: span 2;
            background-color: #3182ce;
            color: white;
            font-weight: 600;
            padding: 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            margin-top: 10px;
            transition: background-color 0.2s ease;
        }
        button:hover {
            background-color: #2b6cb0;
        }
        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .At-Risk { background-color: #feebc8; color: #c05621; border: 1px solid #fbd38d; }
        .High-Risk { background-color: #fed7d7; color: #c53030; border: 1px solid #feb2b2; }
        .Safe { background-color: #c6f6d5; color: #22543d; border: 1px solid #9ae6b4; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Risk Level Prediction</h2>
        <form action="/predict" method="POST" class="grid-form">
            
            <div class="form-group">
                <label for="attendance">Attendance (%)</label>
                <input type="number" step="any" name="attendance" id="attendance" required placeholder="0 - 100">
            </div>

            <div class="form-group">
                <label for="study_hours">Study Hours / Week</label>
                <input type="number" step="any" name="study_hours" id="study_hours" required placeholder="e.g. 15">
            </div>

            <div class="form-group">
                <label for="past_failures">Past Failures</label>
                <input type="number" name="past_failures" id="past_failures" required placeholder="0, 1, 2...">
            </div>

            <div class="form-group">
                <label for="assignments_completed_pct">Assignments Completed (%)</label>
                <input type="number" step="any" name="assignments_completed_pct" id="assignments_completed_pct" required placeholder="0 - 100">
            </div>

            <!-- Categorical Field: Parental Education -->
            <div class="form-group">
                <label for="parental_education">Parental Education</label>
                <select name="parental_education" id="parental_education" required>
                    <option value="" disabled selected>Select Category</option>
                    <option value="0">High School</option>
                    <option value="1">Associate Degree</option>
                    <option value="2">Bachelor's Degree</option>

                    <option value="3">Master's Degree / Higher</option>
                </select>
            </div>

            <!-- Categorical Field: Family Income -->
            <div class="form-group">
                <label for="family_income">Family Income Level</label>
                <select name="family_income" id="family_income" required>
                    <option value="" disabled selected>Select Category</option>
                    <option value="0">Low</option>
                    <option value="1">Medium</option>
                    <option value="2">High</option>
                </select>
            </div>

            <!-- Categorical Field: Extracurricular -->
            <div class="form-group">
                <label for="extracurricular">Extracurricular Activities</label>
                <select name="extracurricular" id="extracurricular" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <!-- Categorical Field: Internet Access -->
            <div class="form-group">
                <label for="internet_access">Internet Access</label>
                <select name="internet_access" id="internet_access" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="previous_grade">Previous Grade Score</label>
                <input type="number" step="any" name="previous_grade" id="previous_grade" required placeholder="e.g. 75">
            </div>

            <div class="form-group">
                <label for="final_score">Final Score</label>
                <input type="number" step="any" name="final_score" id="final_score" required placeholder="e.g. 80">
            </div>

            <button type="submit">Predict Risk Status</button>
        </form>

        {% if prediction %}
        <div class="result-box {{ prediction_class }}">
            Predicted Student Status: <strong>{{ prediction }}</strong>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features in the exact order expected by the model
        feature_order = [
            "attendance",
            "study_hours",
            "past_failures",
            "assignments_completed_pct",
            "parental_education",
            "family_income",
            "extracurricular",
            "internet_access",
            "previous_grade",
            "final_score"
        ]

        # Convert form values to floating point values for inference
        features = [float(request.form.get(feature)) for feature in feature_order]
        
        # Predict class using loaded model
        prediction_result = model.predict([features])[0]
        
        # Safe CSS class name handling
        css_class = prediction_result.replace(" ", "-")

        return render_template_string(
            HTML_TEMPLATE,
            prediction=prediction_result,
            prediction_class=css_class
        )
    except Exception as e:
        return f"Error processing prediction: {str(e)}", 400

# Vercel requires the app object to be exposed directly
if __name__ == "__main__":
    app.run(debug=True)

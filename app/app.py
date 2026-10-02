# ==============================================================
# BANK CUSTOMER CHURN PREDICTION SYSTEM
# PHASE 4 - FLASK DEPLOYMENT
# ==============================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path


# ==============================================================
# 1. FLASK APPLICATION
# ==============================================================

app = Flask(__name__)


# ==============================================================
# 2. PROJECT PATHS
# ==============================================================

# app.py is inside:
# Bank_Customer_Churn_Risk/app/
#
# parent      = app/
# parent.parent = Bank_Customer_Churn_Risk/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"


# ==============================================================
# 3. MODEL FILE PATHS
# ==============================================================

MODEL_PATH = MODELS_DIR / "final_churn_model.pkl"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

THRESHOLD_PATH = MODELS_DIR / "final_threshold.pkl"


# ==============================================================
# 4. CHECK MODEL FILES
# ==============================================================

print("=" * 70)
print("BANK CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 70)

print("\nPROJECT PATHS")
print("-" * 70)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Models Folder: {MODELS_DIR}")

print("\nCHECKING REQUIRED FILES")
print("-" * 70)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nFinal model not found:\n{MODEL_PATH}"
    )


if not PREPROCESSOR_PATH.exists():

    raise FileNotFoundError(
        f"\nPreprocessor not found:\n{PREPROCESSOR_PATH}"
    )


if not THRESHOLD_PATH.exists():

    raise FileNotFoundError(
        f"\nThreshold file not found:\n{THRESHOLD_PATH}"
    )


print("✓ final_churn_model.pkl found")
print("✓ preprocessor.pkl found")
print("✓ final_threshold.pkl found")


# ==============================================================
# 5. LOAD TRAINED COMPONENTS
# ==============================================================

print("\nLOADING TRAINED COMPONENTS")
print("-" * 70)


model = joblib.load(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

threshold = joblib.load(
    THRESHOLD_PATH
)


print("✓ Final model loaded")
print("✓ Preprocessor loaded")
print(
    f"✓ Final threshold loaded: {threshold:.2f}"
)


# ==============================================================
# 6. FEATURE ENGINEERING
# ==============================================================

def create_features(data):

    """
    Creates the same engineered features
    used during Phase 2 preprocessing.
    """

    data = data.copy()


    # ----------------------------------------------------------
    # Balance to Salary
    # ----------------------------------------------------------

    data["Balance_to_Salary"] = (
        data["Balance"] /
        (data["EstimatedSalary"] + 1)
    )


    # ----------------------------------------------------------
    # Product Density
    # ----------------------------------------------------------

    data["Product_Density"] = (
        data["NumOfProducts"] /
        (data["Age"] + 1)
    )


    # ----------------------------------------------------------
    # Engagement Product Interaction
    # ----------------------------------------------------------

    data["Engagement_Product_Interaction"] = (
        data["IsActiveMember"] *
        data["NumOfProducts"]
    )


    # ----------------------------------------------------------
    # Age Tenure Interaction
    # ----------------------------------------------------------

    data["Age_Tenure_Interaction"] = (
        data["Age"] *
        data["Tenure"]
    )


    return data


# ==============================================================
# 7. RISK LEVEL FUNCTION
# ==============================================================

def determine_risk(probability):

    """
    Converts churn probability into
    a simple risk category.
    """

    if probability < 0.30:

        return "LOW", "low"

    elif probability < 0.60:

        return "MEDIUM", "medium"

    else:

        return "HIGH", "high"


# ==============================================================
# 8. RECOMMENDATION FUNCTION
# ==============================================================

def get_recommendation(risk_level):

    if risk_level == "HIGH":

        return (
            "Consider proactive retention measures such as "
            "personalized offers, customer support follow-ups, "
            "loyalty benefits, or targeted engagement."
        )

    elif risk_level == "MEDIUM":

        return (
            "Monitor customer engagement and consider targeted "
            "retention strategies to reduce future churn risk."
        )

    else:

        return (
            "The customer currently shows relatively low churn "
            "risk. Continue regular engagement and service."
        )


# ==============================================================
# 9. HOME PAGE
# ==============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ==============================================================
# 10. PREDICTION ROUTE
# ==============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("\n" + "=" * 70)
        print("NEW CUSTOMER PREDICTION")
        print("=" * 70)


        # ======================================================
        # READ INPUT VALUES
        # ======================================================

        credit_score = float(
            request.form["CreditScore"]
        )

        geography = request.form["Geography"]

        gender = request.form["Gender"]

        age = float(
            request.form["Age"]
        )

        tenure = float(
            request.form["Tenure"]
        )

        balance = float(
            request.form["Balance"]
        )

        num_products = float(
            request.form["NumOfProducts"]
        )

        has_card = float(
            request.form["HasCrCard"]
        )

        active_member = float(
            request.form["IsActiveMember"]
        )

        estimated_salary = float(
            request.form["EstimatedSalary"]
        )


        # ======================================================
        # BASIC INPUT VALIDATION
        # ======================================================

        if not 300 <= credit_score <= 900:

            raise ValueError(
                "Credit Score must be between 300 and 900."
            )


        if not 18 <= age <= 100:

            raise ValueError(
                "Age must be between 18 and 100."
            )


        if not 0 <= tenure <= 10:

            raise ValueError(
                "Tenure must be between 0 and 10."
            )


        if balance < 0:

            raise ValueError(
                "Balance cannot be negative."
            )


        if not 1 <= num_products <= 4:

            raise ValueError(
                "Number of products must be between 1 and 4."
            )


        if estimated_salary < 0:

            raise ValueError(
                "Estimated salary cannot be negative."
            )


        # ======================================================
        # CREATE INPUT DATAFRAME
        # ======================================================

        input_data = pd.DataFrame({

            "CreditScore": [
                credit_score
            ],

            "Geography": [
                geography
            ],

            "Gender": [
                gender
            ],

            "Age": [
                age
            ],

            "Tenure": [
                tenure
            ],

            "Balance": [
                balance
            ],

            "NumOfProducts": [
                num_products
            ],

            "HasCrCard": [
                has_card
            ],

            "IsActiveMember": [
                active_member
            ],

            "EstimatedSalary": [
                estimated_salary
            ]

        })


        print("\nOriginal Input:")
        print(
            input_data.to_string(
                index=False
            )
        )


        # ======================================================
        # FEATURE ENGINEERING
        # ======================================================

        input_data = create_features(
            input_data
        )


        print("\nEngineered Features:")
        print(
            input_data.to_string(
                index=False
            )
        )


        # ======================================================
        # PREPROCESSING
        # ======================================================

        processed_data = preprocessor.transform(
            input_data
        )


        print(
            f"\nProcessed Input Shape: "
            f"{processed_data.shape}"
        )


        # ======================================================
        # PREDICTION PROBABILITY
        # ======================================================

        probability = model.predict_proba(
            processed_data
        )[0][1]


        probability_percent = (
            probability * 100
        )


        # ======================================================
        # APPLY FINAL THRESHOLD
        # ======================================================

        prediction = int(
            probability >= threshold
        )


        # ======================================================
        # RISK LEVEL
        # ======================================================

        risk_level, risk_class = determine_risk(
            probability
        )


        # ======================================================
        # PREDICTION TEXT
        # ======================================================

        if prediction == 1:

            prediction_text = (
                "Customer Likely to Churn"
            )

        else:

            prediction_text = (
                "Customer Likely to Stay"
            )


        # ======================================================
        # RECOMMENDATION
        # ======================================================

        recommendation = get_recommendation(
            risk_level
        )


        # ======================================================
        # PRINT RESULT
        # ======================================================

        print("\nPREDICTION RESULT")
        print("-" * 70)

        print(
            f"Churn Probability : "
            f"{probability_percent:.2f}%"
        )

        print(
            f"Threshold         : "
            f"{threshold:.2f}"
        )

        print(
            f"Prediction        : "
            f"{prediction_text}"
        )

        print(
            f"Risk Level        : "
            f"{risk_level}"
        )


        # ======================================================
        # SEND RESULT TO HTML
        # ======================================================

        return render_template(

            "index.html",

            prediction=prediction_text,

            probability=f"{probability_percent:.2f}",

            risk_level=risk_level,

            risk_class=risk_class,

            recommendation=recommendation,

            form_data=request.form

        )


    except Exception as e:

        print("\nERROR:")
        print(str(e))


        return render_template(

            "index.html",

            error=(
                "Prediction error: "
                + str(e)
            ),

            form_data=request.form

        )


# ==============================================================
# 11. RUN FLASK APPLICATION
# ==============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "Starting Bank Customer Churn Prediction System..."
    )

    print(
        "Open your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
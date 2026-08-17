import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import joblib
import json


# --- CONFIGURATION ---
# Part 1: Risk Intelligence Service Weights (These are configurable as per the SRS)
# Note: Weights should ideally sum to 1 for each score.
SUPPLIER_RISK_WEIGHTS = {
    'financial_health_score': 0.4,
    'delivery_performance': 0.3,
    'political_risk_index': 0.2,
    'compliance_violations': 0.1
}

ROUTE_RISK_WEIGHTS = {
    'customs_clearance_hours': 0.5,
    'weather_delay_hours': 0.5
}

# Part 2: ETA Model Configuration
MODEL_VERSION = "1.0.0"

def calculate_risk_scores(suppliers_df, shipments_df):
    """
    Calculates supplier and route risk scores based on pre-defined weights.
    This function fulfills the "Risk Intelligence Service" requirement.
    """
    print("Step 1: Calculating Risk Intelligence Scores...")
    
    # --- Supplier Risk Score ---
    # Normalize the input features to a 0-1 scale so weights can be applied fairly.
    scaler = MinMaxScaler()
    
    # Note: We invert financial_health and delivery_performance because a HIGHER score is BETTER.
    # For risk, a LOWER score is better, so we subtract from 1.
    suppliers_df['financial_health_normalized'] = 1 - scaler.fit_transform(suppliers_df[['financial_health_score']])
    suppliers_df['delivery_performance_normalized'] = 1 - scaler.fit_transform(suppliers_df[['delivery_performance']])
    suppliers_df['political_risk_normalized'] = scaler.fit_transform(suppliers_df[['political_risk_index']])
    suppliers_df['compliance_violations_normalized'] = scaler.fit_transform(suppliers_df[['compliance_violations']])
    
    # Calculate the weighted score
    suppliers_df['supplier_risk_score'] = (
        suppliers_df['financial_health_normalized'] * SUPPLIER_RISK_WEIGHTS['financial_health_score'] +
        suppliers_df['delivery_performance_normalized'] * SUPPLIER_RISK_WEIGHTS['delivery_performance'] +
        suppliers_df['political_risk_normalized'] * SUPPLIER_RISK_WEIGHTS['political_risk_index'] +
        suppliers_df['compliance_violations_normalized'] * SUPPLIER_RISK_WEIGHTS['compliance_violations']
    )
    
    # --- Route Risk Score ---
    shipments_df['customs_normalized'] = scaler.fit_transform(shipments_df[['customs_clearance_hours']])
    shipments_df['weather_normalized'] = scaler.fit_transform(shipments_df[['weather_delay_hours']])
    
    shipments_df['route_risk_score'] = (
        shipments_df['customs_normalized'] * ROUTE_RISK_WEIGHTS['customs_clearance_hours'] +
        shipments_df['weather_normalized'] * ROUTE_RISK_WEIGHTS['weather_delay_hours']
    )
    
    print("-> Risk scores calculated successfully.")
    return suppliers_df, shipments_df


def train_eta_prediction_model(suppliers_df, shipments_df):
    """
    Trains models to predict delay probability and delay days.
    This function fulfills the "ETA Prediction Model" requirement.
    """
    print("\nStep 2: Building ETA Prediction Model...")
    
    # Merge supplier data (especially the new risk score) into shipments data
    full_data = pd.merge(shipments_df, suppliers_df[['supplier_id', 'supplier_risk_score']], on='supplier_id', how='left')
    
    # --- Feature Engineering ---
    # Create the target variables
    # 1. Is the shipment delayed? (For classification)
    full_data['is_delayed'] = full_data['status'].apply(lambda x: 1 if x == 'Delayed' else 0)
    # 2. How many days was the delay? (For regression)
    full_data['delay_days'] = full_data['weather_delay_hours'] / 24

    # Select features for the model
    features = [
        'shipment_value_usd', 'weight_kg', 'volume_cbm', 
        'freight_cost_usd', 'route_optimization_score', 
        'supplier_risk_score', 'route_risk_score'
    ]
    
    # Convert categorical features into numbers using one-hot encoding
    categorical_features = ['origin_port', 'destination_port', 'product_category', 'transportation_mode']
    full_data = pd.get_dummies(full_data, columns=categorical_features, drop_first=True)
    
    # Update feature list with new dummy columns
    encoded_cols = [col for col in full_data.columns if any(cat_feat in col for cat_feat in categorical_features)]
    features.extend(encoded_cols)

    # Handle potential missing values (simple imputation)
    full_data[features] = full_data[features].fillna(full_data[features].median())

    X = full_data[features]
    y_class = full_data['is_delayed']
    y_reg = full_data['delay_days']
    
    # Split data into training and testing sets
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=42
    )

    # --- Train Classification Model (for delay probability) ---
    print("-> Training classification model for delay probability...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1) # Changed to n_jobs=1 to prevent potential memory issues
    classifier.fit(X_train, y_class_train)
    delay_probabilities = classifier.predict_proba(X)[:, 1] # Get probability of class '1' (Delayed)

    # --- Train Regression Model (for delay days) ---
    print("-> Training regression model for expected delay days...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1) # Changed to n_jobs=1 to prevent potential memory issues
    regressor.fit(X_train, y_reg_train)
    delay_day_predictions = regressor.predict(X)

    # --- Evaluate Models (as per SRS) ---
    print("\n--- Model Evaluation ---")
    test_probs = classifier.predict_proba(X_test)[:, 1]
    brier_score = brier_score_loss(y_class_test, test_probs)
    print(f"Brier Score (lower is better): {brier_score:.4f}")

    test_reg_preds = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, test_reg_preds))
    print(f"Delay Days RMSE (lower is better): {rmse:.4f} days")
    
    # Generate Reliability Curve plot (as per SRS)
    calibration_plot_path = "ETA_model/ETA_training/calibration_plot.png"
    print(f"-> Generating reliability curve plot ({calibration_plot_path})...")
    fig, ax = plt.subplots()
    CalibrationDisplay.from_estimator(classifier, X_test, y_class_test, ax=ax)
    plt.title("Reliability Curve (Calibration Plot)")
    plt.savefig(calibration_plot_path)
    plt.close()
    
    # --- Add predictions to the final dataframe ---
    full_data['predicted_delay_prob'] = delay_probabilities
    full_data['predicted_delay_days'] = delay_day_predictions
    full_data['prediction_timestamp'] = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")
    full_data['model_version'] = MODEL_VERSION
    
    print("-> ETA prediction model built successfully.")
    return full_data

if __name__ == "__main__":
    print("--- Starting V1 Model Build Process ---")
    
    # Load the datasets from the 'dataset' folder
    try:
        suppliers = pd.read_csv("dataset/project1_suppliers.csv")
        shipments = pd.read_csv("dataset/project1_shipments.csv")
        print("-> Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required data file: {e.filename}")
        print("Please ensure you are running this script from the main 'ashwini/' directory.")
        exit()

    # Step 1: Run the Risk Intelligence Service
    suppliers_with_risk, shipments_with_risk = calculate_risk_scores(suppliers, shipments)

    # Save the output with risk scores for analysis to the 'risk' folder
    risk_output_path = "risk/risk_scores_output.csv"
    suppliers_with_risk.to_csv(risk_output_path, index=False)
    
    # Step 2: Run the ETA Prediction Model Builder
    final_predictions_df = train_eta_prediction_model(suppliers_with_risk, shipments_with_risk)

    # Keep only the original columns plus the new required output columns
    original_shipment_cols = list(shipments.columns)
    output_cols = original_shipment_cols + [
        'predicted_delay_prob', 'predicted_delay_days', 'prediction_timestamp', 'model_version'
    ]
    # Filter out columns that might not exist in the final df (like dummy cols)
    output_cols_exist = [col for col in output_cols if col in final_predictions_df.columns]
    
    final_predictions_df = final_predictions_df[output_cols_exist]
    
    # Save the final predictions to the 'ETA_model/ETA_prediction' folder
    prediction_output_path = "ETA_model/ETA_prediction/eta_prediction_output.csv"
    final_predictions_df.to_csv(prediction_output_path, index=False)
    
    print("\n--- Process Complete ---")
    print("Files have been created in their respective folders:")
    print(f"1. {risk_output_path}")
    print(f"2. {prediction_output_path}")
    print(f"3. ETA_model/ETA_training/calibration_plot.png")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, mean_squared_error # --- ADDED ---
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt
import joblib
import json
import os

# --- ETA Model Configuration ---
MODEL_VERSION = "2.0.0"

def train_eta_prediction_model(suppliers_df, shipments_df):
    """
    Trains a final CALIBRATED model to predict delay probability and delay days.
    """
    print("\nStep 2: Building Final ETA Prediction Model...")
    
    # Merge the risk-scored data
    full_data = pd.merge(
        shipments_df, 
        suppliers_df[['supplier_id', 'supplier_risk_score']], 
        on='supplier_id', how='left'
    )
    
    # Feature Engineering
    full_data['is_delayed'] = full_data['status'].apply(lambda x: 1 if x == 'Delayed' else 0)
    full_data['delay_days'] = full_data['weather_delay_hours'] / 24

    features = [
        'shipment_value_usd', 'weight_kg', 'volume_cbm', 
        'freight_cost_usd', 'route_optimization_score', 
        'supplier_risk_score', 'route_risk_score'
    ]
    
    categorical_features = ['origin_port', 'destination_port', 'product_category', 'transportation_mode']
    full_data = pd.get_dummies(full_data, columns=categorical_features, drop_first=True)
    
    encoded_cols = [col for col in full_data.columns if any(cat_feat in col for cat_feat in categorical_features)]
    features.extend(encoded_cols)
    full_data[features] = full_data[features].fillna(full_data[features].median())

    X = full_data[features]
    y_class = full_data['is_delayed']
    y_reg = full_data['delay_days']
    
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=42
    )

    # --- Train Final Calibrated Classifier ---
    print("-> Training calibrated classifier...")
    base_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    calibrated_classifier = CalibratedClassifierCV(
        base_classifier, method='isotonic', cv=5, n_jobs=-1
    )
    calibrated_classifier.fit(X_train, y_class_train)
    delay_probabilities = calibrated_classifier.predict_proba(X)[:, 1]

    # --- Regression Model ---
    print("-> Training regression model for expected delay days...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_reg_train)
    delay_day_predictions = regressor.predict(X)

    # --- Model Evaluation ---
    print("\n--- Model Evaluation ---")
    cal_probs = calibrated_classifier.predict_proba(X_test)[:, 1]
    brier_score = brier_score_loss(y_class_test, cal_probs)
    print(f"Final Calibrated Brier Score: {brier_score:.4f} (Lower is better!)")
    
    # --- ADDED: RMSE Calculation ---
    test_reg_preds = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, test_reg_preds))
    print(f"Delay Days RMSE: {rmse:.4f} days (Lower is better!)")
    
    # --- ADDED: Save metrics to a JSON file for tracking ---
    metrics = {
        'model_version': MODEL_VERSION,
        'brier_score': brier_score,
        'delay_days_rmse': rmse,
        'evaluation_timestamp': pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")
    }
    with open('ETA_model/ETA_training/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    # --- END OF ADDED SECTION ---

    # --- Save the models and features ---
    print("-> Saving models and feature list...")
    os.makedirs("ETA_model/ETA_prediction", exist_ok=True)
    os.makedirs("ETA_model/ETA_training", exist_ok=True)
    
    joblib.dump(calibrated_classifier, 'ETA_model/ETA_prediction/classifier_model.joblib')
    joblib.dump(regressor, 'ETA_model/ETA_prediction/regressor_model.joblib')
    
    feature_list = list(X.columns)
    with open('ETA_model/ETA_training/model_features.json', 'w') as f:
        json.dump(feature_list, f)
    
    print("-> Models and features saved successfully.")

    # --- Generate Calibration Plot ---
    print("-> Generating reliability plot (calibration_plot_final.png)...")
    fig, ax = plt.subplots()
    CalibrationDisplay.from_estimator(calibrated_classifier, X_test, y_class_test, ax=ax, name="Calibrated Model")
    plt.title("Reliability Curve for Final Model")
    plt.savefig("ETA_model/ETA_training/calibration_plot_final.png")
    plt.close()
    
    # --- Add predictions to the final dataframe ---
    # Using .copy() to avoid PerformanceWarning about fragmentation
    full_data = full_data.copy() 
    full_data['predicted_delay_prob'] = delay_probabilities
    full_data['predicted_delay_days'] = delay_day_predictions
    full_data['prediction_timestamp'] = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")
    full_data['model_version'] = MODEL_VERSION
    
    print("-> ETA prediction model built successfully.")
    return full_data


if __name__ == "__main__":
    print("--- Starting Final ETA Model Training Process ---")
    
    try:
        suppliers_with_risk = pd.read_csv("risk/risk_scores_suppliers.csv")
        shipments_with_risk = pd.read_csv("risk/risk_scores_shipments.csv")
        original_shipments = pd.read_csv("dataset/project1_shipments.csv")
        print("Risk-scored and original data loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file. {e}")
        print("Please ensure risk_analyzer.py has been run and its outputs are in the 'risk/' folder.")
        exit()
    
    final_predictions_df = train_eta_prediction_model(suppliers_with_risk, shipments_with_risk)

    # Format the final output
    original_shipment_cols = list(original_shipments.columns)
    output_cols = original_shipment_cols + [
        'supplier_risk_score', 'route_risk_score', 
        'predicted_delay_prob', 'predicted_delay_days', 
        'prediction_timestamp', 'model_version'
    ]
    output_cols_exist = [col for col in output_cols if col in final_predictions_df.columns]
    
    final_predictions_df = final_predictions_df[output_cols_exist]
    final_predictions_df.to_csv("ETA_model/ETA_training/eta_prediction_output_final.csv", index=False)
    
    print("\n--- Process Complete ---")
    print("Final models saved to 'ETA_model/ETA_prediction/'.")
    print("Training outputs (plot, features, csv, metrics) saved to 'ETA_model/ETA_training/'.")

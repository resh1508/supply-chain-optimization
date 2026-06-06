import pandas as pd
import joblib
import json
import os

# --- LOAD MODELS AND FEATURES ---
print("Loading prediction models...")
try:
    # CORRECTED PATHS: Point to the 'prediction' and 'training' folders as needed.
    classifier_path = 'ETA_model/ETA_prediction/classifier_model.joblib'
    regressor_path = 'ETA_model/ETA_prediction/regressor_model.joblib'
    features_path = 'ETA_model/ETA_training/model_features.json'
    
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    with open(features_path, 'r') as f:
        model_features = json.load(f)
    print("-> Models and features loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: A required model file was not found. {e}")
    print("Please run the training script (v2_model_final.py) first to create the models.")
    exit()

def predict_single_shipment(shipment_data, supplier_risk, route_risk):
    """
    Predicts delay probability and days for a single new shipment.
    """
    print("\n--- Predicting for New Shipment ---")
    
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([shipment_data])
    
    # Add the pre-calculated risk scores
    input_df['supplier_risk_score'] = supplier_risk
    input_df['route_risk_score'] = route_risk

    # Perform the exact same one-hot encoding as in training
    input_df_encoded = pd.get_dummies(input_df)

    # **FIXED**: Align columns using reindex - this is the modern, warning-free way
    final_df = input_df_encoded.reindex(columns=model_features, fill_value=0)
    
    # Ensure the column order is identical to the training set (reindex already does this, but this is a safe double-check)
    final_df = final_df[model_features]

    # Make predictions
    prob = classifier.predict_proba(final_df)[:, 1]
    days = regressor.predict(final_df)
    
    print(f"-> Predicted Delay Probability: {prob[0]:.2%}")
    print(f"-> Predicted Delay (in Days): {days[0]:.2f}")
    
    return prob[0], days[0]

def predict_from_excel(input_file, output_file, supplier_risk, route_risk):
    """
    Predicts delay probability and days for multiple shipments from an Excel file.
    """
    print(f"\n--- Reading shipments from {input_file} ---")
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_file}'")
        return

    # Add pre-calculated risks to every shipment
    df['supplier_risk_score'] = supplier_risk
    df['route_risk_score'] = route_risk

    # One-hot encode
    df_encoded = pd.get_dummies(df)

    # **FIXED**: Align with model features using reindex
    final_df = df_encoded.reindex(columns=model_features, fill_value=0)
    
    # Ensure column order
    final_df = final_df[model_features]

    # Predictions
    df['predicted_delay_probability'] = classifier.predict_proba(final_df)[:, 1]
    df['predicted_delay_days'] = regressor.predict(final_df)

    # Save to new Excel
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)
    print(f"-> Predictions saved to {output_file}")


if __name__ == "__main__":
    # --- SCENARIO 1: Predict a single, ad-hoc shipment ---
    new_shipment = {
        'shipment_value_usd': 50000,
        'weight_kg': 1200,
        'volume_cbm': 5.5,
        'freight_cost_usd': 2500,
        'route_optimization_score': 85,
        'origin_port': 'Port of Shanghai',
        'destination_port': 'Port of Los Angeles',
        'product_category': 'Electronics',
        'transportation_mode': 'Ocean'
    }
    example_supplier_risk = 0.72 
    example_route_risk = 0.45
    predict_single_shipment(new_shipment, example_supplier_risk, example_route_risk)
    
    print("\n" + "="*50 + "\n")

    # --- SCENARIO 2: Predict for a batch of new shipments from an Excel file ---
    # CORRECTED PATHS for batch prediction
    input_excel = "ETA_model/ETA_prediction/predict_excel_sample.xlsx"
    output_excel = "ETA_model/ETA_prediction/shipments_with_predictions.xlsx"
    
    # These risks could be dynamic, but we'll use an example for this run
    predict_from_excel(input_excel, output_excel, example_supplier_risk, example_route_risk)

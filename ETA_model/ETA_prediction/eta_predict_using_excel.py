import pandas as pd
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

def process_and_predict(input_excel_path, output_excel_path, supplier_risks_path, config_path, models_dir, training_artifacts_dir):
    """
    Loads new shipments, looks up pre-calculated supplier risks, calculates route risks,
    makes predictions, and saves the result.
    """
    
    # --- 1. Load All Necessary Files ---
    print("Step 1: Loading models, configuration, and data...")
    try:
        # Load all dependencies using the provided paths
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        classifier = joblib.load(os.path.join(models_dir, 'classifier_model.joblib'))
        regressor = joblib.load(os.path.join(models_dir, 'regressor_model.joblib'))
        
        with open(os.path.join(training_artifacts_dir, 'model_features.json'), 'r') as f:
            model_features = json.load(f)

        # Load the new shipment data and pre-calculated supplier risks
        df = pd.read_excel(input_excel_path)
        supplier_risks_df = pd.read_csv(supplier_risks_path)
        print(f"-> Loaded {len(df)} new shipments and {len(supplier_risks_df)} supplier risks.")

    except FileNotFoundError as e:
        print(f"ERROR: A required file is missing. Please check the path for: {e.filename}")
        print("-> Ensure you have run 'risk_analyzer.py' and 'v2_model_final.py' first.")
        return

    # --- 2. Look Up Supplier Risk & Calculate Route Risk ---
    print("\nStep 2: Looking up supplier risks and calculating route risks...")

    # Merge to look up supplier risk for each new shipment
    df = pd.merge(
        df,
        supplier_risks_df[['supplier_id', 'supplier_risk_score']],
        on='supplier_id',
        how='left'
    )

    # Handle cases where a supplier_id in new shipments might not be in our risk file
    if df['supplier_risk_score'].isnull().any():
        print("WARNING: Some shipments had a supplier_id not found in the risk file. These rows will not be predicted.")
        df.dropna(subset=['supplier_risk_score'], inplace=True)

    # Calculate on-the-fly route risk for the new shipments
    scaler = MinMaxScaler()
    df['customs_normalized'] = scaler.fit_transform(df[['customs_clearance_hours']])
    df['weather_normalized'] = scaler.fit_transform(df[['weather_delay_hours']])
    df['route_risk_score'] = (
        df['customs_normalized'] * config['ROUTE_RISK_WEIGHTS']['customs_clearance_hours'] +
        df['weather_normalized'] * config['ROUTE_RISK_WEIGHTS']['weather_delay_hours']
    )
    print("-> Supplier risks looked up and route risks calculated.")
    
    # --- 3. Prepare Data for Prediction ---
    print("\nStep 3: Preparing data for prediction...")
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df)
    # Align columns with the features the model was trained on, filling missing ones with 0
    df_aligned = df_encoded.reindex(columns=model_features, fill_value=0)
    print("-> Data aligned with model features.")

    # --- 4. Make Predictions ---
    print("\nStep 4: Making predictions...")
    delay_probabilities = classifier.predict_proba(df_aligned)[:, 1]
    predicted_days = regressor.predict(df_aligned)
    
    df['predicted_delay_prob'] = delay_probabilities
    df['predicted_delay_days'] = predicted_days
    print("-> Predictions are complete.")

    # --- 5. Save Results ---
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    
    # Define the final columns for the output file
    output_cols = list(pd.read_excel(input_excel_path).columns) + [
        'supplier_risk_score', 'route_risk_score', 
        'predicted_delay_prob', 'predicted_delay_days'
    ]
    df[output_cols].to_excel(output_excel_path, index=False)
    print(f"\n✅ Success! Predictions saved to '{output_excel_path}'")


if __name__ == "__main__":
    # --- Define your file paths based on the project structure ---
    # CORRECTED PATHS
    INPUT_FILE = "ETA_model/ETA_prediction/predict_excel_sample.xlsx"
    SUPPLIER_RISK_FILE = "risk/risk_scores_suppliers.csv" 
    OUTPUT_FILE = "ETA_model/ETA_prediction/shipments_with_predictions.xlsx"
    CONFIG_FILE = "risk/config.json"
    MODELS_DIRECTORY = "ETA_model/ETA_prediction"
    TRAINING_ARTIFACTS_DIRECTORY = "ETA_model/ETA_training"
    
    process_and_predict(
        INPUT_FILE, 
        OUTPUT_FILE, 
        SUPPLIER_RISK_FILE,
        CONFIG_FILE,
        MODELS_DIRECTORY,
        TRAINING_ARTIFACTS_DIRECTORY
    )

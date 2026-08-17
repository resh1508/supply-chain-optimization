import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import json
import os # Added to ensure the output directory exists

# --- CONFIGURATION LOADING ---
# CORRECTED PATH: Load config from the same 'risk' folder as the script
try:
    with open('risk/config.json', 'r') as f:
        config = json.load(f)
    SUPPLIER_RISK_WEIGHTS = config['SUPPLIER_RISK_WEIGHTS']
    ROUTE_RISK_WEIGHTS = config['ROUTE_RISK_WEIGHTS']
    print("Configuration loaded successfully from risk/config.json.")
except FileNotFoundError:
    print("ERROR: risk/config.json not found. Please ensure it exists in the 'risk' directory.")
    exit()
except KeyError:
    print("ERROR: config.json is missing required keys like 'SUPPLIER_RISK_WEIGHTS' or 'ROUTE_RISK_WEIGHTS'.")
    exit()

def calculate_risk_scores(suppliers_df, shipments_df):
    """
    Calculates supplier and route risk scores based on pre-defined weights.
    """
    print("\nStep 1: Calculating Risk Intelligence Scores...")
    scaler = MinMaxScaler()
    
    # --- Supplier Risk Score ---
    suppliers_df['financial_health_normalized'] = 1 - scaler.fit_transform(suppliers_df[['financial_health_score']])
    suppliers_df['delivery_performance_normalized'] = 1 - scaler.fit_transform(suppliers_df[['delivery_performance']])
    suppliers_df['political_risk_normalized'] = scaler.fit_transform(suppliers_df[['political_risk_index']])
    suppliers_df['compliance_violations_normalized'] = scaler.fit_transform(suppliers_df[['compliance_violations']])
    
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


def check_risk_alerts(suppliers_with_risk_df):
    """
    Checks supplier risk scores against regional thresholds, prints alerts,
    and saves high-risk suppliers into a new Excel sheet.
    """
    print("\nStep 2: Checking for High-Risk Supplier Alerts...")
    try:
        # CORRECTED PATH: Look for thresholds file in the 'risk' folder
        thresholds_df = pd.read_csv("risk/risk_thresholds.csv")
    except FileNotFoundError:
        print("-> WARNING: risk/risk_thresholds.csv not found. Skipping alert check.")
        return

    if 'region' not in suppliers_with_risk_df.columns:
        print("-> WARNING: 'region' column not in supplier data. Skipping alert check.")
        return
        
    suppliers_merged = pd.merge(suppliers_with_risk_df, thresholds_df, on='region', how='left')
    high_risk_suppliers = suppliers_merged[suppliers_merged['supplier_risk_score'] > suppliers_merged['risk_limit']]

    if high_risk_suppliers.empty:
        print("-> All supplier risk scores are within their regional limits.")
    else:
        print("!!! ALERT: The following suppliers have exceeded their risk threshold !!!")
        for _, row in high_risk_suppliers.iterrows():
            print(
                f"  - Supplier ID: {row['supplier_id']} in Region: {row['region']} "
                f"| Score: {row['supplier_risk_score']:.2f} "
                f"| Limit: {row['risk_limit']:.2f}"
            )
        
        # CORRECTED PATH: Save alerts to the 'risk' folder
        alerts_path = "risk/high_risk_alerts.xlsx"
        high_risk_suppliers.to_excel(alerts_path, index=False)
        print(f"\n-> High-risk supplier alerts saved to '{alerts_path}'")


if __name__ == "__main__":
    print("--- Starting Risk Analysis Process ---")
    
    try:
        # CORRECTED PATH: Read raw data from the 'dataset' folder
        suppliers = pd.read_csv("dataset/project1_suppliers.csv")
        shipments = pd.read_csv("dataset/project1_shipments.csv")
        print("Data loaded successfully from 'dataset/' folder.")
    except FileNotFoundError:
        print("ERROR: Make sure 'project1_suppliers.csv' and 'project1_shipments.csv' are in the 'dataset' folder.")
        exit()

    # Calculate the risk scores
    suppliers_with_risk, shipments_with_risk = calculate_risk_scores(suppliers, shipments)

    # Run the new alert check
    check_risk_alerts(suppliers_with_risk)

    # --- Save the outputs for the next script ---
    # CORRECTED PATH: Save outputs to the 'risk' folder
    os.makedirs("risk", exist_ok=True) # Ensure the directory exists
    suppliers_output_path = "risk/risk_scores_suppliers.csv"
    shipments_output_path = "risk/risk_scores_shipments.csv"
    
    suppliers_with_risk.to_csv(suppliers_output_path, index=False)
    shipments_with_risk.to_csv(shipments_output_path, index=False)
    
    print("\n--- Process Complete ---")
    print("Risk analysis output files have been saved in the 'risk/' folder.")

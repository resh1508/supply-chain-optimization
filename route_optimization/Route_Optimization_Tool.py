import pandas as pd
import os

# ==============================================================================
# --- USER CONFIGURATION ---
# ==============================================================================
# Set the non-negotiable on-time delivery constraint for all options
ON_TIME_SLA_CONSTRAINT = 95.0  # (e.g., 95.0%)

# When optimizing for carbon, set the maximum acceptable cost increase over the cheapest option
COST_TOLERANCE_FOR_CARBON = 0.15  # (e.g., 15%)
# ==============================================================================

def create_route_options_from_data(shipments_df):
    """
    Analyzes historical shipment data to create a summary of route/mode performance.
    """
    print("Step 1: Analyzing historical data to build route performance profiles...")

    # --- Data Preparation ---
    df = shipments_df.copy()
    # Create a 'Route' identifier from origin and destination ports
    df['Route'] = df['origin_port'] + ' -> ' + df['destination_port']
    # Determine if a shipment was on-time based on the 'status' column
    df['is_on_time'] = (df['status'] != 'Delayed').astype(int)

    # --- Aggregation ---
    # Group by each unique route and mode combination to calculate performance
    route_summary = df.groupby(['Route', 'transportation_mode']).agg(
        avg_cost=('freight_cost_usd', 'mean'),
        avg_carbon=('carbon_footprint_kg', 'mean'),
        on_time_percent=('is_on_time', lambda x: x.mean() * 100), # Calculate On-Time %
        shipment_count=('shipment_id', 'count')
    ).reset_index()

    print(f"-> Created {len(route_summary)} unique route/mode profiles from {len(shipments_df)} shipments.")
    return route_summary.round(2)


def optimize_routes(route_options_df, sla_constraint, goal='cost', cost_tolerance=0.10):
    """
    Filters and sorts route options based on the user's primary goal and constraints.
    """
    # Step 1: Filter by SLA
    feasible_options = route_options_df[route_options_df['on_time_percent'] >= sla_constraint].copy()
    
    if feasible_options.empty:
        return pd.DataFrame()

    if goal == 'cost':
        print(f"🎯 Goal: Minimize Total Cost (while On-Time % is >= {sla_constraint}%)")
        frontier_table = feasible_options.sort_values(by='avg_cost', ascending=True)
        
    elif goal == 'carbon':
        min_cost_option = feasible_options.sort_values(by='avg_cost', ascending=True).iloc[0]
        cost_ceiling = min_cost_option['avg_cost'] * (1 + cost_tolerance)

        print(f"🎯 Goal: Minimize Carbon Emissions (while On-Time % is >= {sla_constraint}% and Avg Cost is <= ${cost_ceiling:,.2f})")
        options_in_budget = feasible_options[feasible_options['avg_cost'] <= cost_ceiling]
        frontier_table = options_in_budget.sort_values(by='avg_carbon', ascending=True)
        
    else:
        raise ValueError("Goal must be either 'cost' or 'carbon'")

    return frontier_table


if __name__ == "__main__":
    print("--- Data-Driven Route and Mode Optimization Tool ---")
    
    try:
        # CORRECTED PATH: Point to the 'dataset' folder to find the data file.
        shipments_path = "dataset/project1_shipments.csv"
        shipments = pd.read_csv(shipments_path)
        print(f"Historical shipment data loaded successfully from '{shipments_path}'.")
    except FileNotFoundError:
        print(f"ERROR: The file '{shipments_path}' was not found. Make sure it exists.")
        exit()

    try:
        # Step 1: Create the performance profiles from your data
        route_options = create_route_options_from_data(shipments)
        
        # Save the generated route options to a CSV file in the same directory as the script
        output_dir = "route_optimization"
        os.makedirs(output_dir, exist_ok=True)
        route_options.to_csv(os.path.join(output_dir, "route_performance_summary.csv"), index=False)
        print(f"-> Route performance summary saved to '{os.path.join(output_dir, 'route_performance_summary.csv')}'")


        # --- Scenario 1: Primary Goal - Minimize Cost ---
        print("\n\n## Scenario 1: Optimizing for MINIMUM COST")
        cost_frontier = optimize_routes(
            route_options,
            sla_constraint=ON_TIME_SLA_CONSTRAINT,
            goal='cost'
        )
        if not cost_frontier.empty:
            print(cost_frontier.to_string(index=False))
        else:
            print(f"No routes found that meet the {ON_TIME_SLA_CONSTRAINT}% on-time SLA.")

        print("\n" + "="*80 + "\n")

        # --- Scenario 2: Optional Goal - Minimize Carbon Emissions ---
        print("## Scenario 2: Optimizing for MINIMUM CARBON EMISSIONS")
        carbon_frontier = optimize_routes(
            route_options,
            sla_constraint=ON_TIME_SLA_CONSTRAINT,
            goal='carbon',
            cost_tolerance=COST_TOLERANCE_FOR_CARBON
        )
        if not carbon_frontier.empty:
            print(carbon_frontier.to_string(index=False))
        else:
            print(f"No routes found that meet the constraints.")

    except (KeyError, Exception) as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")

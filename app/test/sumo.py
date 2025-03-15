import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ---------------------------
# Utility Functions
# ---------------------------


def clear_output_folder(folder_path):
    """Clear and recreate a folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    print(f"Cleared and recreated: {folder_path}")


def generate_random_trips(
    name,
    network_file,
    output_prefix,
    begin=0,
    end=3600,
    period=1,
):
    """Generate random trips using SUMO's randomTrips.py with --random flag."""
    output_file = f"{output_prefix}_{name}.trips.xml"
    sumo_tools = os.path.join(os.getenv("SUMO_HOME", ""), "tools", "randomTrips.py")
    if not os.path.exists(sumo_tools):
        raise FileNotFoundError("randomTrips.py not found. Check SUMO_HOME.")

    command = [
        "python",
        sumo_tools,
        "-n",
        network_file,
        "-o",
        output_file,
        "--begin",
        str(begin),
        "--end",
        str(end),
        "--period",
        str(period),
        "--route-file",
        f"{output_prefix}_{name}.rou.xml",
        "--validate",
        "--random",  # Added --random flag, removed seed
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Trips generated: {output_file}")
        return output_file, f"{output_prefix}_{name}.rou.xml"
    except subprocess.CalledProcessError as e:
        print(f"Error generating trips: {e.stderr}")
        return None, None


def generate_fcd(route_file, config_file, output_prefix):
    """Generate FCD output from a route file using SUMO."""
    if not route_file:
        return None
    scenario = os.path.basename(route_file).split("_")[1].split(".")[0]
    output_file = f"{output_prefix}_{scenario}.xml"
    command = [
        "sumo",
        "-c",
        config_file,
        "--route-files",
        route_file,
        "--fcd-output",
        output_file,
        "--fcd-output.geo",
        "--random",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"FCD generated: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error generating FCD: {e.stderr}")
        return None


def process_fcd(fcd_file, scenario=None):
    """Process FCD file into a DataFrame with aggregated lane data."""
    if not fcd_file:
        return None
    try:
        tree = etree.parse(fcd_file)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: File {fcd_file} not found.")
        return None

    data = [
        {
            "time": float(t.get("time")),
            "speed": float(v.get("speed")),
            "lane": v.get("lane"),
            "vehicle": v.get("id"),
        }
        for t in root.findall("timestep")
        for v in t.findall("vehicle")
    ]
    df = pd.DataFrame(data)
    df["stopped"] = (df["speed"] < 0.1).astype(np.int8)

    agg_df = (
        df.groupby(["time", "lane"], observed=True)
        .agg({"speed": "mean", "vehicle": "count", "stopped": "sum"})
        .reset_index()
        .rename(columns={"vehicle": "density", "stopped": "stopped_count"})
    )
    agg_df["scenario"] = (
        scenario or os.path.basename(fcd_file).split("_")[1].split(".")[0]
    )
    return agg_df


# ---------------------------
# Multiprocessing Helpers
# ---------------------------


def generate_scenario(
    scenario_params, network_file, trips_prefix, config_file, fcd_prefix
):
    """Generate a scenario's trips and FCD."""
    scenario, params = scenario_params
    print(
        f"\nGenerating {scenario}: period={params['period']}, begin={params['begin']}, "
        f"end={params['end']}"
    )
    trips_file, route_file = generate_random_trips(
        scenario,
        network_file,
        trips_prefix,
        begin=params["begin"],
        end=params["end"],
        period=params["period"],
    )
    return generate_fcd(route_file, config_file, fcd_prefix) if route_file else None


def process_fcd_with_scenario(fcd_file):
    """Process FCD file with scenario label."""
    scenario_label = os.path.basename(fcd_file).split("_")[1].split(".")[0]
    return process_fcd(fcd_file, scenario=scenario_label)


# ---------------------------
# Model Training and Evaluation
# ---------------------------


def train_and_evaluate_split(train_df, test_df):
    """Train and evaluate multiple models on the data."""
    features = [
        "density",
        "time",
        "speed",
        "stopped_count",
        "lane_avg_speed",
        "lane_avg_density",
        "lane_avg_stopped_count",
    ]
    X_train = train_df[features].astype(np.float32)
    y_train = train_df["congestion"]
    X_test = test_df[features].astype(np.float32)
    y_test = test_df["congestion"]

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        probs = model.predict_proba(X_test)[:, 1]  # Probability of congestion
        print(f"{name} - Test Accuracy: {accuracy:.2f}")
        results[name] = {"model": model, "accuracy": accuracy, "probs": probs}

    # Plot for RandomForest (as original)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        test_df["density"],
        test_df["speed"],
        c=test_df["congestion"],
        cmap="coolwarm",
        alpha=0.5,
    )
    plt.colorbar(label="Congestion (0: No, 1: Yes)")
    plt.xlabel("Density (vehicles)")
    plt.ylabel("Mean Speed (m/s)")
    plt.title("Density vs Speed with Congestion (Test Data)")
    plt.savefig("scatter_test.png")
    plt.close()

    return results


# ---------------------------
# Main Pipeline
# ---------------------------


def main_pipeline():
    """Run the full pipeline for traffic simulation and congestion prediction."""
    base_path = "/Users/gilsilva/Work/thesis"
    network_file = f"{base_path}/output/test/osm.net.xml"
    trips_prefix = f"{base_path}/output/trips/trips"
    config_file = f"{base_path}/output/test/osm.sumocfg"
    fcd_prefix = f"{base_path}/output/fcd/fcd"

    # Clear output directories
    clear_output_folder(f"{base_path}/output/trips")
    clear_output_folder(f"{base_path}/output/fcd")

    # Define scenarios
    scenario_settings = {
        "scenario1": {"period": 1, "begin": 0, "end": 7200},  # Training
        "scenario2": {"period": 2, "begin": 0, "end": 3600},  # Testing
        "scenario3": {"period": 0.9, "begin": 0, "end": 3600},  # Testing
    }

    # Generate scenarios in parallel
    num_cores = min(cpu_count(), len(scenario_settings))
    with Pool(num_cores) as pool:
        fcd_files = pool.map(
            partial(
                generate_scenario,
                network_file=network_file,
                trips_prefix=trips_prefix,
                config_file=config_file,
                fcd_prefix=fcd_prefix,
            ),
            scenario_settings.items(),
        )
    fcd_files = [f for f in fcd_files if f]

    if not fcd_files:
        print("No FCD files generated.")
        return None, None

    # Process FCD files in parallel
    with Pool(num_cores) as pool:
        dfs = pool.map(process_fcd_with_scenario, fcd_files)
    dfs = [df for df in dfs if df is not None]

    if not dfs:
        print("No FCD processed successfully.")
        return None, None

    # Combine and process data
    df_combined = pd.concat(dfs, ignore_index=True, copy=False)
    print(f"\nCombined data from {len(dfs)} scenarios:")
    print(df_combined.head())

    # Calculate lane-specific averages
    lane_averages = (
        df_combined.groupby("lane", observed=True)
        .agg({"speed": "mean", "density": "mean", "stopped_count": "mean"})
        .rename(
            columns={
                "speed": "lane_avg_speed",
                "density": "lane_avg_density",
                "stopped_count": "lane_avg_stopped_count",
            }
        )
    )
    df_combined = df_combined.merge(lane_averages, on="lane", how="left")

    # Define congestion
    df_combined["congestion"] = (
        (df_combined["speed"] < df_combined["lane_avg_speed"] * 0.8)
        & (df_combined["density"] > df_combined["lane_avg_density"] * 1.2)
        & (
            (df_combined["lane_avg_stopped_count"] > 0)
            & (
                df_combined["stopped_count"]
                > df_combined["lane_avg_stopped_count"] * 1.5
            )
            | (df_combined["lane_avg_stopped_count"] == 0)
            & (df_combined["stopped_count"] > 0)
        )
    ).astype(np.int8)

    # Print congestion stats
    print("\nCongestion distribution:")
    print(df_combined["congestion"].value_counts(normalize=True))

    for state, label in [(1, "congested"), (0, "not congested")]:
        subset = df_combined[df_combined["congestion"] == state]
        print(f"\nMetrics when {label}:")
        print(f"Mean speed: {subset['speed'].mean():.2f}")
        print(f"Mean density: {subset['density'].mean():.2f}")
        print(f"Mean stopped_count: {subset['stopped_count'].mean():.2f}")

    # Train/test split
    train_df = df_combined[df_combined["scenario"] == "scenario1"]
    test_df = df_combined[df_combined["scenario"].isin(["scenario2", "scenario3"])]
    print(f"\nTraining scenario: {train_df['scenario'].unique()}")
    print(f"Testing scenarios: {test_df['scenario'].unique()}")

    # Train and evaluate models
    results = train_and_evaluate_split(train_df, test_df)
    return results, df_combined


if __name__ == "__main__":
    results, df = main_pipeline()
    if results and df is not None:
        print("\nPipeline completed successfully!")

import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# ---------------------------
# Utility Functions (Mantidas iguais, exceto por mais variabilidade)
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
    fringe_factor=1,
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
        "--fringe-factor",
        str(fringe_factor),
        "--random",
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
    # Adicionar variação de velocidade
    agg_df["speed_diff"] = agg_df.groupby("lane")["speed"].diff().fillna(0)
    return agg_df


# ---------------------------
# Multiprocessing Helpers (Mantidos iguais)
# ---------------------------


def generate_scenario(
    scenario_params, network_file, trips_prefix, config_file, fcd_prefix
):
    """Generate a scenario's trips and FCD."""
    scenario, params = scenario_params
    print(
        f"\nGenerating {scenario}: period={params['period']}, begin={params['begin']}, "
        f"end={params['end']}, fringe_factor={params['fringe_factor']}"
    )
    trips_file, route_file = generate_random_trips(
        scenario,
        network_file,
        trips_prefix,
        begin=params["begin"],
        end=params["end"],
        period=params["period"],
        fringe_factor=params["fringe_factor"],
    )
    return generate_fcd(route_file, config_file, fcd_prefix) if route_file else None


def process_fcd_with_scenario(fcd_file):
    """Process FCD file with scenario label."""
    scenario_label = os.path.basename(fcd_file).split("_")[1].split(".")[0]
    return process_fcd(fcd_file, scenario=scenario_label)


# ---------------------------
# Model Training and Evaluation (Revisado)
# ---------------------------


def define_congestion_kmeans(df):
    """Use K-means para definir congestionamento de forma não supervisionada."""
    features_for_kmeans = ["speed", "density", "stopped_count", "speed_diff"]
    X_kmeans = df[features_for_kmeans].astype(np.float32)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df["congestion"] = kmeans.fit_predict(X_kmeans)

    # Assumir que o cluster com menor velocidade média é congestionado
    cluster_speed_means = df.groupby("congestion")["speed"].mean()
    if cluster_speed_means[0] > cluster_speed_means[1]:
        df["congestion"] = 1 - df["congestion"]  # Inverter se necessário

    return df


def train_and_evaluate_split(train_df, test_df):
    """Train and evaluate multiple models on the data."""
    # Features revisadas (sem médias por faixa para evitar vazamento)
    features = ["density", "time", "speed", "stopped_count", "speed_diff"]
    X_train = train_df[features].astype(np.float32)
    y_train = train_df["congestion"]
    X_test = test_df[features].astype(np.float32)
    y_test = test_df["congestion"]

    # Modelos com regularização
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            penalty="l2", C=0.1, random_state=42, max_iter=1000
        ),
    }

    results = {}
    for name, model in models.items():
        # Validação cruzada no treino
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} - CV Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

        # Treino e teste
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Test Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(
            cm, display_labels=["No Congestion", "Congestion"]
        ).plot()
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(f"{name}_confusion_matrix.png")
        plt.close()

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "probs": model.predict_proba(X_test)[:, 1],
        }

    # Gráfico de dispersão
    plt.figure(figsize=(10, 6))
    plt.scatter(
        test_df["density"],
        test_df["speed"],
        c=y_test,
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
# Main Pipeline (Revisado)
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

    # Cenários mais variados
    scenario_settings = {
        "scenario1": {
            "period": 1,
            "begin": 0,
            "end": 3600,
            "fringe_factor": 1,
        },  # Normal
        "scenario2": {
            "period": 0.5,
            "begin": 0,
            "end": 3600,
            "fringe_factor": 5,
        },  # Pico
        "scenario3": {
            "period": 2,
            "begin": 0,
            "end": 3600,
            "fringe_factor": 10,
        },  # Baixa demanda
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

    # Combine data
    df_combined = pd.concat(dfs, ignore_index=True, copy=False)
    print(f"\nCombined data from {len(dfs)} scenarios:")
    print(df_combined.head())

    # Definir congestionamento com K-means
    df_combined = define_congestion_kmeans(df_combined)

    # Estatísticas de congestionamento
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

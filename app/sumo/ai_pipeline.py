import os
import logging
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from lxml import etree
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import json
from torch.amp import GradScaler, autocast
import argparse
from functools import partial
from bayes_opt import BayesianOptimization
from datetime import datetime

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


def setup_logging(base_path, date_folder):
    output_dir = os.path.join(base_path, "output", date_folder, "PIPELINE")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "ai_pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_array_module(logger):
    module = cp if GPU_AVAILABLE and torch.cuda.is_available() else np
    logger.info(f"Array module selected: {'CuPy' if module is cp else 'NumPy'}")
    return module


def process_fcd(
    fcd_file, scenario=None, logger=None, window_size=120, reference_metrics=None
):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"Starting process_fcd for file: {fcd_file}")
    start_time = time.time()
    fcd_file = os.path.abspath(fcd_file)
    if not os.path.exists(fcd_file):
        logger.error(f"FCD file does not exist: {fcd_file}")
        return None

    xp = get_array_module(logger)
    data = []
    try:
        parser = etree.iterparse(
            fcd_file, events=("end",), tag="timestep", huge_tree=True
        )
        for event, elem in parser:
            timestep = float(elem.get("time"))
            for vehicle in elem.findall("vehicle"):
                speed = float(vehicle.get("speed"))
                vehicle_id = vehicle.get("id")
                lane = vehicle.get("lane")
                data.append((timestep, speed, lane, vehicle_id))
            elem.clear()
        logger.info(
            f"Parsed {len(data)} records in {time.time() - start_time:.2f} seconds"
        )
    except Exception as e:
        logger.error(f"Error parsing {fcd_file}: {str(e)}")
        return None

    if not data:
        logger.warning(f"No vehicle data found in {fcd_file}")
        return None

    times = xp.array([d[0] for d in data], dtype=xp.float64)
    speeds = xp.array([d[1] for d in data], dtype=xp.float32)
    lanes = np.array([d[2] for d in data], dtype="U50")
    vehicle_ids = np.array([d[3] for d in data], dtype="U50")

    df = pd.DataFrame(
        {
            "time": times.get() if xp is cp else times,
            "speed": speeds.get() if xp is cp else speeds,
            "lane": lanes,
            "vehicle_id": vehicle_ids,
        }
    )
    df["time_window"] = (df["time"] // window_size) * window_size
    df["stopped"] = (df["speed"] < 0.1).astype(int)

    stopped_per_vehicle = (
        df.groupby(["lane", "time_window", "vehicle_id"])["stopped"].sum().reset_index()
    )
    stopped_per_vehicle = stopped_per_vehicle.rename(
        columns={"stopped": "vehicle_stopped_time"}
    )

    agg_df = (
        stopped_per_vehicle.groupby(["lane", "time_window"])
        .agg({"vehicle_stopped_time": "sum", "vehicle_id": "count"})
        .rename(columns={"vehicle_id": "vehicle_count"})
        .reset_index()
    )
    agg_df["time_stopped"] = agg_df["vehicle_stopped_time"] / agg_df["vehicle_count"]

    speed_density = (
        df.groupby(["lane", "time_window"])
        .agg({"speed": "mean", "vehicle_id": "size"})
        .rename(columns={"vehicle_id": "density"})
        .reset_index()
    )
    agg_df = agg_df.merge(speed_density, on=["lane", "time_window"], how="left")

    agg_df["hour"] = (agg_df["time_window"] // 3600) % 24
    agg_df["scenario"] = (
        scenario or os.path.basename(fcd_file).split("_")[1].split(".")[0]
    )

    if reference_metrics is not None:
        agg_df = agg_df.merge(
            reference_metrics[["speed_historical", "density_threshold"]],
            on="lane",
            how="left",
        ).fillna({"speed_historical": 0, "density_threshold": 0})

    logger.info(
        f"process_fcd completed in {time.time() - start_time:.2f} seconds with {len(agg_df)} windows"
    )
    return agg_df


def process_fcd_with_scenario(fcd_file, logger, reference_metrics=None):
    if not fcd_file.endswith(".xml") or "additional" in fcd_file:
        logger.info(f"Skipping file: {fcd_file}")
        return None
    scenario = os.path.basename(fcd_file).split("_")[1].split(".")[0]
    return process_fcd(
        fcd_file, scenario=scenario, logger=logger, reference_metrics=reference_metrics
    )


def calculate_reference_metrics(df_dataset1, logger):
    logger.info("Calculating reference metrics from training scenarios")
    reference_metrics = (
        df_dataset1.groupby("lane")
        .agg({"speed": "mean", "density": lambda x: np.percentile(x, 75)})
        .rename(columns={"speed": "speed_historical", "density": "density_threshold"})
    )
    logger.info(f"Reference metrics: {reference_metrics.describe()}")
    return reference_metrics


def label_congestion(df, reference_metrics, logger):
    logger.info("Labeling congestion")
    if "speed_historical" not in df.columns or "density_threshold" not in df.columns:
        df = df.merge(
            reference_metrics[["speed_historical", "density_threshold"]],
            on="lane",
            how="left",
        ).fillna({"speed_historical": 0, "density_threshold": 0})
    df["congestion"] = (
        (df["speed"] < 0.5 * df["speed_historical"])
        & (df["density"] > df["density_threshold"])
    ).astype(int)
    congestion_ratio = df["congestion"].mean()
    logger.info(f"Congestion ratio: {congestion_ratio:.2%}, Total samples: {len(df)}")
    return df


class SimpleCNNLSTM(nn.Module):
    def __init__(self, input_size, sequence_length=5, dropout=0.2):
        super().__init__()
        self.sequence_length = sequence_length
        self.cnn = nn.Conv1d(
            in_channels=input_size, out_channels=16, kernel_size=3, padding=1
        )
        if sequence_length > 1:
            self.lstm = nn.LSTM(
                input_size=16, hidden_size=32, num_layers=2, batch_first=True
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32 if sequence_length > 1 else 16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        if self.sequence_length > 1:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = self.dropout(x[:, -1, :])
        else:
            x = x.squeeze(2)
            x = self.dropout(x)
        return self.fc(x)


def prepare_sequences(df, features, sequence_length, lanes, logger):
    logger.info("Preparing sequences")
    xp = get_array_module(logger)
    X, y, times, lane_labels = [], [], [], []

    for lane in lanes:
        lane_df = df[df["lane"] == lane][
            ["time_window"] + features + ["congestion"]
        ].sort_values("time_window")
        if lane_df.empty:
            logger.debug(f"Lane {lane} has no data. Skipping.")
            continue

        available_features = [f for f in features if f in lane_df.columns]
        logger.debug(f"Lane {lane} available features: {available_features}")

        for feature in available_features:
            mean = lane_df[feature].mean()
            std = lane_df[feature].std()
            if std == 0 or pd.isna(std):
                logger.warning(f"Lane {lane} has zero std for {feature}. Setting to 0.")
                lane_df[feature] = 0
            else:
                lane_df[feature] = (lane_df[feature] - mean) / std

        if (
            lane_df[available_features].isna().any().any()
            or np.isinf(lane_df[available_features]).any().any()
        ):
            logger.warning(
                f"Lane {lane} contains NaN or Inf after normalization. Skipping."
            )
            continue

        data = xp.asarray(lane_df[available_features].to_numpy())
        labels = xp.asarray(lane_df["congestion"].to_numpy())
        lane_times = xp.asarray(lane_df["time_window"].to_numpy())

        if len(lane_times) == 0:
            logger.debug(f"Lane {lane} has no time windows after processing. Skipping.")
            continue

        n_samples = len(data)
        if n_samples < sequence_length:
            padding = sequence_length - n_samples
            data = xp.pad(
                data, ((padding, 0), (0, 0)), mode="constant", constant_values=0
            )
            labels = xp.pad(labels, (padding, 0), mode="constant", constant_values=0)
            lane_times = xp.pad(
                lane_times,
                (padding, 0),
                mode="constant",
                constant_values=lane_times[0].item(),
            )
            n = 1
        else:
            n = n_samples - sequence_length + 1

        if n > 0:
            idx = xp.arange(sequence_length) + xp.arange(n)[:, None]
            X.append(data[idx])
            y.append(labels[sequence_length - 1 : n + sequence_length - 1])
            times.extend(lane_times[sequence_length - 1 : n + sequence_length - 1])
            lane_labels.extend([lane] * n)

    if X:
        X = xp.concatenate(X).astype(xp.float32)
        y = xp.concatenate(y).astype(xp.float32)
        X = X.get() if xp is cp else X
        y = y.get() if xp is cp else y
        if (
            np.isnan(X).any()
            or np.isinf(X).any()
            or np.isnan(y).any()
            or np.isinf(y).any()
        ):
            logger.error("NaN or Inf found in X or y. Aborting sequence preparation.")
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                [],
                [],
            )
        logger.info(
            f"Sequences prepared: X shape {X.shape}, y shape {y.shape}, Congestion ratio {y.mean():.2%}"
        )
    else:
        X, y = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        times, lane_labels = [], []
        logger.warning("No sequences generated")
    return X, y, times, lane_labels


def calculate_metrics(y_true, y_pred, y_prob, logger):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        logger.warning("Only one class present in y_true or y_pred. Metrics will be 0.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    logger.info(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
    )
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc}


def train_and_evaluate(
    train_df,
    test_dfs,
    logger,
    base_path,
    lr=0.001,
    batch_size=512,
    epochs=10,
    dropout=0.2,
    pos_weight=1.0,
    date_folder=None,
    save_model=False,
):
    logger.info("Starting train_and_evaluate")
    start_time = time.time()
    features = ["speed", "density", "time_stopped", "hour", "speed_historical"]
    sequence_length = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    all_lanes = train_df["lane"].unique().tolist()
    logger.info(f"Training with {len(all_lanes)} lanes")

    X_train, y_train, train_times, train_lanes = prepare_sequences(
        train_df, features, sequence_length, all_lanes, logger
    )
    test_data = [
        prepare_sequences(test_df, features, sequence_length, all_lanes, logger)
        for test_df in test_dfs
    ]

    if X_train.size == 0 or y_train.size == 0:
        logger.error("No training sequences generated.")
        return None

    model = SimpleCNNLSTM(
        input_size=len(features), sequence_length=sequence_length, dropout=dropout
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=min(cpu_count(), 8),
        pin_memory=True,
    )

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        logger.info(
            f"Epoch {epoch + 1}/{int(epochs)}, Loss: {total_loss / len(train_loader):.4f}"
        )

    metrics_dict = {}
    model.eval()
    for i, (X_test, y_test, test_times, test_lanes) in enumerate(test_data):
        scenario = f"scenario{i + 4}"
        if X_test.size == 0 or y_test.size == 0:
            logger.warning(f"No test sequences for {scenario}.")
            continue
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=min(cpu_count(), 8),
            pin_memory=True,
        )
        probs, true_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                with autocast("cuda"):
                    outputs = model(X_batch)
                probs.extend(torch.sigmoid(outputs.squeeze()).cpu().numpy())
                true_labels.extend(y_batch.numpy())
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [
            f1_score(true_labels, [1 if p > t else 0 for p in probs], zero_division=0)
            for t in thresholds
        ]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"{scenario} - Optimal threshold (max F1): {optimal_threshold:.4f}")
        y_pred = [1 if p > optimal_threshold else 0 for p in probs]
        metrics = calculate_metrics(true_labels, y_pred, probs, logger)
        metrics_dict[scenario] = metrics

    if save_model:
        model_file = os.path.join(
            base_path, "output", date_folder, "PIPELINE", "models", "best_cnn_lstm.pth"
        )
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(model.state_dict(), model_file)
        logger.info(f"Best model saved to {model_file}")

    logger.info(
        f"train_and_evaluate completed in {time.time() - start_time:.2f} seconds"
    )
    return {"CNN-LSTM": {"model": model, "metrics": metrics_dict}}


def optimize_hyperparameters(train_df, test_dfs, logger, base_path, date_folder):
    def evaluate_model(lr, batch_size, dropout, pos_weight, epochs):
        results = train_and_evaluate(
            train_df,
            test_dfs,
            logger,
            base_path,
            lr=lr,
            batch_size=batch_size,
            dropout=dropout,
            pos_weight=pos_weight,
            epochs=epochs,
            date_folder=date_folder,
            save_model=False,
        )
        if results is None:
            return 0.0
        avg_f1 = (
            sum(
                results["CNN-LSTM"]["metrics"][s]["f1"]
                for s in results["CNN-LSTM"]["metrics"]
            )
            / 3
        )
        return avg_f1

    pbounds = {
        "lr": (0.0001, 0.01),
        "batch_size": (128, 1024),
        "dropout": (0.1, 0.5),
        "pos_weight": (1.0, 10.0),
        "epochs": (5, 20),
    }
    optimizer = BayesianOptimization(f=evaluate_model, pbounds=pbounds, random_state=42)
    optimization_history = []
    logger.info("Starting Bayesian Optimization")
    optimizer.maximize(init_points=5, n_iter=15)

    for i, res in enumerate(optimizer.res):
        params = res["params"]
        target = res["target"]
        optimization_history.append(
            {
                "lr": params["lr"],
                "batch_size": params["batch_size"],
                "dropout": params["dropout"],
                "pos_weight": params["pos_weight"],
                "epochs": int(params["epochs"]),
                "avg_f1_score": target,
            }
        )
        logger.info(f"Tentativa {i+1}: {params}, Avg F1-score: {target}")

    history_file = os.path.join(
        base_path,
        "output",
        date_folder,
        "PIPELINE",
        "json",
        "optimization_history.json",
    )
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w") as f:
        json.dump(optimization_history, f, indent=4)
    logger.info(f"Histórico de otimização salvo em {history_file}")

    best_params = optimizer.max["params"]
    best_params["epochs"] = int(best_params["epochs"])
    logger.info(f"Best Avg F1: {optimizer.max['target']}, Best Params: {best_params}")

    logger.info("Running final evaluation with best parameters")
    final_results = train_and_evaluate(
        train_df,
        test_dfs,
        logger,
        base_path,
        **best_params,
        date_folder=date_folder,
        save_model=True,
    )

    final_output = {
        "best_hyperparameters": best_params,
        "metrics": final_results["CNN-LSTM"]["metrics"] if final_results else {},
        "avg_f1_score": optimizer.max["target"],
    }
    final_results_file = os.path.join(
        base_path,
        "output",
        date_folder,
        "PIPELINE",
        "json",
        "final_optimization_results.json",
    )
    os.makedirs(os.path.dirname(final_results_file), exist_ok=True)
    with open(final_results_file, "w") as f:
        json.dump(final_output, f, indent=4)
    logger.info(f"Resultados finais salvos em {final_results_file}")

    return best_params


def check_imbalance(df_combined, logger):
    logger.info("Verificando desbalanceamento nos cenários")
    for scenario in [f"scenario{i}" for i in range(1, 7)]:
        scenario_df = df_combined[df_combined["scenario"] == scenario]
        if not scenario_df.empty:
            congestion_ratio = scenario_df["congestion"].mean()
            total_samples = len(scenario_df)
            logger.info(
                f"{scenario}: Congestion ratio: {congestion_ratio:.2%}, Samples: {total_samples}"
            )


def ai_pipeline(args, logger, date_folder="default"):
    logger.info("Starting ai_pipeline")
    start_time = time.time()
    base_path = os.path.abspath(args.base_path)
    # Ajusta o caminho do fcd_dir conforme o date_folder
    fcd_dir = os.path.join(
        base_path,
        "output",
        date_folder,
        "SUMO" if date_folder != "default" else "",
        "fcd",
    ).rstrip(os.sep)
    os.makedirs(fcd_dir, exist_ok=True)
    fcd_files = [
        os.path.join(fcd_dir, f)
        for f in os.listdir(fcd_dir)
        if f.endswith(".xml") and "additional" not in f
    ]
    logger.info(f"Found {len(fcd_files)} FCD files in {fcd_dir}")

    with Pool(cpu_count()) as pool:
        dfs = pool.map(partial(process_fcd_with_scenario, logger=logger), fcd_files)
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        logger.error("No valid FCD files processed. Aborting.")
        return None, None
    df_combined = pd.concat(dfs, ignore_index=True)

    train_df = pd.concat(
        [
            df_combined[df_combined["scenario"] == "scenario1"],
            df_combined[df_combined["scenario"] == "scenario2"],
            df_combined[df_combined["scenario"] == "scenario3"],
        ]
    )
    test_dfs = [
        df_combined[df_combined["scenario"] == f"scenario{i}"] for i in range(4, 7)
    ]
    reference_metrics = calculate_reference_metrics(train_df, logger)

    all_lanes = df_combined["lane"].unique()
    missing_lanes = set(all_lanes) - set(reference_metrics.index)
    if missing_lanes:
        logger.warning(
            f"Adding missing lanes to reference_metrics: {len(missing_lanes)} lanes"
        )
        missing_df = pd.DataFrame(
            {"speed_historical": 0, "density_threshold": 0},
            index=pd.Index(list(missing_lanes), name="lane"),
        )
        reference_metrics = pd.concat([reference_metrics, missing_df])

    with Pool(cpu_count()) as pool:
        process_func = partial(
            process_fcd_with_scenario,
            logger=logger,
            reference_metrics=reference_metrics,
        )
        dfs = pool.map(process_func, fcd_files)
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        logger.error("No valid FCD files processed with reference metrics. Aborting.")
        return None, None
    df_combined = pd.concat(dfs, ignore_index=True)

    train_df = pd.concat(
        [
            df_combined[df_combined["scenario"] == "scenario1"],
            df_combined[df_combined["scenario"] == "scenario2"],
            df_combined[df_combined["scenario"] == "scenario3"],
        ]
    )
    test_dfs = [
        df_combined[df_combined["scenario"] == f"scenario{i}"] for i in range(4, 7)
    ]

    train_df = label_congestion(train_df, reference_metrics, logger)
    test_dfs = [
        label_congestion(test_df, reference_metrics, logger) for test_df in test_dfs
    ]

    df_combined = pd.concat([train_df] + test_dfs, ignore_index=True)
    check_imbalance(df_combined, logger)

    if args.optimize:
        logger.info("Using Bayesian Optimization to find best hyperparameters")
        best_params = optimize_hyperparameters(
            train_df, test_dfs, logger, base_path, date_folder
        )
    else:
        logger.info("Using default hyperparameters")
        best_params = {
            "lr": 0.001,
            "batch_size": 512,
            "dropout": 0.2,
            "pos_weight": 1.0,
            "epochs": 10,
        }
        logger.info(f"Default parameters: {best_params}")
        results = train_and_evaluate(
            train_df,
            test_dfs,
            logger,
            base_path,
            **best_params,
            date_folder=date_folder,
            save_model=True,
        )

    train_df_file = os.path.join(
        base_path, "output", date_folder, "PIPELINE", "csv", "train_df.csv"
    )
    os.makedirs(os.path.dirname(train_df_file), exist_ok=True)
    train_df.to_csv(train_df_file, index=False)
    logger.info(f"Train DataFrame saved to {train_df_file}")

    logger.info(f"ai_pipeline completed in {time.time() - start_time:.2f} seconds")
    return results, df_combined


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="AI Pipeline com CuPy")
    parser.add_argument("--base-path", default=os.getcwd(), help="Caminho base")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Usar Otimização Bayesiana para hiperparâmetros (padrão: False)",
    )
    args = parser.parse_args()
    date_folder = datetime.now().strftime("%Y-%m-%d")
    logger = setup_logging(args.base_path, date_folder)
    logger.info(f"Starting execution with args: {args}")
    results, df_combined = ai_pipeline(args, logger, date_folder="default")
    logger.info("Execution completed")

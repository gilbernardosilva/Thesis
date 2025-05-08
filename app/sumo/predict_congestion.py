import os
import logging
import time
import numpy as np
import pandas as pd
from lxml import etree
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import json
import argparse

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


# Configuração de logging com o nome do arquivo FCD
def setup_logging(base_path, fcd_file):
    output_dir = os.path.join(base_path, "output", "log")
    os.makedirs(output_dir, exist_ok=True)
    fcd_filename = os.path.basename(fcd_file).split(".")[0]
    log_file = os.path.join(output_dir, f"predict_congestion_{fcd_filename}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# Escolher entre CuPy e NumPy
def get_array_module(logger):
    module = cp if GPU_AVAILABLE and torch.cuda.is_available() else np
    logger.info(f"Array module selected: {'CuPy' if module is cp else 'NumPy'}")
    return module


# Processamento de arquivos FCD
def process_fcd(fcd_file, scenario=None, logger=None):
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
                data.append(
                    (timestep, float(vehicle.get("speed")), vehicle.get("lane"))
                )
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
    speeds = xp.array([d[1] for d in data], dtype=xp.float64)
    lanes = np.array([d[2] for d in data], dtype="U50")

    df = pd.DataFrame(
        {
            "time": times.get() if xp is cp else times,
            "speed": speeds.get() if xp is cp else speeds,
            "lane": lanes,
        }
    )
    agg_df = (
        df.groupby(["time", "lane"])
        .agg({"speed": "mean", "lane": "size"})
        .rename(columns={"lane": "density"})
        .reset_index()
    )
    agg_df["scenario"] = scenario or os.path.basename(fcd_file).split(".")[0]
    logger.info(f"process_fcd completed in {time.time() - start_time:.2f} seconds")
    return agg_df


# Calcular métricas de referência
def calculate_reference_metrics(df_dataset1, logger):
    logger.info("Calculating reference metrics")
    reference_metrics = (
        df_dataset1.groupby("lane")
        .agg({"speed": "mean", "density": "mean"})
        .rename(columns={"speed": "mean_speed", "density": "mean_density"})
    )
    reference_metrics["speed_threshold"] = reference_metrics["mean_speed"] * 0.9
    reference_metrics["density_threshold"] = reference_metrics["mean_density"] * 1.1
    logger.info(f"Reference metrics: {reference_metrics.describe()}")
    return reference_metrics


# Rotular congestionamento
def label_congestion(df, reference_metrics, logger):
    logger.info("Labeling congestion")
    df = df.merge(
        reference_metrics[["speed_threshold", "density_threshold"]],
        on="lane",
        how="left",
    )
    df["congestion"] = (
        (df["speed"] < df["speed_threshold"])
        & (df["density"] > df["density_threshold"])
    ).astype(int)
    logger.info(
        f"Congestion ratio: {df['congestion'].mean():.2%}, Total samples: {len(df)}"
    )
    return df


# Modelo CNN-LSTM
class SimpleCNNLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=input_size, out_channels=8, kernel_size=3, padding=1
        )
        self.lstm = nn.LSTM(
            input_size=8, hidden_size=16, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


# Preparação de sequências
def prepare_sequences(df, features, sequence_length, selected_lanes, logger):
    logger.info("Preparing sequences")
    xp = get_array_module(logger)
    X, times, lane_labels = [], [], []
    for lane in selected_lanes:
        lane_df = df[df["lane"] == lane][["time"] + features].sort_values("time")
        logger.info(f"Lane {lane}: {len(lane_df)} samples")
        data = xp.asarray(lane_df[features].to_numpy())
        lane_times = lane_df["time"].to_numpy()
        n = len(data) - sequence_length + 1
        if n > 0:
            idx = xp.arange(sequence_length) + xp.arange(n)[:, None]
            X.append(data[idx])
            times.extend(lane_times[sequence_length - 1 : n + sequence_length - 1])
            lane_labels.extend([lane] * n)
        else:
            logger.warning(
                f"Lane {lane} has insufficient data for sequence_length {sequence_length}"
            )
    if X:
        X = xp.concatenate(X).astype(xp.float32)
        logger.info(f"Sequences prepared: X shape {X.shape}")
    else:
        X = xp.array([], dtype=xp.float32)
        times, lane_labels = [], []
        logger.warning("No sequences generated")
    return X, times, lane_labels


# Função para prever probabilidades
def predict_congestion(fcd_file, base_path, logger):
    # Carregar configuração do modelo
    config_path = os.path.join(base_path, "output", "model", "model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Processar o novo arquivo FCD
    df = process_fcd(fcd_file, logger=logger)
    if df is None:
        logger.error("Failed to process FCD file.")
        return None

    # Carregar train_df para reference_metrics consistentes
    train_df_path = os.path.join(base_path, "output", "train_df.csv")
    if not os.path.exists(train_df_path):
        logger.error(
            f"Train DataFrame not found at {train_df_path}. Run ai_pipeline.py first."
        )
        return None
    train_df = pd.read_csv(train_df_path)
    reference_metrics = calculate_reference_metrics(train_df, logger)
    df = label_congestion(df, reference_metrics, logger)

    # Preparar sequências
    features = ["speed", "density"]
    sequence_length = config["sequence_length"]
    selected_lanes = config["selected_lanes"]
    X, times, lane_labels = prepare_sequences(
        df, features, sequence_length, selected_lanes, logger
    )

    if X.size == 0:
        logger.error("No sequences generated for prediction.")
        return None

    # Converter para PyTorch
    xp = get_array_module(logger)
    X = X.get() if xp is cp else X
    test_dataset = TensorDataset(torch.from_numpy(X))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Carregar o modelo treinado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNLSTM(input_size=len(features)).to(device)
    model.load_state_dict(
        torch.load(config["model_path"], map_location=device, weights_only=True)
    )  # Adicionado weights_only=True
    model.eval()

    # Fazer previsões
    probs = []
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(X_batch)
            probs.extend(torch.sigmoid(outputs.squeeze()).cpu().numpy())

    # Criar DataFrame com resultados
    results_df = pd.DataFrame(
        {
            "lane": lane_labels,
            "time": times,
            "congestion_probability": [p * 100 for p in probs],
        }
    )

    # Salvar resultados
    output_dir = os.path.join(base_path, "output")
    output_file = os.path.join(
        output_dir, f"predictions_{os.path.basename(fcd_file).split('.')[0]}.csv"
    )
    results_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    return results_df


# Função principal
def main(args):
    logger = setup_logging(args.base_path, args.fcd_file)
    logger.info(f"Starting prediction with args: {args}")
    results_df = predict_congestion(args.fcd_file, args.base_path, logger)
    if results_df is not None:
        logger.info("Prediction completed successfully")
        print("First few predictions:")
        # Limitar precisão para evitar overflow no Pandas
        with pd.option_context("display.float_format", "{:.2f}".format):
            print(results_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict congestion with trained model"
    )
    parser.add_argument(
        "--base-path", default=os.getcwd(), help="Base path for model and data"
    )
    parser.add_argument(
        "--fcd-file", required=True, help="Path to the FCD file for prediction"
    )
    args = parser.parse_args()
    main(args)

import logging

import numpy as np
import pandas as pd

from app.fcd.config.variables import (
    FEATURES_DATA_FILE,
    SUMO_ACCIDENTS_FILE,
    SUMO_ACCIDENTS_PREDICTION_FILE,
    SUMO_MATCHED_FILE,
)
from app.fcd.model import CongestionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sumo/output/congestion_eval.log", mode="w"),
        logging.StreamHandler(),
    ],
)


def calculate_features_for_accident_window(fcd_df, accident, edge_id):
    edge_id = int(edge_id)

    if "edge_id" not in fcd_df.columns:
        logging.error(
            "Column 'edge_id' not found in FCD DataFrame. Ensure map matching was performed."
        )
        return None

    start_time = accident["start_time"]
    end_time = accident["end_time"]
    window_fcd = fcd_df[
        (fcd_df["timestamp"] >= start_time)
        & (fcd_df["timestamp"] <= end_time)
        & (fcd_df["edge_id"] == edge_id)
    ].copy()

    if window_fcd.empty:
        logging.info(
            f"No FCD data for accident on edge {edge_id} from {start_time} to {end_time}"
        )
        return None

    window_fcd.loc[:, "day_of_week"] = window_fcd["timestamp"].dt.dayofweek
    window_fcd.loc[:, "hour"] = window_fcd["timestamp"].dt.hour
    window_fcd.loc[:, "is_peak_hour"] = window_fcd["hour"].apply(
        lambda h: 1 if h in [7, 8, 9, 16, 17, 18] else 0
    )
    window_fcd.loc[:, "is_weekend"] = window_fcd["day_of_week"].apply(
        lambda d: 1 if d >= 5 else 0
    )
    window_fcd.loc[:, "avg_speed_segment"] = window_fcd["probe_speed"].mean()

    speed_bins = pd.cut(
        window_fcd["probe_speed"], bins=5, labels=False, include_lowest=True
    )
    for i in range(5):
        window_fcd.loc[:, f"speed_bin_{i}"] = (speed_bins == i).astype(int)

    window_fcd.loc[:, "is_stopped"] = (window_fcd["probe_speed"] < 1).astype(int)
    stopped_time = window_fcd.groupby("traj_id")["is_stopped"].sum().reset_index()
    stopped_time.columns = ["traj_id", "total_stopped_time"]
    window_fcd = window_fcd.merge(stopped_time, on="traj_id", how="left")
    window_fcd.loc[:, "total_stopped_time_per_edge_per_traj"] = window_fcd[
        "total_stopped_time"
    ]

    return window_fcd


def evaluate_congestion(
    model, fcd_file: str, accidents_file: str, historical_features_file: str
):
    fcd_df = pd.read_parquet(fcd_file)
    accidents_df = pd.read_parquet(accidents_file)
    historical_df = pd.read_parquet(historical_features_file)

    logging.info(f"FCD DataFrame columns: {list(fcd_df.columns)}")

    predictions = []

    for idx, accident in accidents_df.iterrows():
        edge_id = accident["edge_id"]
        sumo_edge_id = accident["sumo_edge_id"]
        start_time = accident["start_time"]
        end_time = accident["end_time"]

        logging.info(
            f"Evaluating accident {idx} on edge {edge_id} from {start_time} to {end_time}"
        )

        historical_features = historical_df[historical_df["edge_id"] == edge_id]
        if historical_features.empty:
            logging.info(f"No historical features for edge {edge_id}")
            predictions.append(
                {
                    "accident_index": idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "edge_id": edge_id,
                    "sumo_edge_id": sumo_edge_id,
                    "predicted_congestion": "unknown",
                }
            )
            continue

        features_df = calculate_features_for_accident_window(fcd_df, accident, edge_id)
        if features_df is None:
            predictions.append(
                {
                    "accident_index": idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "edge_id": edge_id,
                    "sumo_edge_id": sumo_edge_id,
                    "predicted_congestion": "unknown",
                }
            )
            continue

        historical_agg = (
            historical_features.groupby("edge_id")
            .agg(
                {
                    "length_segment": "mean",
                    "road_class_encoded": "first",
                    "speed_segment": "mean",
                }
            )
            .reset_index()
        )

        features_df.loc[:, "length_segment"] = historical_agg["length_segment"].iloc[0]
        features_df.loc[:, "road_class_encoded"] = historical_agg[
            "road_class_encoded"
        ].iloc[0]

        probs = model.predict(features_df)
        predicted_class = np.argmax(probs, axis=1)
        predicted_congestion = pd.Series(
            [model.CLASSES[cls] for cls in predicted_class]
        ).mode()[0]
        logging.info(
            f"Accident {idx} on edge {edge_id}: Predicted congestion - {predicted_congestion}"
        )

        predictions.append(
            {
                "accident_index": idx,
                "start_time": start_time,
                "end_time": end_time,
                "edge_id": edge_id,
                "sumo_edge_id": sumo_edge_id,
                "predicted_congestion": predicted_congestion,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(SUMO_ACCIDENTS_PREDICTION_FILE, index=False)
    logging.info(f"Predictions saved to {SUMO_ACCIDENTS_PREDICTION_FILE}")


def test_model():
    model = CongestionModel(model_type="lightgbm")
    evaluate_congestion(
        model, SUMO_MATCHED_FILE, SUMO_ACCIDENTS_FILE, FEATURES_DATA_FILE
    )


if __name__ == "__main__":
    test_model()

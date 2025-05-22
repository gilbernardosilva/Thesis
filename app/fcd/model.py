import logging
import os
import pickle

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.fcd.config.variables import (
    FEATURES_DATA_FILE,
    LABEL_ENCODER_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    MODEL_PATH,
    PROJECT_BASE_PATH,
    SCALER_PARAMS_PATH,
)

LOG_FILE = os.path.join(os.path.dirname(MODEL_PATH), "congestion_model.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)


class CongestionModel:
    CLASSES = ["none", "light", "moderate", "severe"]

    def __init__(
        self,
        model_path=MODEL_PATH,
        scaler_params_path=SCALER_PARAMS_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
    ):
        self.model_path = model_path
        self.scaler_params_path = scaler_params_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.scaler_params = None
        self.label_encoder = None
        self.edge_id_encoder = None
        self._load_model_files()

    def _load_model_files(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        if os.path.exists(self.scaler_params_path):
            with open(self.scaler_params_path, "rb") as f:
                self.scaler_params = pickle.load(f)
        if os.path.exists(self.label_encoder_path):
            with open(self.label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        edge_id_encoder_path = os.path.join(
            os.path.dirname(self.model_path), "edge_id_encoder.pkl"
        )
        if os.path.exists(edge_id_encoder_path):
            with open(edge_id_encoder_path, "rb") as f:
                self.edge_id_encoder = pickle.load(f)
        if all(
            [self.model, self.scaler_params, self.label_encoder, self.edge_id_encoder]
        ):
            logging.info("Model and parameters loaded successfully.")
        else:
            logging.info("Model not loaded; train a new model with train_model.")

    def _fill_nan(self, df: pd.DataFrame, set_name: str) -> pd.DataFrame:
        """Handle NaN values in the DataFrame."""
        logging.info(f"Verificando valores NaN nas features de {set_name}...")
        nan_counts = df.isna().sum()
        logging.info(f"NaN por coluna ({set_name}):\n{nan_counts.to_string()}")
        if nan_counts.any():
            logging.warning(
                f"Valores NaN encontrados nas features de {set_name}. Preenchendo com 0..."
            )
            df = df.fillna(0)
        return df

    def _normalize_features(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple:
        """Normalize features using mean and standard deviation from training data."""
        self.scaler_params = {}
        x_train = x_train.copy()
        x_test = x_test.copy()
        for col in x_train.columns:
            if col != "edge_id":
                mean = x_train[col].mean()
                std = x_train[col].std()
                self.scaler_params[col] = {"mean": mean, "std": std}
                if std > 0:
                    x_train.loc[:, col] = (x_train[col] - mean) / std
                    x_test.loc[:, col] = (x_test[col] - mean) / std
                else:
                    x_train.loc[:, col] = 0
                    x_test.loc[:, col] = 0
                if x_train[col].isna().any() or x_test[col].isna().any():
                    logging.warning(
                        f"NaN encontrados em {col} após normalização. Preenchendo com 0..."
                    )
                    x_train.loc[:, col] = x_train[col].fillna(0)
                    x_test.loc[:, col] = x_test[col].fillna(0)
        return x_train, x_test

    def _evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate the model on the test set."""
        probabilities = self.model.predict_proba(x_test)
        pred_labels = self.label_encoder.inverse_transform(
            np.argmax(probabilities, axis=1)
        )
        accuracy = accuracy_score(y_test, pred_labels)
        f1 = f1_score(y_test, pred_labels, average="weighted", zero_division=0)
        precision = precision_score(
            y_test, pred_labels, average="weighted", zero_division=0
        )
        recall = recall_score(y_test, pred_labels, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(y_test, pred_labels, labels=self.CLASSES)
        report = classification_report(
            y_test, pred_labels, labels=self.CLASSES, zero_division=0
        )
        per_class_f1 = f1_score(y_test, pred_labels, average=None, zero_division=0)

        logging.info("Test set evaluation metrics:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score (weighted): {f1:.4f}")
        logging.info(f"Per-class F1 scores: {dict(zip(self.CLASSES, per_class_f1))}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"Classification Report:\n{report}")

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def train_model(self, data_path=FEATURES_DATA_FILE):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} was not found.")
        logging.info(f"Loading data from {data_path} for training...")
        ddf = dd.read_parquet(
            data_path,
            engine="pyarrow",
            columns=[
                "timestamp",
                "traj_id",
                "edge_id",
                "speed",
                "free_flow_speed",
                "speed_segment",
                "sri",
            ],
        )

        logging.info("Verificando NaN nos dados brutos...")
        nan_counts_raw = ddf.isna().sum().compute()
        logging.info(f"NaN nos dados brutos:\n{nan_counts_raw.to_string()}")

        self.edge_id_encoder = LabelEncoder()
        edge_ids = ddf["edge_id"].compute()
        self.edge_id_encoder.fit(edge_ids)
        logging.info(
            f"Encoded {len(self.edge_id_encoder.classes_)} unique edge_id values"
        )

        data = self._create_windows(ddf)
        if data.empty:
            raise ValueError("No valid windows created for training.")

        features = [
            "speed_mean",
            "free_flow_speed_mean",
            "speed_change",
            "hour",
            "day_of_week",
            "is_peak_hour",
            "is_weekend",
            "edge_id",
        ]
        x = data[features + ["traj_id"]].copy()
        y = data["label"]

        # Log feature correlations to diagnose overfitting
        correlation = x[features].corr()
        logging.info(f"Feature correlations:\n{correlation}")

        # Divisão 70-30 por traj_id
        unique_traj_ids = (
            x["traj_id"].unique().tolist()
        )  # Convert to list to avoid indexing issues
        traj_train, traj_test = train_test_split(
            unique_traj_ids, test_size=0.3, random_state=42
        )
        x_train = x[x["traj_id"].isin(traj_train)].drop(columns=["traj_id"]).copy()
        x_test = x[x["traj_id"].isin(traj_test)].drop(columns=["traj_id"]).copy()
        y_train = y[x["traj_id"].isin(traj_train)]
        y_test = y[x["traj_id"].isin(traj_test)]

        logging.info(
            f"Training set: {len(x_train)} samples, {len(traj_train)} trajectories"
        )
        logging.info(f"Test set: {len(x_test)} samples, {len(traj_test)} trajectories")

        # Handle NaN
        x_train = self._fill_nan(x_train, "treino")
        x_test = self._fill_nan(x_test, "teste")

        class_counts_train = y_train.value_counts()
        logging.info(f"Class counts in training set: {class_counts_train.to_dict()}")

        # Encode edge_id
        x_train.loc[:, "edge_id"] = self.edge_id_encoder.transform(
            x_train["edge_id"]
        ).astype(np.int64)
        x_test.loc[:, "edge_id"] = self.edge_id_encoder.transform(
            x_test["edge_id"]
        ).astype(np.int64)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Normalize features
        x_train, x_test = self._normalize_features(x_train, x_test)

        # Train model
        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=4,
            random_state=42,
            n_estimators=50,
            max_depth=3,
            class_weight="balanced",
        )
        self.model.fit(x_train, y_train_encoded, categorical_feature=["edge_id"])

        # Evaluate model
        self._evaluate(x_test, y_test)

        # Save model and parameters
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_params_path, "wb") as f:
            pickle.dump(self.scaler_params, f)
        with open(self.label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        edge_id_encoder_path = os.path.join(
            os.path.dirname(self.model_path), "edge_id_encoder.pkl"
        )
        with open(edge_id_encoder_path, "wb") as f:
            pickle.dump(self.edge_id_encoder, f)
        logging.info("Model and parameters saved successfully.")

    def _create_windows(self, df, window_size="15min", prediction_horizon="5min"):
        total_windows = 0
        empty_future_count = 0
        null_sri_count = 0
        is_dask = isinstance(df, dd.DataFrame)

        logging.info("Aggregating features...")
        if is_dask:
            df = df.assign(window_timestamp=df["timestamp"].dt.floor(window_size))
            agg_features = (
                df.groupby(["window_timestamp", "edge_id", "traj_id"])
                .agg(
                    {
                        "speed": "mean",
                        "free_flow_speed": "mean",
                    }
                )
                .reset_index()
            )
            agg_features = agg_features.compute()
        else:
            df = df.assign(window_timestamp=df["timestamp"].dt.floor(window_size))
            agg_features = (
                df.groupby(["window_timestamp", "edge_id", "traj_id"])
                .agg(
                    {
                        "speed": "mean",
                        "free_flow_speed": "mean",
                    }
                )
                .reset_index()
            )

        total_windows = len(agg_features)
        agg_features.columns = [
            "timestamp",
            "edge_id",
            "traj_id",
            "speed_mean",
            "free_flow_speed_mean",
        ]

        logging.info("Verificando NaN após agregação...")
        nan_counts_agg = agg_features.isna().sum()
        logging.info(f"NaN após agregação:\n{nan_counts_agg.to_string()}")

        logging.info("Adding temporal features...")
        agg_features["hour"] = agg_features["timestamp"].dt.hour.astype(np.int8)
        agg_features["day_of_week"] = agg_features["timestamp"].dt.dayofweek.astype(
            np.int8
        )
        agg_features["is_peak_hour"] = (
            agg_features["timestamp"].dt.hour.isin([7, 8, 17, 18]).astype(np.int8)
        )
        agg_features["is_weekend"] = (
            agg_features["timestamp"].dt.dayofweek.isin([5, 6]).astype(np.int8)
        )
        agg_features["speed_change"] = agg_features.groupby("edge_id")[
            "speed_mean"
        ].pct_change()
        agg_features["speed_change"] = (
            agg_features["speed_change"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=-1, upper=1)
        )

        logging.info("Verificando NaN após cálculo de speed_change...")
        nan_counts_speed_change = agg_features["speed_change"].isna().sum()
        logging.info(f"NaN em speed_change: {nan_counts_speed_change}")

        logging.info("Calculating future SRI...")
        future_sri = df[["timestamp", "edge_id", "traj_id", "sri"]].copy()
        if is_dask:
            future_sri = future_sri.assign(
                window_end=future_sri["timestamp"].dt.floor(window_size)
            )
            future_sri_agg = (
                future_sri.groupby(["edge_id", "traj_id", "window_end"])
                .agg({"sri": "mean"})
                .reset_index()
            )
            future_sri_agg = future_sri_agg.compute()
        else:
            future_sri = future_sri.assign(
                window_end=future_sri["timestamp"].dt.floor(window_size)
            )
            future_sri_agg = (
                future_sri.groupby(["edge_id", "traj_id", "window_end"])
                .agg({"sri": "mean"})
                .reset_index()
            )

        logging.info("Distribuição do SRI:")
        logging.info(future_sri_agg["sri"].describe())
        logging.info(f"Valores de SRI negativos: {(future_sri_agg['sri'] < 0).sum()}")
        logging.info(f"Valores de SRI nulos: {future_sri_agg['sri'].isna().sum()}")

        agg_features["future_time"] = agg_features["timestamp"] + pd.Timedelta(
            prediction_horizon
        )
        agg_features["future_time_floored"] = agg_features["future_time"].dt.floor(
            window_size
        )
        agg_features = agg_features.merge(
            future_sri_agg,
            left_on=["edge_id", "traj_id", "future_time_floored"],
            right_on=["edge_id", "traj_id", "window_end"],
            how="left",
        )

        logging.info("Assigning labels...")

        def assign_sri_label(sri):
            if pd.isna(sri):
                logging.info("SRI nulo encontrado, atribuindo 'none'")
                return "none"
            if sri < 0.15:
                return "none"
            elif sri < 0.4:
                return "light"
            elif sri < 0.7:
                return "moderate"
            else:
                return "severe"

        agg_features["label"] = agg_features["sri"].apply(assign_sri_label)
        empty_future_count = agg_features["sri"].isna().sum()
        null_sri_count = agg_features["sri"].isna().sum() - empty_future_count
        agg_features = agg_features.drop(
            columns=["sri", "window_end", "future_time", "future_time_floored"]
        ).dropna(subset=["label"])

        logging.info(f"Created {len(agg_features)} windows with features and labels.")
        logging.info(f"Total initial windows: {total_windows}")
        logging.info(
            f"Windows discarded due to empty future data: {empty_future_count} ({empty_future_count/total_windows*100:.2f}%)"
        )
        logging.info(
            f"Windows discarded due to null SRI: {null_sri_count} ({null_sri_count/total_windows*100:.2f}%)"
        )
        return agg_features

    def prepare_data(self, df):
        if self.scaler_params is None or self.edge_id_encoder is None:
            raise ValueError(
                "Normalization parameters or edge_id encoder not initialized."
            )
        if isinstance(df, pd.DataFrame) and all(
            col in df.columns
            for col in [
                "speed_mean",
                "free_flow_speed_mean",
                "speed_change",
                "hour",
                "day_of_week",
                "is_peak_hour",
                "is_weekend",
                "edge_id",
            ]
        ):
            x = df.copy()
        else:
            required_columns = [
                "timestamp",
                "traj_id",
                "edge_id",
                "speed",
                "free_flow_speed",
                "speed_segment",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            df = df.copy()
            x = self._create_windows(df)

        x["edge_id"] = self.edge_id_encoder.transform(x["edge_id"]).astype(np.int64)
        for col in self.scaler_params:
            mean = self.scaler_params[col]["mean"]
            std = self.scaler_params[col]["std"]
            x[col] = (x[col] - mean) / std if std > 0 else x[col]
        return x

    def predict(self, df):
        if self.model is None or self.label_encoder is None:
            raise ValueError("Model or label encoder not initialized.")
        x_scaled = self.prepare_data(df)
        probabilities = self.model.predict_proba(x_scaled)
        decoded_classes = self.label_encoder.inverse_transform(range(len(self.CLASSES)))
        return [
            {cls: float(prob) for cls, prob in zip(decoded_classes, probs)}
            for probs in probabilities
        ]


def main():
    model = CongestionModel()
    if model.model is None:
        logging.info("Model not loaded. Training a new model...")
        model.train_model()
    logging.info("Model training completed with evaluation on test set.")


if __name__ == "__main__":
    main()

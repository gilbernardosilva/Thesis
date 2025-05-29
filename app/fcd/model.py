import logging
import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.fcd.config.variables import FEATURES_DATA_FILE, MODEL_PATH, SCALER_PARAMS_PATH


class CongestionModel:
    CLASSES = ["none", "light", "moderate", "severe"]

    def __init__(
        self,
        model_type="lightgbm",
        model_path=MODEL_PATH,
        scaler_path=SCALER_PARAMS_PATH,
    ):
        self.model_type = model_type
        self.model_path = f"{model_path}_{model_type}.pkl"  # Diferente para cada modelo
        self.scaler_path = f"{scaler_path}_{model_type}.pkl"
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model_files()

    def _load_model_files(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logging.info(f"{self.model_type} model and scaler loaded successfully.")
        else:
            logging.info(
                f"No existing {self.model_type} model found. Please train a new model."
            )

    def _create_label(self, sri: float) -> int:
        sri = max(0.0, min(1.0, sri))
        if sri < 0.15:
            return 0  # None
        elif sri < 0.35:
            return 1  # Light
        elif sri < 0.70:
            return 2  # Moderate
        else:
            return 3  # Severe

    def _prepare_data(self, df):
        features = [
            "day_of_week",
            "hour",
            "is_peak_hour",
            "is_weekend",
            "length_segment",
            "road_class_encoded",
            "probe_speed",
            "avg_speed_segment",
            "speed_bin_0",
            "speed_bin_1",
            "speed_bin_2",
            "speed_bin_3",
            "speed_bin_4",
            "total_stopped_time_per_edge_per_traj",
        ]
        x = df[features + ["traj_id"]].copy()
        return x

    def _evaluate(self, model, x_test, y_test, model_type):
        x_test_df = pd.DataFrame(x_test, columns=self.feature_names)
        y_pred = model.predict(x_test_df)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
        report = classification_report(y_test, y_pred, target_names=self.CLASSES)
        logging.info(f"Model: {model_type}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score (weighted): {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"Classification Report:\n{report}")
        return accuracy, f1

    def train_model(self, data_path=FEATURES_DATA_FILE, cv_folds=0):
        df = pd.read_parquet(data_path)
        x = self._prepare_data(df)
        y = df["sri"].apply(self._create_label)

        traj_ids = df["traj_id"].unique()
        traj_train, traj_test = train_test_split(
            traj_ids, test_size=0.3, random_state=42
        )
        x_train = x[x["traj_id"].isin(traj_train)]
        x_test = x[x["traj_id"].isin(traj_test)]
        y_train = y[x["traj_id"].isin(traj_train)]
        y_test = y[x["traj_id"].isin(traj_test)]

        if cv_folds > 0:
            logging.info(
                f"Performing {cv_folds}-fold cross-validation for {self.model_type}..."
            )
            gkf = GroupKFold(n_splits=cv_folds)
            groups = x_train["traj_id"]
            cv_accuracies = []
            cv_f1_scores = []
            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(x_train, y_train, groups)
            ):
                x_train_fold = x_train.iloc[train_idx].drop(columns=["traj_id"])
                x_val_fold = x_train.iloc[val_idx].drop(columns=["traj_id"])
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]

                scaler = StandardScaler()
                x_train_fold_scaled = scaler.fit_transform(x_train_fold)
                x_val_fold_scaled = scaler.transform(x_val_fold)

                x_train_fold_scaled = pd.DataFrame(
                    x_train_fold_scaled, columns=x_train_fold.columns
                )
                x_val_fold_scaled = pd.DataFrame(
                    x_val_fold_scaled, columns=x_val_fold.columns
                )

                if self.model_type == "lightgbm":
                    model = LGBMClassifier(
                        objective="multiclass", num_class=4, random_state=42
                    )
                elif self.model_type == "xgboost":
                    model = XGBClassifier(
                        objective="multi:softprob", num_class=4, random_state=42
                    )
                else:
                    raise ValueError(
                        "Unsupported model_type. Use 'lightgbm' or 'xgboost'."
                    )

                model.fit(x_train_fold_scaled, y_train_fold)
                y_pred = model.predict(x_val_fold_scaled)
                accuracy = accuracy_score(y_val_fold, y_pred)
                f1 = f1_score(y_val_fold, y_pred, average="weighted")
                cv_accuracies.append(accuracy)
                cv_f1_scores.append(f1)
                logging.info(
                    f"{self.model_type} Fold {fold + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}"
                )

            avg_cv_accuracy = np.mean(cv_accuracies)
            avg_cv_f1 = np.mean(cv_f1_scores)
            logging.info(
                f"{self.model_type} Average CV Accuracy: {avg_cv_accuracy:.4f}"
            )
            logging.info(f"{self.model_type} Average CV F1 Score: {avg_cv_f1:.4f}")

        x_train_final = x_train.drop(columns=["traj_id"])
        x_test_final = x_test.drop(columns=["traj_id"])
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train_final)
        x_test_scaled = self.scaler.transform(x_test_final)

        self.feature_names = x_train_final.columns.tolist()

        x_train_scaled = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_test_scaled = pd.DataFrame(x_test_scaled, columns=self.feature_names)

        if self.model_type == "lightgbm":
            self.model = LGBMClassifier(
                objective="multiclass", num_class=4, random_state=42
            )
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                objective="multi:softprob", num_class=4, random_state=42
            )

        self.model.fit(x_train_scaled, y_train)
        logging.info(
            f"Training final {self.model_type} model on entire training set..."
        )
        test_accuracy, test_f1 = self._evaluate(
            self.model, x_test_scaled, y_test, self.model_type
        )

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logging.info(f"{self.model_type} model and scaler saved successfully.")

        if cv_folds > 0:
            logging.info(
                f"{self.model_type} Comparison of 70-30 split vs Cross-Validation:"
            )
            logging.info(
                f"70-30 Test Accuracy: {test_accuracy:.4f}, {self.model_type} CV Avg Accuracy: {avg_cv_accuracy:.4f}"
            )
            logging.info(
                f"70-30 Test F1 Score: {test_f1:.4f}, {self.model_type} CV Avg F1 Score: {avg_cv_f1:.4f}"
            )

    def predict(self, df):
        if not self.model or not self.scaler:
            raise ValueError("Model or scaler not initialized. Train the model first.")
        x = self._prepare_data(df)
        x = x.drop(columns=["traj_id"])
        x_scaled = self.scaler.transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=self.feature_names)
        return self.model.predict_proba(x_scaled)


def main():
    model = CongestionModel(model_type="lightgbm")
    if not model.model:
        logging.info("Starting training for LightGBM model with 5-fold CV...")
        model.train_model(cv_folds=5)

    model = CongestionModel(model_type="xgboost")
    if not model.model:
        logging.info("Starting training for XGBoost model with 5-fold CV...")
        model.train_model(cv_folds=5)


if __name__ == "__main__":
    main()

"""Modelo LSTM para predição de risco de burnout em séries temporais."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BurnoutDataset(Dataset):
    """Dataset PyTorch para sequências temporais de burnout."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class BurnoutLSTMModel(nn.Module):
    """Rede LSTM bidirecional com camadas densas para classificação."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        logits = self.fc3(output)
        probabilities = self.softmax(logits)
        return logits, probabilities


class BurnoutPredictor:
    """Orquestra preparo de dados, treino, salvamento e inferência."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model: Optional[BurnoutLSTMModel] = None
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Usando dispositivo: %s", self.device)

    def _prepare_inference_features(self, df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
        if "data_checkin" not in df.columns:
            df["data_checkin"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df))
        df["data_checkin"] = pd.to_datetime(df["data_checkin"], errors="coerce")
        df.sort_values(["usuario_id", "data_checkin"], inplace=True)

        base_defaults = {
            "nivel_stress": 5.0,
            "horas_trabalhadas": 8.0,
            "horas_sono": 7.0,
            "score_bemestar": 60.0,
        }
        for col, default in base_defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        df["stress_ma_7d"] = (
            df.groupby("usuario_id")["nivel_stress"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
            .fillna(method="bfill")
            .fillna(base_defaults["nivel_stress"])
        )

        sentiment_scores = {
            "positivo": 3,
            "neutro": 2,
            "negativo": 1,
            "ansioso": 1.5,
            "frustrado": 1.5,
            "satisfeito": 3,
            "cansado": 1.5,
        }
        if "sentimento" in df.columns:
            df["sentiment_score"] = df["sentimento"].map(sentiment_scores).fillna(2)
        else:
            df["sentiment_score"] = 2
        df["humor_variability"] = (
            df.groupby("usuario_id")["sentiment_score"].transform("std").fillna(0)
        )

        df["produtividade_trend"] = (
            df.groupby("usuario_id")["horas_trabalhadas"]
            .transform(lambda x: x.diff().rolling(window=7, min_periods=1).mean())
            .fillna(0)
        )

        nivel_stress = df["nivel_stress"].clip(1, 10)
        horas_sono = df["horas_sono"].clip(0, 10)
        score_bemestar = df["score_bemestar"].clip(0, 100)
        df["wellbeing_composite"] = (
            (10 - nivel_stress) * 0.4
            + (horas_sono / 8) * 30 * 0.3
            + score_bemestar * 0.3
        )

        df["nivel_stress"] = df["nivel_stress"].clip(1, 10) / 10.0
        df["horas_trabalhadas"] = df["horas_trabalhadas"].clip(0, 16) / 16.0
        df["horas_sono"] = df["horas_sono"].clip(0, 12) / 12.0
        df["score_bemestar"] = df["score_bemestar"].clip(0, 100) / 100.0
        df["stress_ma_7d"] = df["stress_ma_7d"].clip(1, 10) / 10.0
        df["wellbeing_composite"] = df["wellbeing_composite"].clip(0, 100) / 100.0
        df["humor_variability"] = df["humor_variability"].clip(0, 3) / 3.0
        df["produtividade_trend"] = df["produtividade_trend"].clip(-5, 5) / 5.0

        return df

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        target_col: str = "nivel_risco",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constrói janelas temporais e labels derivados."""
        df = self._prepare_inference_features(df.copy(), sequence_length)
        feature_cols = [
            "nivel_stress",
            "horas_trabalhadas",
            "horas_sono",
            "score_bemestar",
            "stress_ma_7d",
            "wellbeing_composite",
            "humor_variability",
            "produtividade_trend",
        ]
        available = [col for col in feature_cols if col in df.columns]
        if not available:
            raise ValueError("Nenhuma feature disponível para construir sequências.")
        self.feature_names = available

        sequences: list[np.ndarray] = []
        labels: list[str] = []

        for _, group in df.groupby("usuario_id"):
            group = group.sort_values("data_checkin")
            if len(group) < sequence_length:
                continue
            feature_matrix = group[available].to_numpy(dtype=float)
            for i in range(len(feature_matrix) - sequence_length + 1):
                window = feature_matrix[i : i + sequence_length]
                sequences.append(window)
                if target_col in group.columns:
                    label_value = group[target_col].iloc[i + sequence_length - 1]
                else:
                    label_value = self._calculate_risk_level(group.iloc[i + sequence_length - 1])
                labels.append(label_value)

        if not sequences:
            raise ValueError("Nenhuma sequência válida foi gerada.")

        X = np.stack(sequences)
        if target_col in df.columns:
            y_encoded = self.label_encoder.fit_transform(labels)
            logger.info("Sequências criadas: %s amostras", len(X))
            logger.info("Shape das sequências: %s", X.shape)
            logger.info("Distribuição de classes: %s", np.bincount(y_encoded))
        else:
            y_encoded = np.zeros(len(sequences), dtype=int)
            logger.info("Sequências criadas (inferência): %s amostras", len(X))
        return X, y_encoded

    def _calculate_risk_level(self, row: pd.Series) -> str:
        """Classifica risco em faixas com base em múltiplas métricas."""
        stress = float(row.get("nivel_stress", 0.5))
        horas_trabalhadas = float(row.get("horas_trabalhadas", 0.5))
        horas_sono = float(row.get("horas_sono", 0.5))
        score_bemestar = float(row.get("score_bemestar", 0.5))

        stress_component = stress * 40
        work_component = horas_trabalhadas * 20
        sleep_component = max(0.0, 1 - horas_sono) * 20
        wellbeing_component = max(0.0, 1 - score_bemestar) * 20

        risk_score = stress_component + work_component + sleep_component + wellbeing_component

        if risk_score < 25:
            return "baixo"
        if risk_score < 50:
            return "medio"
        if risk_score < 75:
            return "alto"
        return "critico"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list[float]]:
        """Treina o modelo e retorna histórico de métricas."""
        logger.info("Iniciando treinamento do modelo LSTM")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        train_loader = DataLoader(BurnoutDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(BurnoutDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        input_size = X.shape[2]
        self.model = BurnoutLSTMModel(
            input_size=input_size,
            hidden_size=self.config.get("hidden_size", 128),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.3),
            num_classes=len(np.unique(y)),
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        history: Dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = self.config.get("early_stopping_patience", 10)

        for epoch in range(epochs):
            train_loss, train_correct, train_total = 0.0, 0, 0
            self.model.train()
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(sequences)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_total += labels.size(0)
                train_correct += (predictions == labels).sum().item()

            val_loss, val_correct, val_total = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)
                    logits, _ = self.model(sequences)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    val_total += labels.size(0)
                    val_correct += (predictions == labels).sum().item()

            train_loss_avg = train_loss / max(len(train_loader), 1)
            train_acc = train_correct / max(train_total, 1)
            val_loss_avg = val_loss / max(len(val_loader), 1)
            val_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(train_loss_avg)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss_avg)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss_avg)

            logger.info(
                "Época %d/%d - Train Loss: %.4f, Train Acc: %.4f - Val Loss: %.4f, Val Acc: %.4f",
                epoch + 1,
                epochs,
                train_loss_avg,
                train_acc,
                val_loss_avg,
                val_acc,
            )

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                if checkpoint_path:
                    self.save_model(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping acionado na época %d", epoch + 1)
                    break

        logger.info("Treinamento concluído")
        return history

    def predict(self, sequence: np.ndarray) -> Dict[str, object]:
        """Realiza predição a partir de uma sequência temporal normalizada."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ou carregado.")
        if not self.feature_names:
            raise ValueError("Label encoder não foi ajustado.")

        self.model.eval()
        tensor = torch.as_tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, probabilities = self.model(tensor)
            predicted_index = torch.argmax(logits, dim=1).item()
            probs = probabilities.squeeze(0).cpu().numpy()

        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]
        prob_map = {label: float(prob) for label, prob in zip(self.label_encoder.classes_, probs)}
        return {
            "predicted_class": predicted_label,
            "probabilities": prob_map,
            "confidence": float(np.max(probs)),
            "risk_score": float(np.max(probs)),
        }

    def save_model(self, path: str) -> None:
        """Persiste o modelo e metadados necessários."""
        checkpoint = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Modelo salvo em %s", path)

    def load_model(self, path: str) -> None:
        """Carrega pesos e metadados de um checkpoint salvo."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        feature_names = checkpoint.get("feature_names", [])
        if not feature_names:
            raise ValueError("Checkpoint inválido: feature_names ausentes.")

        self.feature_names = feature_names
        config = checkpoint.get("config", {})
        self.config.update(config)

        self.model = BurnoutLSTMModel(
            input_size=len(feature_names),
            hidden_size=self.config.get("hidden_size", 128),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.3),
            num_classes=len(checkpoint["label_encoder"].classes_),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.label_encoder = checkpoint["label_encoder"]
        logger.info("Modelo carregado de %s", path)


if __name__ == "__main__":
    from pipelines.etl_pipeline import run_etl_pipeline

    dataframe = run_etl_pipeline(
        input_path="data/raw/checkins.csv",
        output_path="data/processed/checkins_processed.parquet",
    )
    predictor = BurnoutPredictor()
    X_data, y_data = predictor.prepare_sequences(dataframe)
    predictor.train(X_data, y_data, epochs=5, batch_size=32, checkpoint_path="models/storage/burnout_example.pt")
    prediction_result = predictor.predict(X_data[0])
    print("Predição:", prediction_result)


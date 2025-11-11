"""Modelo Prophet para previsão de bem-estar e análise temporal."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WellbeingForecaster:
    """Encapsula rotinas de previsão, anomalias e análise sazonal de bem-estar."""

    def __init__(self):
        self.models: Dict[int, Prophet] = {}
        self.historical_data: Dict[int, pd.DataFrame] = {}
        self.external_events: Dict[int, pd.DataFrame] = {}

    def prepare_data(self, df: pd.DataFrame, user_id: int, target_column: str = "score_bemestar") -> pd.DataFrame:
        """Filtra e formata dados históricos para o Prophet."""
        user_df = df[df["usuario_id"] == user_id].copy()
        if user_df.empty:
            raise ValueError(f"Sem dados para o usuário {user_id}.")

        user_df = user_df.sort_values("data_checkin")
        prophet_df = pd.DataFrame({"ds": pd.to_datetime(user_df["data_checkin"], errors="coerce"), "y": user_df[target_column]})
        prophet_df = prophet_df.dropna().drop_duplicates(subset="ds")
        if prophet_df.empty:
            raise ValueError(f"Dados inválidos para o usuário {user_id}.")

        self.historical_data[user_id] = prophet_df
        return prophet_df

    def train_model(
        self,
        user_id: int,
        df: Optional[pd.DataFrame] = None,
        include_holidays: bool = True,
        seasonality_mode: str = "multiplicative",
    ) -> Prophet:
        """Treina um modelo Prophet para o usuário informado."""
        if df is None:
            if user_id not in self.historical_data:
                raise ValueError(f"Nenhum dado histórico para o usuário {user_id}.")
            df = self.historical_data[user_id]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.05,
            interval_width=0.95,
        )

        if include_holidays:
            model.holidays = self._brazilian_holidays()

        model.fit(df)
        self.models[user_id] = model
        logger.info("Modelo Prophet treinado para usuário %s", user_id)
        return model

    def forecast(self, user_id: int, periods: int = 30, include_history: bool = True) -> pd.DataFrame:
        """Gera previsões por usuário."""
        if user_id not in self.models:
            raise ValueError(f"Modelo não treinado para o usuário {user_id}.")
        if user_id not in self.historical_data:
            raise ValueError(f"Histórico indisponível para o usuário {user_id}.")

        model = self.models[user_id]
        future = model.make_future_dataframe(periods=periods, freq="D")
        forecast = model.predict(future)

        if not include_history:
            cutoff = self.historical_data[user_id]["ds"].max()
            forecast = forecast[forecast["ds"] > cutoff]

        return forecast

    def detect_anomalies(self, user_id: int, threshold: float = 2.0) -> pd.DataFrame:
        """Detecta desvios significativos entre histórico e previsão."""
        if user_id not in self.models or user_id not in self.historical_data:
            return pd.DataFrame()

        model = self.models[user_id]
        historical = self.historical_data[user_id]
        forecast = model.predict(historical[["ds"]])

        merged = historical.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
        residual_std = merged["yhat"].std() or 1.0
        merged["residual"] = merged["y"] - merged["yhat"]
        merged["z_score"] = merged["residual"] / residual_std
        merged["is_anomaly"] = merged["z_score"].abs() > threshold

        anomalies = merged[merged["is_anomaly"]].copy()
        logger.info("Detectadas %s anomalias para usuário %s", len(anomalies), user_id)
        return anomalies

    def predict_high_risk_periods(self, user_id: int, forecast_days: int = 30, risk_threshold: float = 0.4) -> List[Dict[str, object]]:
        """Identifica janelas futuras de risco elevado."""
        forecast = self.forecast(user_id, periods=forecast_days, include_history=False)
        risky_points = forecast[forecast["yhat"] < risk_threshold]
        if risky_points.empty:
            return []

        risky_points = risky_points.copy()
        risky_points["gap"] = risky_points["ds"].diff().dt.days.fillna(1)
        risky_points["group"] = (risky_points["gap"] > 1).cumsum()

        periods: List[Dict[str, object]] = []
        for _, group in risky_points.groupby("group"):
            periods.append(
                {
                    "start_date": group["ds"].min(),
                    "end_date": group["ds"].max(),
                    "duration_days": int(group.shape[0]),
                    "min_predicted_score": float(group["yhat"].min()),
                    "max_predicted_score": float(group["yhat"].max()),
                    "avg_predicted_score": float(group["yhat"].mean()),
                }
            )

        logger.info("Identificados %s períodos de risco para usuário %s", len(periods), user_id)
        return periods

    def get_seasonality_analysis(self, user_id: int) -> Dict[str, Dict]:
        """Retorna padrões semanais e mensais previstos."""
        if user_id not in self.models:
            return {}

        forecast = self.forecast(user_id, periods=365, include_history=True)
        forecast["day_of_week"] = forecast["ds"].dt.day_name()
        forecast["month"] = forecast["ds"].dt.month

        trend_direction = "increasing"
        if forecast["trend"].iloc[-1] < forecast["trend"].iloc[0]:
            trend_direction = "decreasing"

        return {
            "weekly_pattern": forecast.groupby("day_of_week")["yhat"].mean().to_dict(),
            "monthly_pattern": forecast.groupby("month")["yhat"].mean().to_dict(),
            "trend": trend_direction,
        }

    def incorporate_external_events(self, user_id: int, events: List[Dict[str, object]]) -> None:
        """Registra eventos externos que podem ser usados em análises futuras."""
        if not events:
            return
        events_df = pd.DataFrame(events)
        events_df["ds"] = pd.to_datetime(events_df["date"], errors="coerce")
        events_df = events_df.dropna(subset=["ds"])
        if events_df.empty:
            return
        current = self.external_events.get(user_id, pd.DataFrame())
        combined = pd.concat([current, events_df], ignore_index=True).drop_duplicates(subset=["ds", "type"])
        self.external_events[user_id] = combined
        logger.info("Eventos externos registrados para usuário %s", user_id)

    def update_model(self, user_id: int, new_data: pd.DataFrame) -> None:
        """Atualiza dados históricos e reentreina o modelo."""
        prepared = new_data.copy()
        if {"ds", "y"} <= set(prepared.columns):
            prepared["ds"] = pd.to_datetime(prepared["ds"], errors="coerce")
            prepared = prepared.dropna(subset=["ds", "y"]).sort_values("ds")
        else:
            raise ValueError("Novos dados devem conter colunas 'ds' e 'y'.")

        existing = self.historical_data.get(user_id, pd.DataFrame())
        combined = pd.concat([existing, prepared], ignore_index=True).drop_duplicates(subset="ds").sort_values("ds")
        self.historical_data[user_id] = combined
        self.train_model(user_id, df=combined)

    def _brazilian_holidays(self) -> pd.DataFrame:
        """Retorna feriados nacionais para alimentar o Prophet."""
        current_year = datetime.now().year
        dates = [
            f"{current_year}-01-01",
            f"{current_year}-02-12",
            f"{current_year}-03-29",
            f"{current_year}-04-21",
            f"{current_year}-05-01",
            f"{current_year}-09-07",
            f"{current_year}-10-12",
            f"{current_year}-11-02",
            f"{current_year}-11-15",
            f"{current_year}-12-25",
        ]
        return pd.DataFrame({"holiday": "brazil_holiday", "ds": pd.to_datetime(dates), "lower_window": 0, "upper_window": 1})


if __name__ == "__main__":
    forecaster = WellbeingForecaster()
    days = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    sample = pd.DataFrame({"ds": days, "y": 0.6 + 0.1 * np.sin(np.linspace(0, 6.28, len(days)))})
    forecaster.historical_data[1] = sample
    forecaster.train_model(1)
    print(forecaster.forecast(1, periods=7).tail())


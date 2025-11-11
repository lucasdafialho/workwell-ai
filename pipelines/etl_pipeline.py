"""Pipeline de ETL para preparação de dados de bem-estar."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline completo de preparação de dados para aprendizado de máquina."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []

    def extract(self, data_source: str) -> pd.DataFrame:
        """Extrai dados da fonte informada."""
        logger.info("Extraindo dados de %s", data_source)

        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
            df = self._convert_dtypes(df)
        else:
            df = self._generate_sample_data()

        logger.info("Dados extraídos: %s registros", len(df))
        return df

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte colunas relevantes para tipos adequados."""
        numeric_cols = ['id', 'usuario_id', 'nivel_stress', 'horas_trabalhadas', 'horas_sono', 'score_bemestar']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'data_checkin' in df.columns:
            df['data_checkin'] = pd.to_datetime(df['data_checkin'], errors='coerce')

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica toda a preparação de dados."""
        logger.info("Iniciando transformação de dados")

        df = self._convert_dtypes(df)
        df = self._clean_data(df)
        df = self._handle_missing_values(df)
        df = self._create_derived_features(df)
        df = self._encode_categorical(df)
        df = self._normalize_features(df)
        df = self._create_temporal_sequences(df)

        logger.info("Transformação concluída")
        return df

    def load(self, df: pd.DataFrame, output_path: str):
        """Salva os dados transformados."""
        logger.info("Salvando dados processados em %s", output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas e outliers extremos."""
        df = df.drop_duplicates()

        numeric_cols = ['id', 'usuario_id', 'nivel_stress', 'horas_trabalhadas', 'horas_sono', 'score_bemestar']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        candidate_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['id', 'usuario_id']]
        for col in candidate_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isna()]

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa valores faltantes com estratégias simples."""
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['id', 'usuario_id']]
        for col in numeric_cols:
            df[col] = df.groupby('usuario_id')[col].transform(lambda x: x.fillna(x.mean())) if 'usuario_id' in df.columns else df[col]
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

        categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'data_checkin']
        for col in categorical_cols:
            mode_value = df[col].mode()
            fill_value = mode_value[0] if not mode_value.empty else 'unknown'
            df[col] = df[col].fillna(fill_value)

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features agregadas relevantes."""
        if 'nivel_stress' in df.columns and 'data_checkin' in df.columns:
            df = df.sort_values('data_checkin')
            df['nivel_stress'] = pd.to_numeric(df['nivel_stress'], errors='coerce')
            df['stress_ma_7d'] = df.groupby('usuario_id')['nivel_stress'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

        if 'sentimento' in df.columns:
            sentiment_scores = {
                'positivo': 3,
                'neutro': 2,
                'negativo': 1,
                'ansioso': 1.5,
                'frustrado': 1.5,
                'satisfeito': 3,
                'cansado': 1.5,
            }
            df['sentiment_score'] = df['sentimento'].map(sentiment_scores).fillna(2)
            df['humor_variability'] = df.groupby('usuario_id')['sentiment_score'].transform('std').fillna(0)

        if 'horas_trabalhadas' in df.columns:
            df['horas_trabalhadas'] = pd.to_numeric(df['horas_trabalhadas'], errors='coerce')
            df['produtividade_trend'] = df.groupby('usuario_id')['horas_trabalhadas'].transform(lambda x: x.diff().rolling(window=7, min_periods=1).mean()).fillna(0)

        if all(col in df.columns for col in ['nivel_stress', 'horas_sono', 'score_bemestar']):
            nivel_stress = pd.to_numeric(df['nivel_stress'], errors='coerce').fillna(5)
            horas_sono = pd.to_numeric(df['horas_sono'], errors='coerce').fillna(8)
            score_bemestar = pd.to_numeric(df['score_bemestar'], errors='coerce').fillna(50)
            df['wellbeing_composite'] = (10 - nivel_stress) * 0.4 + (horas_sono / 8) * 30 * 0.3 + score_bemestar * 0.3

        if 'data_checkin' in df.columns:
            df['data_checkin'] = pd.to_datetime(df['data_checkin'], errors='coerce')
            df['day_of_week'] = df['data_checkin'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['checkin_frequency'] = df.groupby('usuario_id')['data_checkin'].transform(lambda x: x.diff().dt.days.fillna(1))

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica codificação ordinal em variáveis categóricas."""
        categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'data_checkin']
        for col in categorical_cols:
            if col not in self.label_encoders:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza colunas numéricas para o intervalo [0, 1]."""
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['usuario_id', 'id']]
        if not numeric_cols:
            return df

        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[numeric_cols] = self.minmax_scaler.fit_transform(df[numeric_cols])
        self.feature_names = numeric_cols
        return df

    def _create_temporal_sequences(self, df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
        """Mantém compatibilidade sem gerar sequências explícitas."""
        return df

    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica SMOTE para balanceamento de classes."""
        logger.info("Balanceando dataset com SMOTE")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        logger.info("Dataset balanceado: %s amostras", len(X_balanced))
        return X_balanced, y_balanced

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Executa verificações básicas de qualidade dos dados."""
        quality_checks = {
            'has_data': len(df) > 0,
            'no_all_null': not df.isnull().all().any(),
            'has_target': 'nivel_risco' in df.columns or 'score_risco' in df.columns,
            'has_features': len(df.select_dtypes(include=[np.number]).columns) > 0,
            'temporal_consistency': True,
        }

        if 'data_checkin' in df.columns:
            df['data_checkin'] = pd.to_datetime(df['data_checkin'], errors='coerce')
            quality_checks['temporal_consistency'] = df['data_checkin'].max() > df['data_checkin'].min()

        logger.info("Checks de qualidade: %s", quality_checks)
        return quality_checks

    def _generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Gera dados sintéticos para demonstração."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        data = {
            'id': range(1, n_samples + 1),
            'usuario_id': np.random.randint(1, 50, n_samples),
            'data_checkin': np.random.choice(dates, n_samples),
            'nivel_stress': np.random.randint(1, 11, n_samples),
            'horas_trabalhadas': np.random.uniform(4, 12, n_samples).round(2),
            'horas_sono': np.random.uniform(5, 9, n_samples).round(2),
            'sentimento': np.random.choice(['positivo', 'neutro', 'negativo'], n_samples),
            'score_bemestar': np.random.uniform(30, 100, n_samples).round(2),
            'observacoes': [f"Observação {i}" for i in range(n_samples)],
        }
        df = pd.DataFrame(data)
        missing_indices = np.random.choice(df.index, size=int(n_samples * 0.1), replace=False)
        df.loc[missing_indices, 'horas_sono'] = np.nan
        return df


def run_etl_pipeline(input_path: str, output_path: str, config: Optional[Dict] = None) -> pd.DataFrame:
    """Executa o pipeline completo de ETL."""
    pipeline = DataPipeline(config)
    df = pipeline.extract(input_path)
    quality_checks = pipeline.validate_data_quality(df)
    if not all(quality_checks.values()):
        logger.warning("Alguns checks de qualidade falharam")
    df_processed = pipeline.transform(df)
    pipeline.load(df_processed, output_path)
    return df_processed


if __name__ == "__main__":
    result = run_etl_pipeline(
        input_path="data/raw/checkins.csv",
        output_path="data/processed/checkins_processed.parquet",
    )
    print(f"Dados processados: {len(result)} registros")
    print(f"Features: {result.columns.tolist()}")


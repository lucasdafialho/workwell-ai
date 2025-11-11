"""Treinamento do modelo LSTM de predição de burnout."""

import argparse
import sys
from pathlib import Path

import numpy as np

from models.burnout.lstm_model import BurnoutPredictor
from pipelines.etl_pipeline import run_etl_pipeline

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def train_burnout_model(
    data_path: str | None = None,
    output_model_path: str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
) -> bool:
    """Executa o treinamento completo do modelo de burnout."""
    data_target = Path(data_path) if data_path else ROOT_DIR / "data" / "raw" / "checkins.csv"
    if not data_target.is_absolute():
        data_target = ROOT_DIR / data_target

    model_target = Path(output_model_path) if output_model_path else ROOT_DIR / "models" / "storage" / "best_burnout_model.pt"
    if not model_target.is_absolute():
        model_target = ROOT_DIR / model_target

    print("=" * 60)
    print("TREINAMENTO DO MODELO DE PREDIÇÃO DE BURNOUT")
    print("=" * 60)

    if not data_target.exists():
        print(f"\nArquivo de dados não encontrado: {data_target}")
        print("Execute primeiro: python pipelines/generate_data.py")
        return False

    print("\n[1/4] Executando pipeline ETL...")
    try:
        processed_path = ROOT_DIR / "data" / "processed" / "checkins_processed.parquet"
        df_processed = run_etl_pipeline(input_path=str(data_target), output_path=str(processed_path))
        print(f"✓ Dados processados: {len(df_processed)} registros")
    except Exception as exc:
        print(f"Erro no pipeline ETL: {exc}")
        return False

    print("\n[2/4] Preparando sequências...")
    try:
        predictor = BurnoutPredictor()
        X, y = predictor.prepare_sequences(df_processed, sequence_length=30)
        distrib = dict(zip(*np.unique(y, return_counts=True)))
        print(f"✓ Sequências criadas: {len(X)}")
        print(f"  Shape: {X.shape}")
        print("  Distribuição de classes:")
        for cls, count in distrib.items():
            print(f"    {cls}: {count}")
    except Exception as exc:
        print(f"Erro ao preparar sequências: {exc}")
        return False

    print("\n[3/4] Treinando modelo LSTM...")
    try:
        history = predictor.train(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            validation_split=0.2,
        )
        print("✓ Treinamento concluído")
        print(f"  Melhor val_loss: {min(history['val_loss']):.4f}")
        print(f"  Melhor val_acc: {max(history['val_acc']):.4f}")
    except Exception as exc:
        print(f"Erro no treinamento: {exc}")
        return False

    print("\n[4/4] Salvando modelo...")
    try:
        model_target.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(model_target))
        print(f"✓ Modelo salvo em {model_target}")
    except Exception as exc:
        print(f"Erro ao salvar modelo: {exc}")
        return False

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo de burnout")
    parser.add_argument("--data", type=str, default="data/raw/checkins.csv", help="Caminho para dados de entrada")
    parser.add_argument("--output", type=str, default="models/storage/best_burnout_model.pt", help="Caminho para salvar o modelo")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch")

    args = parser.parse_args()

    ok = train_burnout_model(
        data_path=args.data,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    sys.exit(0 if ok else 1)


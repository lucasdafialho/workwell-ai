"""Treinamento do sistema de recomendação híbrido."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from services.recommendation.recommendation_engine import RecommendationEngine

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def train_recommendation_system(interactions_path: str | None = None) -> bool:
    """Treina o sistema de recomendação utilizando dados de interações."""
    path = Path(interactions_path) if interactions_path else ROOT_DIR / "data" / "raw" / "interactions.csv"
    if not path.is_absolute():
        path = ROOT_DIR / path

    print("=" * 60)
    print("TREINAMENTO DO SISTEMA DE RECOMENDAÇÃO")
    print("=" * 60)

    if not path.exists():
        print(f"\nArquivo de interações não encontrado: {path}")
        print("Execute primeiro: python pipelines/generate_data.py")
        return False

    print("\n[1/3] Carregando dados de interações...")
    try:
        interactions_df = pd.read_csv(path)
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        print(f"✓ Dados carregados: {len(interactions_df)} interações")
        print(f"  Usuários únicos: {interactions_df['user_id'].nunique()}")
        print(f"  Itens únicos: {interactions_df['item_id'].nunique()}")
    except Exception as exc:
        print(f"Erro ao carregar dados: {exc}")
        return False

    engine = RecommendationEngine()

    print("\n[2/3] Treinando modelo collaborative filtering...")
    try:
        engine.train_collaborative_model(interactions_df)
        print("✓ Modelo de collaborative filtering treinado")
    except Exception as exc:
        print(f"Aviso no treinamento collaborative: {exc}")

    print("\n[3/3] Preparando modelo content-based...")
    try:
        items_data = []
        for items in engine.items_catalog.values():
            for item in items:
                items_data.append(
                    {
                        'id': item['id'],
                        'title': item['title'],
                        'description': item['description'],
                        'tags': ', '.join(item.get('tags', [])),
                        'duration': item.get('duration', 0),
                    }
                )
        items_df = pd.DataFrame(items_data)
        engine.train_content_model(items_df)
        print("✓ Modelo content-based preparado")
    except Exception as exc:
        print(f"Aviso no treinamento content-based: {exc}")

    print("\n" + "=" * 60)
    print("SISTEMA DE RECOMENDAÇÃO PRONTO!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar sistema de recomendação")
    parser.add_argument("--data", type=str, default="data/raw/interactions.csv", help="Caminho para dados de interações")
    args = parser.parse_args()
    sucesso = train_recommendation_system(interactions_path=args.data)
    sys.exit(0 if sucesso else 1)


"""Orquestra o treinamento de todos os modelos do WorkWell AI."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    """Executa a rotina de treinamento escolhida."""
    parser = argparse.ArgumentParser(description="Treinar modelos do WorkWell AI")
    parser.add_argument(
        "--model",
        choices=["all", "burnout", "sentiment", "recommendation", "data"],
        default="all",
        help='Escolha do módulo a treinar ou "data" para gerar dados sintéticos.',
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Ignora geração de dados sintéticos quando já disponíveis.",
    )

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args(["--model", "all"])

    print("=" * 60)
    print("WORKWELL AI - TREINAMENTO DE MODELOS")
    print("=" * 60)

    if args.model == "data" or (args.model == "all" and not args.skip_data):
        print("\n[ETAPA 0] Gerando dados sintéticos...")
        try:
            from pipelines.generate_data import generate_checkin_data, generate_interaction_data

            generate_checkin_data(n_users=50, days=180)
            generate_interaction_data(n_users=50)
            print("✓ Dados gerados com sucesso\n")
        except Exception as exc:
            print(f"Erro ao gerar dados: {exc}")
            if args.model == "data":
                return
            print("Continuando com dados existentes...\n")

    if args.model in ["all", "burnout"]:
        print("\n[ETAPA 1] Treinando modelo de burnout...")
        try:
            from pipelines.train_burnout import train_burnout_model

            ok = train_burnout_model(epochs=50, batch_size=32)
            if not ok:
                print("Aviso: falha no treinamento de burnout, prosseguindo.\n")
        except Exception as exc:
            print(f"Aviso: erro no treinamento de burnout: {exc}\n")

    if args.model in ["all", "sentiment"]:
        print("\n[ETAPA 2] Preparando modelo de sentimento...")
        try:
            from pipelines.train_sentiment import prepare_sentiment_model

            prepare_sentiment_model()
        except Exception as exc:
            print(f"Aviso: erro ao preparar modelo de sentimento: {exc}\n")

    if args.model in ["all", "recommendation"]:
        print("\n[ETAPA 3] Treinando sistema de recomendação...")
        try:
            from pipelines.train_recommendation import train_recommendation_system

            train_recommendation_system()
        except Exception as exc:
            print(f"Aviso: erro no treinamento de recomendação: {exc}\n")

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. Verifique os modelos em models/storage/")
    print("2. Inicie a API: python main.py api")
    print("3. Teste os endpoints: python main.py test")


if __name__ == "__main__":
    main()


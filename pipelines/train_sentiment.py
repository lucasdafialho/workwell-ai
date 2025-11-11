"""Validação do modelo de análise de sentimento baseado em BERT."""

import sys
from pathlib import Path

from services.nlp.sentiment_analyzer import SentimentAnalyzer

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def prepare_sentiment_model() -> bool:
    """Inicializa e valida o modelo de análise de sentimento."""
    print("=" * 60)
    print("PREPARAÇÃO DO MODELO DE ANÁLISE DE SENTIMENTO")
    print("=" * 60)

    print("\n[1/2] Carregando modelo BERT...")
    try:
        analyzer = SentimentAnalyzer(model_name="neuralmind/bert-base-portuguese-cased")
        print("✓ Modelo BERT carregado")
    except Exception as exc:
        print(f"Erro ao carregar modelo principal: {exc}")
        print("\nTentando modelo alternativo...")
        try:
            analyzer = SentimentAnalyzer(model_name="bert-base-multilingual-cased")
            print("✓ Modelo alternativo carregado")
        except Exception as exc_alt:
            print(f"Erro ao carregar modelo alternativo: {exc_alt}")
            return False

    print("\n[2/2] Testando modelo...")
    exemplos = [
        "Estou muito satisfeito com o trabalho",
        "Estou sobrecarregado e cansado",
        "A semana foi produtiva",
    ]
    try:
        for texto in exemplos:
            resultado = analyzer.analyze_sentiment(texto)
            print(f"  Texto: '{texto[:50]}...'")
            print(f"    Sentimento: {resultado['sentiment']}, Score: {resultado['score']:.2f}")
        print("\n✓ Modelo funcionando corretamente")
    except Exception as exc:
        print(f"Erro ao testar modelo: {exc}")
        return False

    print("\n" + "=" * 60)
    print("MODELO DE SENTIMENTO PRONTO!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    sucesso = prepare_sentiment_model()
    sys.exit(0 if sucesso else 1)


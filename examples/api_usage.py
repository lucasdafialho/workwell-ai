"""
Exemplo de uso da API WorkWell AI
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_burnout_prediction():
    """Testa endpoint de predição de burnout."""
    url = f"{BASE_URL}/api/v1/burnout/predict"
    
    payload = {
        "usuario_id": 1,
        "checkins": [
            {
                "usuario_id": 1,
                "nivel_stress": 8,
                "horas_trabalhadas": 10,
                "horas_sono": 6,
                "sentimento": "sobrecarregado",
                "score_bemestar": 45,
                "data_checkin": datetime.now().isoformat()
            }
        ],
        "sequence_length": 30
    }
    
    response = requests.post(url, json=payload)
    print("Predição de Burnout:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()

def test_sentiment_analysis():
    """Testa endpoint de análise de sentimento."""
    url = f"{BASE_URL}/api/v1/sentiment/analyze"
    
    payload = {
        "texts": [
            "Estou muito sobrecarregado com o trabalho. Os prazos são impossíveis.",
            "Tive uma semana produtiva e estou satisfeito com os resultados."
        ],
        "user_id": 1
    }
    
    response = requests.post(url, json=payload)
    print("\nAnálise de Sentimento:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()

def test_emotional_support():
    """Testa endpoint de suporte emocional."""
    url = f"{BASE_URL}/api/v1/chat/support"
    
    payload = {
        "user_id": 1,
        "message": "Estou me sentindo muito sobrecarregado no trabalho. O que posso fazer?",
        "context": {
            "recent_checkins": [
                {"nivel_stress": 8, "horas_trabalhadas": 10}
            ],
            "wellbeing_score": 45
        }
    }
    
    response = requests.post(url, json=payload)
    print("\nSuporte Emocional:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()

def test_recommendations():
    """Testa endpoint de recomendações."""
    url = f"{BASE_URL}/api/v1/recommendations"
    
    payload = {
        "user_id": 1,
        "user_profile": {
            "preferred_tags": ["stress", "relaxamento"],
            "stress_level": 8,
            "available_time": 10
        },
        "context": {
            "current_hour": datetime.now().hour
        },
        "n_recommendations": 5
    }
    
    response = requests.post(url, json=payload)
    print("\nRecomendações:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()

def test_forecast():
    """Testa endpoint de previsão."""
    url = f"{BASE_URL}/api/v1/forecast/wellbeing"
    
    payload = {
        "user_id": 1,
        "periods": 30,
        "target_column": "score_bemestar"
    }
    
    response = requests.post(url, json=payload)
    print("\nPrevisão de Bem-estar:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()

def test_health():
    """Testa health check."""
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print("\nHealth Check:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTANDO API WORKWELL AI")
    print("=" * 60)

    try:
        test_health()
    except Exception as e:
        print(f"\nErro ao conectar à API: {e}")
        print("Certifique-se de que a API está rodando: uvicorn api.main:app")
        exit(1)
 
    try:
        test_burnout_prediction()
        test_sentiment_analysis()
        test_emotional_support()
        test_recommendations()
        test_forecast()
    except Exception as e:
        print(f"\nErro ao testar endpoints: {e}")
    
    print("\n" + "=" * 60)
    print("TESTES CONCLUÍDOS")
    print("=" * 60)


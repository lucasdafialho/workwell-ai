"""Geração de dados sintéticos para o módulo de IA."""

from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def generate_checkin_data(n_users: int = 50, days: int = 180, output_path: str | None = None) -> pd.DataFrame:
    """Gera dados sintéticos de check-ins diários."""
    target_path = Path(output_path) if output_path else ROOT_DIR / "data" / "raw" / "checkins.csv"
    if not target_path.is_absolute():
        target_path = ROOT_DIR / target_path

    print(f"Gerando dados sintéticos: {n_users} usuários, {days} dias...")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    sentimentos = ['positivo', 'neutro', 'negativo', 'ansioso', 'frustrado', 'satisfeito', 'cansado']
    registros: list[dict[str, object]] = []
    inicio = datetime.now() - timedelta(days=days)

    for user_id in range(1, n_users + 1):
        base_stress = np.random.uniform(3, 8)
        base_horas_trabalho = np.random.uniform(6, 10)
        base_horas_sono = np.random.uniform(6, 8)
        base_bemestar = np.random.uniform(40, 80)
        tendencia = np.random.choice(['melhorando', 'piorando', 'estavel'])

        for day in range(days):
            data = inicio + timedelta(days=day)

            if tendencia == 'piorando':
                stress_factor = 1 + (day / days) * 0.3
                bemestar_factor = 1 - (day / days) * 0.2
            elif tendencia == 'melhorando':
                stress_factor = 1 - (day / days) * 0.2
                bemestar_factor = 1 + (day / days) * 0.2
            else:
                stress_factor = 1
                bemestar_factor = 1

            fim_de_semana = data.weekday() >= 5
            stress_multiplier = 0.7 if fim_de_semana else 1.0
            horas_trabalho_multiplier = 0.3 if fim_de_semana else 1.0
            month_factor = 1 + 0.2 * np.sin(2 * np.pi * day / 30)

            nivel_stress = base_stress * stress_factor * stress_multiplier * month_factor + np.random.normal(0, 1.5)
            nivel_stress = int(min(max(nivel_stress, 1), 10))

            horas_trabalhadas = base_horas_trabalho * horas_trabalho_multiplier + np.random.normal(0, 1.5)
            horas_trabalhadas = round(min(max(horas_trabalhadas, 0), 24), 2)

            horas_sono = base_horas_sono + np.random.normal(0, 1)
            horas_sono = round(min(max(horas_sono, 4), 12), 2)

            score_bemestar = base_bemestar * bemestar_factor - nivel_stress * 5 + (horas_sono - 6) * 5 + np.random.normal(0, 10)
            score_bemestar = round(min(max(score_bemestar, 0), 100), 2)

            if nivel_stress >= 8 or score_bemestar < 40:
                sentimento = random.choice(['negativo', 'ansioso', 'frustrado', 'cansado'])
            elif nivel_stress <= 4 and score_bemestar > 70:
                sentimento = random.choice(['positivo', 'satisfeito'])
            else:
                sentimento = random.choice(sentimentos)

            observacoes = None
            if nivel_stress >= 8:
                observacoes = random.choice([
                    f"Muito sobrecarregado com o projeto {random.randint(1, 10)}",
                    "Deadline impossível esta semana",
                    "Não consigo descansar, trabalho acumulado",
                    "Reuniões demais, sem tempo para trabalho real",
                ])
            elif nivel_stress <= 4:
                observacoes = random.choice([
                    "Semana produtiva, bom progresso",
                    "Equipe colaborando bem",
                    "Consegui manter equilíbrio trabalho-vida",
                ])

            registros.append(
                {
                    'id': len(registros) + 1,
                    'usuario_id': user_id,
                    'data_checkin': data.strftime('%Y-%m-%d'),
                    'nivel_stress': nivel_stress,
                    'horas_trabalhadas': horas_trabalhadas,
                    'horas_sono': horas_sono,
                    'sentimento': sentimento,
                    'observacoes': observacoes,
                    'score_bemestar': score_bemestar,
                }
            )

    df = pd.DataFrame(registros)
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, 'horas_sono'] = np.nan
    df.loc[np.random.choice(df.index, size=int(len(df) * 0.02), replace=False), 'observacoes'] = None
    df.to_csv(target_path, index=False)

    print(f"✓ Dados salvos em {target_path}")
    print(f"  Total de check-ins: {len(df)}")
    print(f"  Período: {df['data_checkin'].min()} a {df['data_checkin'].max()}")
    print(f"  Usuários: {df['usuario_id'].nunique()}")
    return df


def generate_interaction_data(n_users: int = 50, output_path: str | None = None) -> pd.DataFrame:
    """Gera dados sintéticos de interações com recomendações."""
    target_path = Path(output_path) if output_path else ROOT_DIR / "data" / "raw" / "interactions.csv"
    if not target_path.is_absolute():
        target_path = ROOT_DIR / target_path

    print("Gerando dados de interações...")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    itens = ['mind_001', 'mind_002', 'mind_003', 'break_001', 'break_002', 'content_001']
    registros: list[dict[str, object]] = []

    for user_id in range(1, n_users + 1):
        n_interactions = np.random.randint(5, 20)
        for _ in range(n_interactions):
            item_id = random.choice(itens)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            completed = rating >= 3
            date = datetime.now() - timedelta(days=np.random.randint(0, 90))
            registros.append(
                {
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'completed': completed,
                    'timestamp': date.isoformat(),
                }
            )

    df = pd.DataFrame(registros)
    df.to_csv(target_path, index=False)
    print(f"✓ Dados de interações salvos em {target_path}")
    print(f"  Total de interações: {len(df)}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("GERAÇÃO DE DADOS SINTÉTICOS PARA TREINAMENTO")
    print("=" * 60)

    generate_checkin_data(n_users=50, days=180)
    generate_interaction_data(n_users=50)

    print("\n" + "=" * 60)
    print("DADOS GERADOS COM SUCESSO!")
    print("=" * 60)


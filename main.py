"""
Script principal para executar o pipeline completo de IA
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='WorkWell AI - Pipeline de IA')
    parser.add_argument(
        'command',
        choices=['setup', 'etl', 'train', 'api', 'test'],
        help='Comando a executar'
    )
    parser.add_argument(
        '--model',
        choices=['burnout', 'sentiment', 'all'],
        default='all',
        help='Modelo a treinar'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Porta da API'
    )
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        print("Executando setup...")
        from setup import main as setup_main
        setup_main()
    
    elif args.command == 'etl':
        print("Executando pipeline ETL...")
        from pipelines.etl_pipeline import run_etl_pipeline
        run_etl_pipeline(
            input_path="data/raw/checkins.csv",
            output_path="data/processed/checkins_processed.parquet"
        )
    
    elif args.command == 'train':
        print(f"Treinando modelo(s): {args.model}")
        # Importar e executar train_all diretamente
        import subprocess
        cmd = ['python', 'pipelines/train_all.py', '--model', args.model]
        if args.model == 'all':
            cmd.append('--skip-data')  # Assumir que dados já existem se usando main.py
        subprocess.run(cmd)
    
    elif args.command == 'api':
        print(f"Iniciando API na porta {args.port}...")
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=args.port,
            reload=True
        )
    
    elif args.command == 'test':
        print("Testando API...")
        from examples.api_usage import test_health, test_burnout_prediction
        try:
            test_health()
            test_burnout_prediction()
            print("Testes passaram!")
        except Exception as e:
            print(f"Erro nos testes: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()


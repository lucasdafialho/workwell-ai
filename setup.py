"""
Script de inicialização e setup do ambiente WorkWell AI
"""

import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""
    print("Verificando dependências...")
    
    required_packages = [
        'torch', 'tensorflow', 'sklearn', 'pandas', 'numpy',
        'transformers', 'fastapi', 'mlflow', 'prophet'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - FALTANDO")
            missing.append(package)
    
    if missing:
        print(f"\nDependências faltando: {', '.join(missing)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    print("\nTodas as dependências estão instaladas!")
    return True

def setup_directories():
    """Cria diretórios necessários."""
    print("\nCriando diretórios...")
    
    directories = [
        'data/raw',
        'data/processed',
        'models/storage',
        'logs',
        'notebooks',
        'mlruns'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")

def download_nlp_models():
    """Baixa modelos de NLP necessários."""
    print("\nBaixando modelos de NLP...")
    
    commands = [
        ['python', '-m', 'spacy', 'download', 'pt_core_news_sm'],
        ['python', '-m', 'nltk.downloader', 'punkt', 'stopwords']
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ {' '.join(cmd)}")
        except subprocess.CalledProcessError:
            print(f"✗ Erro ao executar: {' '.join(cmd)}")
            print("  Execute manualmente se necessário")

def create_env_file():
    """Cria arquivo .env se não existir."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("\nCriando arquivo .env...")
        env_file.write_text(env_example.read_text())
        print("✓ Arquivo .env criado (configure suas variáveis de ambiente)")
    elif env_file.exists():
        print("\n✓ Arquivo .env já existe")

def main():
    """Função principal de setup."""
    print("=" * 60)
    print("WORKWELL AI - SETUP")
    print("=" * 60)
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Criar diretórios
    setup_directories()
    
    # Criar .env
    create_env_file()
    
    # Baixar modelos NLP
    download_nlp_models()
    
    print("\n" + "=" * 60)
    print("SETUP CONCLUÍDO!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. Configure as variáveis de ambiente no arquivo .env")
    print("2. Execute: python pipelines/etl_pipeline.py para processar dados")
    print("3. Execute: python models/burnout/lstm_model.py para treinar modelo")
    print("4. Execute: uvicorn api.main:app para iniciar API")

if __name__ == "__main__":
    main()


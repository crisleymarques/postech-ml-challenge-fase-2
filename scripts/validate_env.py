#!/usr/bin/env python3
"""Script para validar o ambiente de desenvolvimento local.

Verifica dependências, carregamento das configurações essenciais
e estruturas de diretórios.
"""

import sys
from pathlib import Path


def check_directories() -> bool:
    """Verifica se todos os diretórios exigidos existem.

    Returns:
        True se todos os diretórios existirem, False caso contrário.
    """
    required_dirs = ["src", "tests", "data", "models", "configs", "scripts"]
    all_ok = True
    print("\n--- Verificando Diretórios do Projeto ---")
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.is_dir():
            print(f"✅ O diretório '{dir_name}' existe.")
        else:
            print(f"❌ O diretório '{dir_name}' está faltando.")
            all_ok = False
    return all_ok


def check_imports() -> bool:
    """Verifica a importação das principais bibliotecas e suas versões.

    Returns:
        True se todas as bibliotecas foram importadas com sucesso, False caso contrário.
    """
    libraries = ["torch", "sklearn", "mlflow", "dvc"]
    all_ok = True
    print("\n--- Verificando Importações de Bibliotecas ---")
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "versão desconhecida")
            print(f"✅ Importou '{lib}' com sucesso (versão: {version}).")
        except ImportError as e:
            print(f"❌ Falha ao importar '{lib}': {e}")
            all_ok = False
    return all_ok


def check_config() -> bool:
    """Verifica se o Pydantic Settings carrega as configurações corretamente.

    Returns:
        True se as configurações foram carregadas com sucesso, False caso contrário.
    """
    print("\n--- Verificando Pydantic Settings ---")
    try:
        from src.config import settings

        print("✅ Configurações do Pydantic carregadas com sucesso.")
        print(f"   Ambiente (Environment): {settings.environment}")
        print(f"   MLflow URI: {settings.mlflow_tracking_uri}")
        print(f"   MLflow Exp: {settings.mlflow_experiment_name}")
        print(f"   DVC Remote: {settings.dvc_remote_path}")
        return True
    except Exception as e:
        print(f"❌ Falha ao carregar as configurações do Pydantic: {e}")
        return False


def main() -> None:
    """Função principal de validação."""
    print("Iniciando a validação do ambiente do Postech ML Challenge...")
    dirs_ok = check_directories()
    imports_ok = check_imports()
    config_ok = check_config()

    if dirs_ok and imports_ok and config_ok:
        print("\n🎉 Validação do ambiente BEM-SUCEDIDA! Pronto para o desenvolvimento.")
        sys.exit(0)
    else:
        print("\n⚠️ Validação do ambiente FALHOU. Verifique os erros acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()

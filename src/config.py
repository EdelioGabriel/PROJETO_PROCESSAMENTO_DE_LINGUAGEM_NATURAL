"""Carregamento de configuracao e de segredos.

Duas fontes, com papeis distintos:

- ``config/config.yaml`` guarda parametros de experimento. Vai para o
  repositorio, porque sem ele os resultados nao sao reproduziveis.
- ``.env`` guarda segredos. Nunca vai para o repositorio.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent

_env_carregado = False


def carregar_env() -> None:
    """Le o .env de chave/chave.env, na raiz do repositorio. Idempotente."""
    global _env_carregado
    if not _env_carregado:
        load_dotenv(RAIZ / "chave" / "chave.env")
        _env_carregado = True


def carregar_config(caminho: str | Path | None = None) -> dict:
    """Le o config.yaml e devolve o dicionario."""
    caminho = Path(caminho) if caminho else RAIZ / "config" / "config.yaml"
    return yaml.safe_load(caminho.read_text(encoding="utf-8"))


def obter_env(nome: str, padrao: str | None = None) -> str | None:
    """Le uma variavel do .env (ou do ambiente do shell, que tem prioridade)."""
    carregar_env()
    return os.environ.get(nome, padrao)
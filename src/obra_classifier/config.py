"""Configurações centrais do projeto.

Todos os valores "mágicos" (classes, extensões, hiperparâmetros e caminhos padrão)
ficam aqui, para que notebook, CLI e app usem exatamente a mesma definição.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CLASSES: tuple[str, ...] = ("Fundação", "Alvenaria", "Cobertura", "Revestimento Externo")

EXTENSOES_IMAGEM: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

SEED_PADRAO = 42

# Caminho do modelo usado pelo app; pode ser sobrescrito pela variável de ambiente.
VARIAVEL_AMBIENTE_MODELO = "OBRA_MODELO"
CAMINHO_MODELO_PADRAO = Path("models/best.pt")

# Pesos publicados como asset do último GitHub Release do repositório.
URL_MODELO_RELEASE = (
    "https://github.com/FredLisboa77/Acompanhamento-de-Obra/releases/latest/download/best.pt"
)


@dataclass(frozen=True)
class ConfigDataset:
    """Parâmetros de preparação do dataset."""

    fracao_validacao: float = 0.2
    aumentos_por_imagem: int = 2
    seed: int = SEED_PADRAO

    def __post_init__(self) -> None:
        if not 0 < self.fracao_validacao < 1:
            raise ValueError("fracao_validacao deve estar entre 0 e 1 (exclusivo).")
        if self.aumentos_por_imagem < 0:
            raise ValueError("aumentos_por_imagem não pode ser negativo.")


@dataclass(frozen=True)
class ConfigTreino:
    """Hiperparâmetros de treino do YOLOv8 de classificação."""

    modelo_base: str = "yolov8n-cls.pt"
    epocas: int = 50
    tamanho_imagem: int = 224
    paciencia: int = 15
    seed: int = SEED_PADRAO
    projeto: Path = Path("runs/classify")
    nome: str = "train"


def caminho_modelo() -> Path:
    """Retorna o caminho do modelo, respeitando a variável de ambiente ``OBRA_MODELO``."""
    return Path(os.environ.get(VARIAVEL_AMBIENTE_MODELO, CAMINHO_MODELO_PADRAO))

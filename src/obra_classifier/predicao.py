"""Carregamento do modelo treinado e classificação de imagens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Predicao:
    """Resultado da classificação de uma imagem."""

    classe: str
    confianca: float
    probabilidades: dict[str, float]


def carregar_modelo(caminho: Path) -> Any:
    """Carrega os pesos YOLO. Lança ``FileNotFoundError`` com mensagem clara se faltar."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{caminho}'. Treine o modelo ou defina a "
            "variável de ambiente OBRA_MODELO com o caminho do best.pt."
        )
    from ultralytics import YOLO

    return YOLO(str(caminho))


def classificar(modelo: Any, imagem: Any) -> Predicao:
    """Classifica uma imagem (PIL, ndarray ou caminho) e devolve a fase mais provável."""
    resultado = modelo(imagem, verbose=False)[0]
    probs = [float(p) for p in resultado.probs.data.tolist()]
    nomes = resultado.names
    top1 = int(resultado.probs.top1)
    return Predicao(
        classe=nomes[top1],
        confianca=probs[top1],
        probabilidades={nomes[i]: p for i, p in enumerate(probs)},
    )

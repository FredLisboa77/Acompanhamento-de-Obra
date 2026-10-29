"""Carregamento do modelo treinado e classificação de imagens."""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obra_classifier.config import URL_MODELO_RELEASE


@dataclass(frozen=True)
class Predicao:
    """Resultado da classificação de uma imagem."""

    classe: str
    confianca: float
    probabilidades: dict[str, float]


def baixar_modelo(destino: Path, url: str = URL_MODELO_RELEASE, timeout_s: int = 60) -> Path:
    """Baixa os pesos (por padrão, do GitHub Release) para ``destino``.

    O download vai para um arquivo temporário e só é renomeado ao final, para que
    uma falha no meio não deixe um ``best.pt`` truncado no lugar.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resposta, temporario.open("wb") as f:
            shutil.copyfileobj(resposta, f)
        temporario.replace(destino)
    finally:
        temporario.unlink(missing_ok=True)
    return destino


def carregar_modelo(caminho: Path) -> Any:
    """Carrega os pesos YOLO. Lança ``FileNotFoundError`` com mensagem clara se faltar."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{caminho}'. Treine o modelo, baixe-o do GitHub "
            "Release (obra-classifier baixar-modelo) ou defina a variável de ambiente "
            "OBRA_MODELO com o caminho do best.pt."
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

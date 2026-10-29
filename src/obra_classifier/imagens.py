"""Leitura e escrita de imagens com suporte a caminhos não-ASCII.

``cv2.imread``/``cv2.imwrite`` falham silenciosamente no Windows com caminhos que
contêm acentos (ex.: ``Fundação``). Lendo os bytes com NumPy e decodificando em
memória, o comportamento fica igual em qualquer sistema operacional.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from obra_classifier.config import EXTENSOES_IMAGEM


def listar_imagens(pasta: Path) -> list[Path]:
    """Lista as imagens de uma pasta (não recursivo), em ordem determinística."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        return []
    return sorted(
        p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM
    )


def ler_imagem_rgb(caminho: Path) -> np.ndarray | None:
    """Lê uma imagem em RGB. Retorna ``None`` se o arquivo estiver corrompido/ilegível."""
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
    except OSError:
        return None
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        return None
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def salvar_imagem_rgb(imagem: np.ndarray, caminho: Path) -> None:
    """Salva uma imagem RGB no formato indicado pela extensão do caminho."""
    caminho = Path(caminho)
    ok, buffer = cv2.imencode(caminho.suffix, cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Falha ao codificar a imagem: {caminho}")
    buffer.tofile(str(caminho))

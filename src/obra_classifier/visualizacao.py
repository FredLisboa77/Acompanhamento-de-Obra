"""Visualizações auxiliares para inspeção do dataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from obra_classifier.config import CLASSES
from obra_classifier.imagens import ler_imagem_rgb, listar_imagens


def exibir_amostras(
    raiz: Path, classes: Sequence[str] = CLASSES, amostras_por_classe: int = 3
) -> Figure:
    """Monta uma grade com as primeiras imagens de cada classe para conferência visual."""
    fig, eixos = plt.subplots(
        len(classes),
        amostras_por_classe,
        figsize=(3 * amostras_por_classe, 3 * len(classes)),
        squeeze=False,
    )
    for linha, classe in enumerate(classes):
        imagens = listar_imagens(Path(raiz) / classe)
        for coluna in range(amostras_por_classe):
            eixo = eixos[linha][coluna]
            eixo.axis("off")
            if coluna < len(imagens):
                imagem = ler_imagem_rgb(imagens[coluna])
                if imagem is not None:
                    eixo.imshow(imagem)
            if coluna == 0:
                eixo.set_title(f"{classe} ({len(imagens)} fotos)", loc="left")
    fig.tight_layout()
    return fig

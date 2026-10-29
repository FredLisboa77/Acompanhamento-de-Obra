"""Inspeção, divisão e preparação do dataset no formato esperado pelo YOLOv8-cls.

Estrutura gerada::

    destino/
        train/<classe>/   originais de treino + variações aumentadas
        val/<classe>/     apenas originais (nunca aumentadas)

A divisão treino/validação é feita **antes** do augmentation. Se as variações de
uma mesma foto caíssem em treino e validação ao mesmo tempo, a métrica de
validação seria inflada artificialmente (vazamento de dados).
"""

from __future__ import annotations

import logging
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from obra_classifier.config import CLASSES, ConfigDataset
from obra_classifier.imagens import ler_imagem_rgb, listar_imagens, salvar_imagem_rgb

logger = logging.getLogger(__name__)

Transformacao = Callable[..., dict]


@dataclass
class ResumoDataset:
    """Contagem de imagens geradas por classe e por partição."""

    treino: dict[str, int] = field(default_factory=dict)
    validacao: dict[str, int] = field(default_factory=dict)
    aumentadas: dict[str, int] = field(default_factory=dict)
    ignoradas: list[Path] = field(default_factory=list)


def contar_por_classe(raiz: Path, classes: Sequence[str] = CLASSES) -> dict[str, int]:
    """Conta quantas imagens existem em cada subpasta de classe."""
    return {classe: len(listar_imagens(Path(raiz) / classe)) for classe in classes}


def dividir_treino_validacao(
    imagens: Sequence[Path], fracao_validacao: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Embaralha de forma reprodutível e separa as imagens em treino e validação.

    Garante ao menos uma imagem de validação quando a classe tem 2 ou mais imagens.
    """
    embaralhadas = list(imagens)
    random.Random(seed).shuffle(embaralhadas)
    if len(embaralhadas) < 2:
        return embaralhadas, []
    n_val = max(1, round(len(embaralhadas) * fracao_validacao))
    return embaralhadas[n_val:], embaralhadas[:n_val]


def _validar_destino(origem: Path, destino: Path) -> None:
    origem, destino = origem.resolve(), destino.resolve()
    if destino == origem or destino in origem.parents:
        raise ValueError(
            f"O destino '{destino}' não pode ser a origem nem conter a origem '{origem}'."
        )


def preparar_dataset(
    origem: Path,
    destino: Path,
    transformacao: Transformacao | None,
    classes: Sequence[str] = CLASSES,
    config: ConfigDataset = ConfigDataset(),
) -> ResumoDataset:
    """Divide o dataset bruto em treino/validação e aplica augmentation só no treino.

    O diretório ``destino`` é recriado do zero a cada execução.
    """
    origem, destino = Path(origem), Path(destino)
    _validar_destino(origem, destino)
    if destino.exists():
        shutil.rmtree(destino)

    resumo = ResumoDataset()
    for indice, classe in enumerate(classes):
        imagens = listar_imagens(origem / classe)
        if not imagens:
            logger.warning("Nenhuma imagem encontrada para a classe '%s'.", classe)
        # Semente distinta por classe, mas determinística.
        treino, validacao = dividir_treino_validacao(
            imagens, config.fracao_validacao, config.seed + indice
        )

        pasta_treino = destino / "train" / classe
        pasta_val = destino / "val" / classe
        pasta_treino.mkdir(parents=True, exist_ok=True)
        pasta_val.mkdir(parents=True, exist_ok=True)

        resumo.validacao[classe] = _copiar_validas(validacao, pasta_val, resumo.ignoradas)
        resumo.treino[classe] = 0
        resumo.aumentadas[classe] = 0

        for caminho in tqdm(treino, desc=classe, leave=False):
            imagem = ler_imagem_rgb(caminho)
            if imagem is None:
                logger.warning("Imagem ilegível ignorada: %s", caminho)
                resumo.ignoradas.append(caminho)
                continue
            shutil.copy2(caminho, pasta_treino / caminho.name)
            resumo.treino[classe] += 1

            if transformacao is None:
                continue
            for j in range(config.aumentos_por_imagem):
                aumentada: np.ndarray = transformacao(image=imagem)["image"]
                salvar_imagem_rgb(aumentada, pasta_treino / f"aug{j}_{caminho.name}")
                resumo.aumentadas[classe] += 1

    return resumo


def _copiar_validas(imagens: Sequence[Path], pasta: Path, ignoradas: list[Path]) -> int:
    copiadas = 0
    for caminho in imagens:
        if ler_imagem_rgb(caminho) is None:
            logger.warning("Imagem ilegível ignorada: %s", caminho)
            ignoradas.append(caminho)
            continue
        shutil.copy2(caminho, pasta / caminho.name)
        copiadas += 1
    return copiadas

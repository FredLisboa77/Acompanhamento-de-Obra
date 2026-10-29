"""Treino, avaliação e exportação do modelo YOLOv8 de classificação.

O ``ultralytics`` é importado dentro das funções: assim o restante do pacote
(e os testes) funciona sem instalar PyTorch.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from obra_classifier.config import CAMINHO_MODELO_PADRAO, ConfigTreino

logger = logging.getLogger(__name__)


def _verificar_estrutura(dados: Path) -> None:
    faltando = [p for p in ("train", "val") if not (dados / p).is_dir()]
    if faltando:
        raise FileNotFoundError(
            f"Pastas {faltando} não encontradas em '{dados}'. "
            "Execute a preparação do dataset (preparar_dataset) antes de treinar."
        )


def treinar(dados: Path, config: ConfigTreino = ConfigTreino()) -> Path:
    """Treina o modelo e retorna o caminho do ``best.pt`` gerado."""
    from ultralytics import YOLO

    dados = Path(dados)
    _verificar_estrutura(dados)

    modelo = YOLO(config.modelo_base)
    modelo.train(
        data=str(dados),
        epochs=config.epocas,
        imgsz=config.tamanho_imagem,
        patience=config.paciencia,
        seed=config.seed,
        deterministic=True,
        project=str(config.projeto.resolve()),
        name=config.nome,
        exist_ok=True,  # evita criar train2, train3... ao re-treinar
    )

    melhor = Path(modelo.trainer.save_dir) / "weights" / "best.pt"
    if not melhor.is_file():
        raise FileNotFoundError(f"Treino concluído, mas best.pt não foi gerado em {melhor}.")
    logger.info("Melhor modelo salvo em %s", melhor)
    return melhor


def avaliar(pesos: Path, dados: Path, tamanho_imagem: int = 224) -> dict[str, float]:
    """Avalia o modelo na partição de validação e retorna as acurácias top-1 e top-5."""
    from ultralytics import YOLO

    metricas = YOLO(str(pesos)).val(data=str(dados), imgsz=tamanho_imagem, split="val")
    return {"top1": float(metricas.top1), "top5": float(metricas.top5)}


def exportar_modelo(pesos: Path, destino: Path = CAMINHO_MODELO_PADRAO) -> Path:
    """Copia os pesos treinados para o caminho de onde o app os carrega."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pesos, destino)
    return destino

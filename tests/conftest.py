from pathlib import Path

import numpy as np
import pytest

from obra_classifier.imagens import salvar_imagem_rgb


@pytest.fixture
def dataset_bruto(tmp_path: Path) -> Path:
    """Dataset sintético: 2 classes (uma com acento) com 5 imagens + 1 arquivo corrompido."""
    raiz = tmp_path / "bruto"
    rng = np.random.default_rng(0)
    for classe in ("Fundação", "Alvenaria"):
        pasta = raiz / classe
        pasta.mkdir(parents=True)
        for i in range(5):
            imagem = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            salvar_imagem_rgb(imagem, pasta / f"{i}.jpg")
        (pasta / "notas.txt").write_text("não é imagem")
    (raiz / "Alvenaria" / "corrompida.jpg").write_bytes(b"isto nao e um jpeg")
    return raiz

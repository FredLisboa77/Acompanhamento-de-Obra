from pathlib import Path
from types import SimpleNamespace

import pytest

from obra_classifier.predicao import carregar_modelo, classificar


class _ModeloFalso:
    """Imita a interface de resultado do YOLO de classificação."""

    def __call__(self, imagem, verbose=False):
        probs = SimpleNamespace(data=SimpleNamespace(tolist=lambda: [0.1, 0.7, 0.2]), top1=1)
        return [SimpleNamespace(probs=probs, names={0: "Alvenaria", 1: "Cobertura", 2: "Fundação"})]


def test_classificar_retorna_classe_dominante() -> None:
    predicao = classificar(_ModeloFalso(), imagem=None)
    assert predicao.classe == "Cobertura"
    assert predicao.confianca == pytest.approx(0.7)
    assert predicao.probabilidades == {"Alvenaria": 0.1, "Cobertura": 0.7, "Fundação": 0.2}


def test_carregar_modelo_inexistente_explica_o_erro(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OBRA_MODELO"):
        carregar_modelo(tmp_path / "best.pt")

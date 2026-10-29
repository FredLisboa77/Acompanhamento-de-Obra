from pathlib import Path

import pytest

from obra_classifier.augmentation import criar_pipeline
from obra_classifier.config import ConfigDataset
from obra_classifier.dataset import (
    contar_por_classe,
    dividir_treino_validacao,
    preparar_dataset,
)
from obra_classifier.imagens import ler_imagem_rgb, listar_imagens

CLASSES_TESTE = ("Fundação", "Alvenaria")


def test_listar_imagens_ignora_arquivos_que_nao_sao_imagem(dataset_bruto: Path) -> None:
    nomes = [p.name for p in listar_imagens(dataset_bruto / "Fundação")]
    assert nomes == ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg"]


def test_listar_imagens_em_pasta_inexistente_retorna_vazio(tmp_path: Path) -> None:
    assert listar_imagens(tmp_path / "nao_existe") == []


def test_ler_imagem_corrompida_retorna_none(dataset_bruto: Path) -> None:
    assert ler_imagem_rgb(dataset_bruto / "Alvenaria" / "corrompida.jpg") is None


def test_contar_por_classe(dataset_bruto: Path) -> None:
    assert contar_por_classe(dataset_bruto, CLASSES_TESTE) == {"Fundação": 5, "Alvenaria": 6}


def test_divisao_e_disjunta_reprodutivel_e_tem_validacao() -> None:
    imagens = [Path(f"{i}.jpg") for i in range(10)]
    treino, val = dividir_treino_validacao(imagens, 0.2, seed=1)
    assert len(val) == 2 and len(treino) == 8
    assert set(treino).isdisjoint(val)
    assert (treino, val) == dividir_treino_validacao(imagens, 0.2, seed=1)


def test_divisao_com_uma_imagem_vai_toda_para_treino() -> None:
    assert dividir_treino_validacao([Path("a.jpg")], 0.2, seed=1) == ([Path("a.jpg")], [])


def test_preparar_dataset_nao_vaza_aumentos_para_validacao(
    dataset_bruto: Path, tmp_path: Path
) -> None:
    destino = tmp_path / "processado"
    config = ConfigDataset(fracao_validacao=0.2, aumentos_por_imagem=2, seed=7)
    resumo = preparar_dataset(dataset_bruto, destino, criar_pipeline(7), CLASSES_TESTE, config)

    for classe in CLASSES_TESTE:
        treino = listar_imagens(destino / "train" / classe)
        val = listar_imagens(destino / "val" / classe)
        assert not any(p.name.startswith("aug") for p in val)
        originais_treino = {p.name for p in treino if not p.name.startswith("aug")}
        assert originais_treino.isdisjoint({p.name for p in val})
        assert len(treino) == resumo.treino[classe] + resumo.aumentadas[classe]
        assert resumo.aumentadas[classe] == 2 * resumo.treino[classe]

    assert [p.name for p in resumo.ignoradas] == ["corrompida.jpg"]


def test_preparar_dataset_recria_destino(dataset_bruto: Path, tmp_path: Path) -> None:
    destino = tmp_path / "processado"
    (destino / "train" / "Lixo").mkdir(parents=True)
    preparar_dataset(dataset_bruto, destino, None, CLASSES_TESTE)
    assert not (destino / "train" / "Lixo").exists()


def test_preparar_dataset_recusa_destino_que_contem_origem(dataset_bruto: Path) -> None:
    with pytest.raises(ValueError):
        preparar_dataset(dataset_bruto, dataset_bruto.parent, None, CLASSES_TESTE)


@pytest.mark.parametrize("kwargs", [{"fracao_validacao": 0}, {"aumentos_por_imagem": -1}])
def test_config_dataset_invalida(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ConfigDataset(**kwargs)

"""Interface de linha de comando.

Exemplos::

    obra-classifier preparar --origem "Dataset_Bruto" --destino data/processado
    obra-classifier treinar --dados data/processado --epocas 50
    obra-classifier avaliar --dados data/processado
    obra-classifier prever foto.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from obra_classifier.config import ConfigDataset, ConfigTreino, caminho_modelo


def _preparar(args: argparse.Namespace) -> None:
    from obra_classifier.augmentation import criar_pipeline
    from obra_classifier.dataset import preparar_dataset

    config = ConfigDataset(
        fracao_validacao=args.fracao_validacao,
        aumentos_por_imagem=args.aumentos,
        seed=args.seed,
    )
    resumo = preparar_dataset(args.origem, args.destino, criar_pipeline(args.seed), config=config)
    for classe in resumo.treino:
        print(
            f"{classe:<22} treino={resumo.treino[classe]:>4} "
            f"(+{resumo.aumentadas[classe]} aumentadas)  val={resumo.validacao[classe]:>4}"
        )
    if resumo.ignoradas:
        print(f"{len(resumo.ignoradas)} imagem(ns) ilegível(is) ignorada(s).")


def _treinar(args: argparse.Namespace) -> None:
    from obra_classifier.treino import exportar_modelo, treinar

    config = ConfigTreino(epocas=args.epocas, tamanho_imagem=args.imgsz, seed=args.seed)
    melhor = treinar(args.dados, config)
    print(f"Modelo exportado para: {exportar_modelo(melhor, args.saida)}")


def _avaliar(args: argparse.Namespace) -> None:
    from obra_classifier.treino import avaliar

    metricas = avaliar(args.modelo, args.dados, args.imgsz)
    print(f"Acurácia top-1: {metricas['top1']:.3f} | top-5: {metricas['top5']:.3f}")


def _prever(args: argparse.Namespace) -> None:
    from obra_classifier.predicao import carregar_modelo, classificar

    modelo = carregar_modelo(args.modelo)
    for imagem in args.imagens:
        predicao = classificar(modelo, str(imagem))
        print(f"{imagem}: {predicao.classe} ({predicao.confianca:.1%})")


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="obra-classifier", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("preparar", help="Divide treino/val e aplica augmentation no treino.")
    p.add_argument("--origem", type=Path, required=True, help="Pasta com uma subpasta por classe.")
    p.add_argument("--destino", type=Path, default=Path("data/processado"))
    p.add_argument("--fracao-validacao", type=float, default=ConfigDataset.fracao_validacao)
    p.add_argument("--aumentos", type=int, default=ConfigDataset.aumentos_por_imagem)
    p.add_argument("--seed", type=int, default=ConfigDataset.seed)
    p.set_defaults(func=_preparar)

    p = sub.add_parser("treinar", help="Treina o YOLOv8-cls e exporta o best.pt.")
    p.add_argument("--dados", type=Path, default=Path("data/processado"))
    p.add_argument("--epocas", type=int, default=ConfigTreino.epocas)
    p.add_argument("--imgsz", type=int, default=ConfigTreino.tamanho_imagem)
    p.add_argument("--seed", type=int, default=ConfigTreino.seed)
    p.add_argument("--saida", type=Path, default=caminho_modelo())
    p.set_defaults(func=_treinar)

    p = sub.add_parser("avaliar", help="Calcula a acurácia na partição de validação.")
    p.add_argument("--dados", type=Path, default=Path("data/processado"))
    p.add_argument("--modelo", type=Path, default=caminho_modelo())
    p.add_argument("--imgsz", type=int, default=ConfigTreino.tamanho_imagem)
    p.set_defaults(func=_avaliar)

    p = sub.add_parser("prever", help="Classifica uma ou mais imagens.")
    p.add_argument("imagens", type=Path, nargs="+")
    p.add_argument("--modelo", type=Path, default=caminho_modelo())
    p.set_defaults(func=_prever)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = criar_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

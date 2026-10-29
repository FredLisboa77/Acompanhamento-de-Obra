"""Pipeline de Data Augmentation (Albumentations).

As transformações simulam variações comuns em canteiros de obra: iluminação,
sombras fortes, fotos levemente desfocadas e ângulos espelhados.
"""

from __future__ import annotations

import albumentations as A

from obra_classifier.config import SEED_PADRAO


def criar_pipeline(seed: int = SEED_PADRAO) -> A.Compose:
    """Cria o pipeline de augmentation com semente fixa (resultados reprodutíveis)."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomGamma(p=0.3),
            A.Blur(blur_limit=(3, 3), p=0.2),
            A.CLAHE(p=0.2),  # realça contraste em fotos com muita sombra
        ],
        seed=seed,
    )

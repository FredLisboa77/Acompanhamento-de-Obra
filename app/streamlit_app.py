"""Interface Streamlit para classificar a fase de uma obra a partir de uma foto.

Execução::

    streamlit run app/streamlit_app.py

O caminho do modelo vem da variável de ambiente ``OBRA_MODELO`` (padrão: models/best.pt).
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from obra_classifier.config import caminho_modelo
from obra_classifier.predicao import Predicao, carregar_modelo, classificar

IMAGENS_EXEMPLO = {
    "Exemplo Fundação": "https://images.unsplash.com/photo-1541888946425-d81bb19480c5?q=80&w=500",
    "Exemplo Alvenaria": "https://images.unsplash.com/photo-1584622650111-993a426fbf0a?q=80&w=500",
    "Exemplo Cobertura": "https://images.unsplash.com/photo-1632759145351-1d592919f522?q=80&w=500",
}
TIMEOUT_DOWNLOAD_S = 10


@st.cache_resource
def obter_modelo():
    return carregar_modelo(caminho_modelo())


@st.cache_data(show_spinner="Baixando imagem de exemplo...")
def baixar_imagem(url: str) -> bytes:
    resposta = requests.get(url, timeout=TIMEOUT_DOWNLOAD_S)
    resposta.raise_for_status()
    return resposta.content


def selecionar_imagem() -> Image.Image | None:
    """Retorna a imagem enviada pelo usuário ou o exemplo escolhido na barra lateral."""
    enviada = st.sidebar.file_uploader("Envie uma foto da obra", type=["jpg", "jpeg", "png"])

    st.sidebar.markdown("---")
    st.sidebar.write("### Teste rápido")
    exemplo = st.sidebar.selectbox("Escolha um exemplo:", ["Nenhum", *IMAGENS_EXEMPLO])

    try:
        if enviada is not None:
            return Image.open(enviada).convert("RGB")
        if exemplo != "Nenhum":
            return Image.open(BytesIO(baixar_imagem(IMAGENS_EXEMPLO[exemplo]))).convert("RGB")
    except (requests.RequestException, OSError) as erro:
        st.error(f"Não foi possível carregar a imagem: {erro}")
    return None


def exibir_resultado(imagem: Image.Image, predicao: Predicao) -> None:
    col_imagem, col_resultado = st.columns(2)
    with col_imagem:
        st.image(imagem, caption="Imagem analisada", width="stretch")
    with col_resultado:
        st.write("### Resultado da predição")
        st.bar_chart(pd.Series(predicao.probabilidades, name="Probabilidade"))
        st.success(f"Fase identificada: **{predicao.classe}** ({predicao.confianca:.1%})")


def main() -> None:
    st.set_page_config(page_title="IA de Monitoramento de Obras", layout="wide")
    st.title("🏗️ Classificador de Fases de Obra")
    st.write("Sistema de Visão Computacional para Loteamentos")

    try:
        modelo = obter_modelo()
    except FileNotFoundError as erro:
        st.error(str(erro))
        st.stop()

    imagem = selecionar_imagem()
    if imagem is not None:
        exibir_resultado(imagem, classificar(modelo, imagem))


main()

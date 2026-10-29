# Acompanhamento de Obra

Solução prática e automatizada para o monitoramento da evolução física de obras civis. A solução visa classificar imagens em diferentes estágios construtivos, auxiliando na precisão técnica necessária para liberações de crédito em financiamentos imobiliários, mitigando a subjetividade inerente às inspeções manuais remotas.

Classificador **YOLOv8-cls** de 4 fases: **Fundação, Alvenaria, Cobertura e Revestimento Externo**.

## Estrutura

```
├── app/streamlit_app.py          # interface web (upload + gráfico de probabilidades)
├── notebooks/                    # notebook do trabalho (orquestra o pacote no Colab)
├── src/obra_classifier/
│   ├── config.py                 # classes, hiperparâmetros, seed e caminhos
│   ├── imagens.py                # leitura/escrita de imagens (caminhos acentuados)
│   ├── dataset.py                # inspeção, divisão treino/val e preparação
│   ├── augmentation.py           # pipeline Albumentations
│   ├── treino.py                 # treino, avaliação e exportação do best.pt
│   ├── predicao.py               # carregamento do modelo e classificação
│   ├── visualizacao.py           # grade de amostras do dataset
│   └── cli.py                    # linha de comando `obra-classifier`
└── tests/                        # testes (pytest), sem dependência de PyTorch
```

## Uso no Google Colab

1. Abra [`notebooks/Trabalho_de_Visao_Computacional.ipynb`](notebooks/Trabalho_de_Visao_Computacional.ipynb) no Colab e selecione a GPU T4.
2. Execute as células em ordem. O dataset bruto deve estar no Drive em
   `MyDrive/Trabalho - Visão Computacional/Dataset_Bruto/<classe>/`.
3. O treino salva o `best.pt` em `models/` e uma cópia no Drive. Nas próximas sessões, pule a célula 1.4.
4. A célula 2.3 publica a interface e imprime um link `*.trycloudflare.com`.

## Uso local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[train,app,dev]"

obra-classifier preparar --origem caminho/Dataset_Bruto --destino data/processado
obra-classifier treinar  --dados data/processado          # gera models/best.pt
obra-classifier avaliar  --dados data/processado
obra-classifier prever   foto1.jpg foto2.jpg

streamlit run app/streamlit_app.py                         # usa models/best.pt
```

O caminho do modelo pode ser alterado com a variável de ambiente `OBRA_MODELO`.

## Decisões de projeto

- **Sem vazamento de dados:** a divisão treino/validação acontece **antes** do augmentation, e as variações sintéticas existem só no treino. Na versão original, o YOLO dividia a pasta já aumentada, então cópias da mesma foto caíam nos dois conjuntos e a validação marcava 100% de acurácia.
- **Reprodutibilidade:** a mesma seed (`config.SEED_PADRAO`) controla a divisão, o augmentation e o treino.
- **Código testável:** a lógica fica em módulos Python e o notebook só os chama. Os testes rodam no CI a cada push.

## Desenvolvimento

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

Imagens, pesos (`*.pt`) e documentos `.docx` não são versionados (veja `.gitignore`).

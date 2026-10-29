# Acompanhamento de Obra

[![CI](https://github.com/FredLisboa77/Acompanhamento-de-Obra/actions/workflows/ci.yml/badge.svg)](https://github.com/FredLisboa77/Acompanhamento-de-Obra/actions/workflows/ci.yml)
[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredLisboa77/Acompanhamento-de-Obra/blob/main/notebooks/Trabalho_de_Visao_Computacional.ipynb)

Solução prática e automatizada para o monitoramento da evolução física de obras civis. A solução visa classificar imagens em diferentes estágios construtivos, auxiliando na precisão técnica necessária para liberações de crédito em financiamentos imobiliários, mitigando a subjetividade inerente às inspeções manuais remotas.

O modelo é um classificador **YOLOv8-cls** que reconhece 4 fases de obra: **Fundação, Alvenaria, Cobertura e Revestimento Externo**.

> Projeto acadêmico de Visão Computacional, de Carlos Frederico Coelho Lisboa.

Tudo fica neste repositório: código, dataset, notebook e modelo treinado (em [Releases](https://github.com/FredLisboa77/Acompanhamento-de-Obra/releases)). **Não é preciso Google Drive nem nenhuma conta além do Colab.**

## Início rápido (Google Colab)

1. Clique em **Abrir no Colab** acima.
2. Em *Ambiente de execução → Alterar o tipo de ambiente de execução*, selecione **GPU T4**.
3. Execute as células em ordem. A primeira clona este repositório e instala as dependências.
4. **Treino (célula 1.4):** é opcional. Se você pular, a célula 2.1 baixa o modelo já treinado do último Release.
5. A célula 2.3 publica a interface web e imprime um link `*.trycloudflare.com`. Abra o link e envie uma foto. Há fotos de teste em [`data/teste/`](data/teste).

## Como funciona

```
data/bruto/<fase>/  →  divisão treino/val  →  augmentation (só treino)  →  YOLOv8n-cls  →  models/best.pt  →  app Streamlit
```

1. **Dataset:** 269 fotos organizadas em uma pasta por fase, em `data/bruto/` (Fundação 60, Alvenaria 64, Cobertura 74, Revestimento Externo 71). Antes do treino, um hash perceptual procura fotos duplicadas, mesmo redimensionadas ou recomprimidas.
2. **Divisão:** 80% das fotos de cada fase vão para treino e 20% para validação, com seed fixa.
3. **Data augmentation (Albumentations):** brilho/contraste, gama, espelhamento, desfoque e CLAHE geram 2 variações de cada foto de **treino**, simulando iluminação e ângulos diferentes no canteiro.
4. **Treino:** *transfer learning* a partir do YOLOv8 Nano pré-treinado no ImageNet, com *early stopping*.
5. **Avaliação:** acurácia top-1 e matriz de confusão, medidas só em fotos originais que o modelo não viu no treino.
6. **Interface:** app Streamlit com upload de foto, gráfico de probabilidades por fase e a fase identificada.

## Estrutura do repositório

```
├── data/
│   ├── bruto/<fase>/            # dataset rotulado (uma pasta por fase)
│   └── teste/                   # fotos sem rótulo para testar a interface
├── notebooks/
│   └── Trabalho_de_Visao_Computacional.ipynb   # notebook do trabalho (Colab)
├── app/streamlit_app.py         # interface web
├── src/obra_classifier/         # pacote Python com toda a lógica
│   ├── config.py                #   classes, hiperparâmetros, seed e caminhos
│   ├── imagens.py               #   leitura/escrita de imagens (caminhos acentuados)
│   ├── dataset.py               #   inspeção, divisão treino/val e preparação
│   ├── augmentation.py          #   pipeline Albumentations
│   ├── treino.py                #   treino, avaliação e exportação do best.pt
│   ├── predicao.py              #   download, carregamento e uso do modelo
│   ├── visualizacao.py          #   grade de amostras do dataset
│   └── cli.py                   #   comando `obra-classifier`
├── tests/                       # testes automatizados (pytest)
└── .github/workflows/ci.yml     # CI: lint + testes a cada push/PR
```

Arquivos gerados (`data/processado/`, `runs/`, `models/*.pt`) não são versionados.

## Uso local

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/FredLisboa77/Acompanhamento-de-Obra.git
cd Acompanhamento-de-Obra
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[train,app]"
```

**Só testar o modelo publicado:**

```bash
obra-classifier baixar-modelo                     # baixa models/best.pt do último Release
obra-classifier prever data/teste/1.jpeg data/teste/2.jpeg
streamlit run app/streamlit_app.py                # interface em http://localhost:8501
```

**Treinar do zero:**

```bash
obra-classifier preparar                          # data/bruto -> data/processado (train/val)
obra-classifier treinar                           # gera models/best.pt (use --epocas para ajustar)
obra-classifier avaliar                           # acurácia na validação
```

Use `obra-classifier <comando> --help` para ver todas as opções. O app lê o modelo de `models/best.pt`; para usar outro arquivo, defina a variável de ambiente `OBRA_MODELO`.

## Versionamento do modelo

Os pesos treinados (`best.pt`, ~3 MB) não ficam no histórico do Git: são publicados como asset de um **GitHub Release**. O notebook e o comando `baixar-modelo` usam sempre o link do Release mais recente:

```
https://github.com/FredLisboa77/Acompanhamento-de-Obra/releases/latest/download/best.pt
```

Para publicar um modelo novo após treinar (com o [GitHub CLI](https://cli.github.com/)):

```bash
gh release create v1.1.0 models/best.pt --title "Modelo v1.1.0" --notes "Descreva o que mudou e a acurácia obtida."
```

## Como ampliar o dataset

Adicione fotos `.jpg`, `.jpeg`, `.png` ou `.webp` na pasta da fase correspondente em `data/bruto/`. As pastas precisam se chamar exatamente `Fundação`, `Alvenaria`, `Cobertura` e `Revestimento Externo`. Depois:

1. Verifique duplicatas (a célula 1.2 do notebook faz isso) e remova uma foto de cada par apontado.
2. Faça commit, treine de novo e publique um novo Release.

## Decisões de projeto

- **Sem duplicatas:** 13 pares de fotos repetidas foram removidos do dataset original (282 → 269). Cópias da mesma foto em treino e validação inflariam a métrica.
- **Sem vazamento de dados:** a divisão treino/validação acontece **antes** do augmentation, e as variações sintéticas existem só no treino. Se cópias da mesma foto caíssem nos dois conjuntos, a validação mediria memorização em vez de generalização. Isso aconteceu na primeira versão do projeto e fazia a acurácia aparecer como 100%.
- **Reprodutibilidade:** a mesma seed (`config.SEED_PADRAO`) controla a divisão, o augmentation e o treino.
- **Lógica fora do notebook:** o código fica em módulos Python testados, e o notebook só os chama. Assim, notebook, linha de comando e app usam exatamente o mesmo código.

## Limitações

- **Dataset pequeno** (60 a 74 fotos por fase): o augmentation ajuda, mas não substitui fotos reais de obras distintas. Parte das imagens vem da internet e inclui ilustrações e banners, não só fotos de canteiro.
- **Fases que se sobrepõem:** a transição entre alvenaria e revestimento é gradual, e uma foto pode mostrar as duas fases. O modelo escolhe só uma.
- **Oclusão, iluminação e ângulo:** tapumes, andaimes, contraluz e fotos aéreas ou muito próximas reduzem a confiabilidade.
- **Uso em decisões de crédito:** a saída é um **apoio** à vistoria, acompanhado da confiança da predição, não um laudo.

## Desenvolvimento

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

O CI roda essas mesmas verificações a cada push no `main` e em todo pull request.

# Passo a passo para rodar o projeto

Este guia mostra como executar o classificador de fases de obra, do início até a interface web onde você envia uma foto e vê a fase identificada.

Há dois caminhos:

- **[Opção A: Google Colab](#opção-a--google-colab-recomendado)** (recomendado). Roda no navegador e não exige instalar nada. Basta uma conta Google.
- **[Opção B: no seu computador](#opção-b--no-seu-computador)**. Para quem quer rodar localmente ou alterar o código.

Você **não precisa** de acesso ao Google Drive nem de nenhuma senha. O código, as fotos e o modelo treinado estão todos neste repositório do GitHub.

---

## Opção A — Google Colab (recomendado)

Tempo estimado: **5 a 10 minutos**.

### 1. Abrir o notebook

1. Acesse o repositório: <https://github.com/FredLisboa77/Acompanhamento-de-Obra>
2. No início do README, clique no botão **Open in Colab**.
   Se preferir, acesse direto:
   <https://colab.research.google.com/github/FredLisboa77/Acompanhamento-de-Obra/blob/main/notebooks/Trabalho_de_Visao_Computacional.ipynb>
3. Entre com sua conta Google, se for solicitado.

### 2. Ativar a GPU

1. No menu, clique em **Ambiente de execução → Alterar o tipo de ambiente de execução**.
2. Em *Acelerador de hardware*, selecione **GPU T4** e clique em **Salvar**.

A GPU só é necessária para treinar (etapa 1.4). Sem ela, o resto funciona, mas o treino fica bem mais lento.

### 3. Executar as células em ordem

Para executar uma célula, clique nela e pressione **Shift + Enter**, ou clique no botão ▶ à esquerda dela. Espere cada célula terminar antes de passar para a próxima.

| Etapa | O que faz | O que você deve ver |
|---|---|---|
| **0. Preparação do ambiente** | Baixa o repositório e instala as dependências (1 a 2 min) | `Diretório do projeto: /content/Acompanhamento-de-Obra` |
| **1.1. Conexão e Setup** | Define os caminhos das pastas | `✅ Dataset encontrado` |
| **1.2. Inspeção de Dados** | Conta as fotos, procura duplicatas e mostra exemplos | A contagem por fase, `✅ Nenhuma foto duplicada encontrada` e uma grade de fotos |
| **1.3. Divisão e Data Augmentation** | Separa treino e validação e gera as variações das fotos | Uma linha por fase, com as quantidades de treino e validação |
| **1.4. Treinamento do Modelo** | Treina o modelo (**opcional**, 2 a 5 min com GPU) | O progresso por época e `✅ Modelo salvo em ...` |
| **2.1. Carregamento do Modelo** | Carrega o modelo treinado ou baixa o modelo pronto do GitHub | `✅ Modelo carregado` |
| **2.2. Métricas na Validação** | Calcula o percentual de acerto | `Acurácia top-1: ...`, mais a matriz de confusão se você treinou na etapa 1.4 |
| **2.3. Interface e Túnel** | Abre a interface web | `🌐 Interface disponível em: https://....trycloudflare.com` |

> **Posso pular a etapa 1.4?** Sim. Se você não treinar, a etapa 2.1 baixa automaticamente o modelo já treinado do último Release no GitHub. As outras etapas (0 a 1.3) precisam ser executadas mesmo assim.

Você também pode usar **Ambiente de execução → Executar tudo** (Ctrl + F9). Assim, todas as células rodam em sequência, inclusive o treino.

### 4. Usar a interface

1. Clique no link `https://....trycloudflare.com` que aparece na saída da etapa 2.3. Ele abre em uma nova aba.
2. Na barra lateral, clique em **Browse files** e envie uma foto de obra (`.jpg`, `.jpeg` ou `.png`).
   Se preferir, escolha um dos exemplos em **Teste rápido**.
3. A tela mostra:
   - a foto analisada;
   - um gráfico com a probabilidade de cada fase;
   - a **fase identificada**, com o percentual de confiança.

**Precisa de fotos para testar?** A pasta [`data/teste/`](data/teste) do repositório tem 29 fotos de obras. Para baixar uma delas:

- **pelo GitHub:** abra a foto e clique no botão de download (↓);
- **pelo Colab:** no ícone de pasta 📁 da barra lateral esquerda, abra `Acompanhamento-de-Obra/data/teste`, clique com o botão direito na foto e escolha **Fazer download**.

O link da interface funciona enquanto a sessão do Colab estiver aberta. Se a sessão cair, execute as células de novo, a partir da etapa 0.

---

## Opção B — No seu computador

Pré-requisitos:

- **Python 3.10 ou mais recente**: <https://www.python.org/downloads/>
- **Git**: <https://git-scm.com/downloads>

### 1. Baixar o projeto e instalar

Abra um terminal (no Windows: *PowerShell* ou *Prompt de Comando*) e execute:

```bash
git clone https://github.com/FredLisboa77/Acompanhamento-de-Obra.git
cd Acompanhamento-de-Obra
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Instale o projeto. A instalação baixa o PyTorch e pode levar alguns minutos:

```bash
python -m pip install --upgrade pip
pip install -e ".[train,app]"
```

### 2. Obter o modelo

Escolha **uma** das opções:

**a) Baixar o modelo já treinado** (rápido):

```bash
obra-classifier baixar-modelo
```

**b) Treinar do zero** (10 a 20 min sem GPU):

```bash
obra-classifier preparar     # separa treino/validação e gera as variações das fotos
obra-classifier treinar      # treina e salva o modelo em models/best.pt
obra-classifier avaliar      # mostra o percentual de acerto
```

### 3. Testar

**Pelo terminal**, classificando uma ou mais fotos:

```bash
obra-classifier prever data/teste/1.jpeg data/teste/2.jpeg
```

**Pela interface web:**

```bash
streamlit run app/streamlit_app.py
```

O navegador abre em <http://localhost:8501>. Se não abrir, acesse o endereço manualmente. Para encerrar, pressione **Ctrl + C** no terminal.

---

## Problemas comuns

| Mensagem ou sintoma | Causa provável | Solução |
|---|---|---|
| `Dataset não encontrado em ...` | A etapa 0 não rodou ou falhou ao baixar o repositório | Execute a etapa 0 de novo e confira se ela terminou sem erro |
| `Modelo não encontrado em ...` | O modelo não foi treinado nem baixado | Execute a etapa 1.4 (treino) ou a 2.1 (download). No computador, use `obra-classifier baixar-modelo` |
| `ModuleNotFoundError: obra_classifier` | A etapa 0 foi pulada ou o Colab reiniciou | Execute de novo a partir da etapa 0 |
| O upload da foto falha com **erro 403** | Proteção do navegador ao usar o túnel | Na etapa 2.3, troque `DESATIVAR_XSRF = False` por `True` e execute a célula de novo |
| A etapa 2.3 fica parada em "Aguardando URL do túnel..." | Falha temporária no serviço do túnel | Interrompa a célula (botão ■) e execute de novo |
| O link `trycloudflare.com` parou de funcionar | A sessão do Colab foi encerrada | Execute as células de novo, a partir da etapa 0. Um link novo é gerado |
| O treino está muito lento | A GPU não foi ativada | Faça o passo [2. Ativar a GPU](#2-ativar-a-gpu) e execute as células de novo |
| `SSL: CERTIFICATE_VERIFY_FAILED` no `pip` (computador) | Antivírus ou rede que inspeciona conexões HTTPS | Atualize o pip (`python -m pip install --upgrade pip`) ou instale com `pip install --use-feature=truststore -e ".[train,app]"` |

---

Para entender como o projeto funciona por dentro (estrutura, decisões de projeto e limitações), veja o [README](README.md).

# Projeto de Análise de Dados e Sinais

Este repositório contém scripts desenvolvidos para analisar o dataset Iris (testando a robustez do K-Means e SVM com injeção de ruído) e também para o processamento digital de sinais em áudio (extração de momentos estatísticos e cálculo de Entropia de Shannon).

## Como executar os scripts

Para não instalar dependências globalmente no seu sistema operacional, prepare o ambiente virtual e rode os scripts com os comandos abaixo (execute todos a partir do diretório raiz do projeto):

### 1. Sincronizar e ativar o ambiente
```bash
uv sync
source .venv/bin/activate
```

### 2. Rodar a Análise do Dataset Iris
```bash
python src/analyze_iris.py
```

### 3. Rodar a Análise de Sinais de Áudio
```bash
python src/analyze_audio.py
```

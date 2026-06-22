# Atividade Avaliativa - Mestrado

Este repositório contém os scripts desenvolvidos para a atividade da disciplina ESPA04. Foram criados códigos para analisar o clássico dataset Iris (injetando *outliers* para testar a robustez do K-Means e SVM) e também códigos de processamento digital de sinais para extração de momentos estatísticos e cálculo de Entropia de Shannon em sinais de áudio.

## Estrutura do Projeto

- `src/`: Contém os scripts Python de análise (`analyze_iris.py` e `analyze_audio.py`).
- `src/images/`: Diretório onde as imagens (gráficos e plotagens) geradas pelos scripts são salvas automaticamente.
- `mds/`: Contém as transcrições e formatações de resultados em Markdown (`atividade.md`, `resultados_iris.md`).

## Como executar os scripts

Para não poluir o sistema operacional, utilize o ambiente virtual (`uv` ou `venv`) que já está configurado na raiz da pasta.

### 1. Ativar o ambiente virtual
Abra o seu terminal no diretório raiz do projeto e ative o ambiente com:
```bash
source .venv/bin/activate
```

*(Observação: as bibliotecas necessárias já foram instaladas no ambiente)*.

### 2. Rodar a Análise do Dataset Iris
Entre na pasta `src` e execute o script:
```bash
cd src
python analyze_iris.py
```
Esse script vai gerar no terminal uma tabela em Markdown com os resultados comparativos e criar a imagem `iris_dispersao_outliers.png` dentro da subpasta `src/images/`.

### 3. Rodar a Análise de Sinais de Áudio
Gere os momentos temporais e o teste T de Welch (ainda dentro da pasta `src`):
```bash
python analyze_audio.py
```
Você verá os valores dos momentos estatísticos e as métricas. As imagens `audio_sinal_tempo.png` e `audio_entropia.png` serão exportadas diretamente para a pasta `src/images/`.

# Classificação de Paisagens Naturais

Este repositório contém um notebook Jupyter e diversos arquivos auxiliares para a classificação de paisagens naturais utilizando técnicas de aprendizado de máquina e deep learning. O projeto está modularizado seguindo boas práticas de Engenharia de Software, garantindo escalabilidade, reuso de código e facilidade de manutenção.

## Estrutura do Projeto

```
Classificacao_de_paisagens/
│── Classificacao_Paisagens_Naturais.ipynb   # Notebook principal
│── config.yaml                               # Arquivo de configuração
│── dataset.py                                # Manipulação e carregamento de dados
│── dataset_utils.py                          # Funções auxiliares para datasets
│── data_processing.py                        # Processamento de dados (normalização, augmentação, etc.)
│── ensemble.py                               # Implementação de técnicas de ensemble learning
│── main.py                                   # Arquivo principal para treinamento e execução
│── models.py                                 # Definição dos modelos de deep learning
│── model_compression.py                      # Técnicas de compressão de modelos
│── optimization.py                           # Algoritmos de otimização
│── training.py                               # Rotinas de treinamento dos modelos
│── utils.py                                  # Funções auxiliares
│── visualization.py                          # Geração de gráficos e análises visuais
```

## Baixar o Dataset

[Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data)

## Local do Diretório ao Extrair

```sh
C:\Dataset\intel-image-classification - para ser usado como caminho absoluto.
- Extraia os arquivos do notebook e os coloque dentro dessa pasta.
```

## Dependências Necessárias

Para executar o notebook localmente, abra o terminal pois você precisará instalar algumas bibliotecas. Recomenda-se o uso de um ambiente virtual.

```sh
pip install -r requirements.txt
```

Caso o arquivo `requirements.txt` não esteja presente, as principais dependências incluem:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow torch torchvision pyyaml jupyter opencv-python Pillow optuna timm tqdm onnx imagehash ensemble
```

## Execução no Jupyter Notebook (Localmente)

1. Certifique-se de que todas as dependências estão instaladas.
2. Abra um terminal e navegue até a pasta do projeto.
3. Inicie o Jupyter Notebook:

```sh
jupyter notebook
```

4. No navegador, abra `Classificacao_Paisagens_Naturais.ipynb` e execute as células sequencialmente.

## Execução no Google Colab

Caso deseje rodar no Google Colab:

1. Acesse [Google Colab](https://colab.research.google.com/).
2. Faça o upload do arquivo `Classificacao_Paisagens_Naturais.ipynb`.
3. Execute a seguinte célula para instalar as dependências:

```python
!pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow torch torchvision pyyaml jupyter opencv-python Pillow optuna timm tqdm onnx imagehash ensemble
```

4. Certifique-se de que todos os arquivos auxiliares estão disponíveis no ambiente do Colab, carregando-os conforme necessário.

## O que o Notebook Faz

O notebook `Classificacao_Paisagens_Naturais.ipynb` executa as seguintes etapas:

1. **Carregamento e processamento dos dados:** Utiliza `dataset.py` e `data_processing.py` para preparar os dados.
2. **Exploração e visualização:** Com `visualization.py`, gera gráficos para entender a distribuição dos dados.
3. **Treinamento de Modelos:** Implementa redes neurais para classificação utilizando `models.py` e `training.py`.
4. **Otimização e ajuste de hiperparâmetros:** Inclui técnicas como grid search para aprimorar o desempenho.
5. **Uso de ensemble learning:** Com `ensemble.py`, combina modelos para melhorar a precisão.
6. **Compressão de modelo:** Usa `model_compression.py` para reduzir o tamanho do modelo sem perder desempenho.
7. **Avaliação de resultados:** Mede a acurácia e exibe métricas como matriz de confusão.

## Resultados Esperados

- Melhoria da precisão do modelo após aplicação de técnicas de otimização.
- Comparativo entre diferentes abordagens, incluindo redes neurais profundas e técnicas de ensemble.
- Visualização clara da classificação das paisagens naturais com gráficos interpretáveis.

## Engenharia de Software e Modularização

O projeto segue princípios de Engenharia de Software:

- **Modularização**: Cada funcionalidade está separada em arquivos específicos (exemplo: `dataset.py` para manipulação de dados, `models.py` para definição dos modelos).
- **Reuso de código**: Funções utilitárias em `utils.py` permitem evitar repetição de código.
- **Escalabilidade**: O uso de `config.yaml` permite ajustes rápidos nos hiperparâmetros sem necessidade de alterações no código.
- **Manutenção fácil**: O código está bem organizado e documentado, facilitando melhorias futuras.

## Conclusão

O projeto fornece uma abordagem completa para classificação de paisagens naturais com aprendizado de máquina, utilizando boas práticas de Engenharia de Software. O uso de diferentes modelos, otimização e técnicas de visualização torna a solução robusta e eficiente.

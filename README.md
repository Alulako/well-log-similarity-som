# Similaridade de Perfis de Poços com SOM

Prova de conceito pequena e reproduzível para avaliar se uma redução baseada em SOM preserva relações úteis de similaridade entre segmentos curtos de perfis de poços.

## Pergunta de pesquisa

Uma representação compacta 1D obtida com um Mapa Auto-Organizável (SOM) consegue preservar parte suficiente da estrutura de similaridade de segmentos multivariados de perfis de poços para permitir comparações mais simples?

O experimento foi mantido propositalmente pequeno. Ele não pretende realizar uma validação geológica ampla nem reproduzir os experimentos do grupo de pesquisa.

## Conjunto de dados

Os dados vêm de um conjunto público de classificação de fácies disponibilizado pela Society of Exploration Geophysicists (SEG) e associado ao trabalho:

Hall, B. (2016). *Facies classification using machine learning*. The Leading Edge, 35(10), 906–909. DOI: 10.1190/tle35100906.1.

O arquivo-fonte foi fixado no commit:

4885188ff29684ca2774662231f3adad6c3ec563

Arquivo utilizado:

https://raw.githubusercontent.com/seg/tutorials-2016/4885188ff29684ca2774662231f3adad6c3ec563/1610_Facies_classification/training_data.csv

O subconjunto selecionado contém:

- 12 segmentos reais;
- 16 amostras por segmento;
- 3 fácies: 2, 3 e 6;
- 4 segmentos por fácies;
- 6 poços distintos;
- 3 curvas: GR, PHIND e PE;
- espaçamento regular de profundidade igual a 0,5 em cada segmento.

### Como o recorte foi escolhido

Os rótulos de fácies foram usados para montar um subconjunto pequeno e balanceado com três grupos e quatro segmentos homogêneos por grupo. Portanto, os rótulos são usados tanto na **seleção do recorte** quanto posteriormente na **avaliação**, mas nunca são usados como atributos de entrada ou como alvos de treinamento do SOM.

As profundidades iniciais foram escolhidas de forma pragmática após a inspeção dos trechos homogêneos disponíveis no CSV público. Esse critério não foi pré-registrado. O objetivo foi obter um exercício compacto, com segmentos de mesmo comprimento e amostragem regular.

As curvas GR, PHIND e PE foram selecionadas porque são variáveis numéricas disponíveis sem valores ausentes nas linhas escolhidas e permitem manter a dimensionalidade pequena para um experimento de disciplina.

São permitidos vizinhos pertencentes ao mesmo poço. O experimento descreve a geometria deste conjunto fixo e não estima desempenho em poços nunca vistos.

## Desenho experimental

Os mesmos dados padronizados são avaliados em duas representações:

1. **Representação original** — curvas GR, PHIND e PE comparadas com DTW multivariado dependente.
2. **Representação reduzida** — cada amostra 3D é quantizada por um SOM 1D com 8 neurônios; a sequência escalar resultante é comparada com DTW 1D.

O SOM utiliza:

- 8 neurônios;
- 2.500 atualizações;
- 30 inicializações independentes, usando sementes de 0 a 29.

As escolhas de 8 neurônios e 2.500 atualizações são configurações pragmáticas e fixas para esta prova de conceito. Elas não foram otimizadas a partir das métricas finais do experimento.

A implementação do SOM utiliza apenas NumPy. Trata-se de uma codificação simples baseada em SOM e **não** de uma implementação de SOrS/IntraSOM.

## Definição do DTW

O DTW utiliza custo acumulado de alinhamento:

- custo local absoluto para a representação 1D;
- custo local euclidiano para a representação multivariada;
- sem divisão posterior pelo comprimento do caminho de alinhamento.

Todos os segmentos comparados possuem 16 amostras.

Um script de validação verifica a simetria da implementação do DTW, a regularidade da grade de profundidade dos segmentos selecionados e se o tratamento de empates na busca pelo vizinho mais próximo é independente da ordem de entrada.

## Métricas

As métricas foram escolhidas para medir aspectos diferentes da preservação de similaridade:

- **Acurácia do vizinho mais próximo por fácies**: verifica se o segmento mais próximo pertence à mesma fácies. Quando há mais de um segmento empatado na menor distância, o crédito é dividido igualmente entre os candidatos.
- **Concordância do primeiro vizinho**: verifica em que medida o primeiro vizinho obtido na representação original permanece entre os vizinhos de distância mínima após a redução. Empates recebem crédito fracionado uniforme, sem depender da ordem ou do identificador dos segmentos.
- **Razão de separação**: distância média entre fácies diferentes dividida pela distância média entre segmentos da mesma fácies.
- **Correlação de Pearson entre distâncias**: mede a associação linear entre as 66 distâncias par a par obtidas nas representações original e reduzida.
- **Tempo de comparação**: mede o tempo necessário para construir a matriz de distâncias depois que as duas representações já estão disponíveis.

O coeficiente de Pearson não deve ser interpretado como uma porcentagem de similaridade preservada. O tratamento dos empates é independente dos nomes dos segmentos e dos rótulos de fácies.

## Repetições do SOM

Como o treinamento do SOM possui componentes aleatórios, a representação reduzida é avaliada em 30 sementes diferentes. As métricas do espaço reduzido são apresentadas como **média ± desvio-padrão amostral**.

Essa dispersão representa a variabilidade de inicialização e amostragem do SOM sobre o mesmo conjunto fixo de dados. Ela não representa um intervalo de confiança de generalização para outros poços.

## Medição de tempo

As comparações de tempo utilizam o mesmo número de repetições para as duas representações. A ordem de execução é alternada para reduzir efeitos de ordem e carga momentânea.

O benchmark mede **somente a etapa de construção da matriz de distâncias**. O treinamento e a transformação pelo SOM ficam fora do cronômetro. Assim, o resultado de tempo deve ser interpretado apenas como custo da comparação depois que as representações já estão disponíveis.

## Resultados

Os resultados atuais são gerados automaticamente pelo experimento e salvos em:

- results/summary.md
- results/metrics_summary.csv
- results/per_seed_metrics.csv
- results/report.html

O README não replica os valores numéricos finais para evitar que eles fiquem desatualizados após alguma alteração metodológica.

Dois arquivos agregados merecem atenção:

- results/distance_matrix_reduced_mean.csv contém a média elemento a elemento das 30 matrizes de distância reduzidas.
- results/nearest_neighbors_reduced_from_mean_matrix.csv contém os vizinhos calculados a partir dessa matriz média.

Esses arquivos são apenas agregações descritivas e não correspondem a uma única execução do SOM. Os arquivos distance_matrix_reduced.csv e nearest_neighbors_reduced.csv foram mantidos apenas como aliases legados dessas agregações.

## Como reproduzir o experimento

O ambiente de referência utilizado durante a revisão foi:

- Python 3.12.4
- NumPy 2.4.6

Instale a dependência fixada:

~~~bash
pip install -r requirements.txt
~~~

Execute as validações:

~~~bash
python validate_experiment.py
~~~

Execute o experimento:

~~~bash
python experiment.py
~~~

Execute o experimento e atualize todos os arquivos de resultado:

~~~bash
python experiment.py --save
~~~

Reconstrua o subconjunto diretamente da fonte pública fixada:

~~~bash
python prepare_data.py
~~~

No Windows, o arquivo rodar_experimento.bat executa primeiro as validações e depois atualiza os resultados.

## Estrutura do repositório

~~~text
.
├── data/
│   └── selected_segments.csv
├── experiment.py
├── prepare_data.py
├── validate_experiment.py
├── requirements.txt
├── environment.txt
├── rodar_experimento.bat
└── results/
    ├── summary.md
    ├── metrics_summary.csv
    ├── per_seed_metrics.csv
    ├── distance_matrix_original.csv
    ├── distance_matrix_reduced_mean.csv
    ├── nearest_neighbors_original.csv
    ├── nearest_neighbors_reduced_from_mean_matrix.csv
    └── report.html
~~~

## Escopo e limitações

Este é um experimento descritivo e de pequena escala sobre um subconjunto selecionado deliberadamente. Não há divisão treino/teste porque a pergunta estudada é sobre a geometria deste conjunto específico, e não sobre capacidade preditiva em poços novos.

Os resultados não devem ser generalizados para outras bacias, outros poços ou outras configurações de perfilagem sem experimentos adicionais.

## Referências

- HALL, B. Facies classification using machine learning. *The Leading Edge*, 35(10), 906–909, 2016. DOI: 10.1190/tle35100906.1.
- KOHONEN, T. *Self-Organizing Maps*. Springer, 3rd ed., 2001.
- SAKOE, H.; CHIBA, S. Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43–49, 1978. DOI: 10.1109/TASSP.1978.1163055.

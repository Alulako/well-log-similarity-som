# Resultado do experimento

## Pergunta testada

Uma representação reduzida baseada em SOM consegue preservar relações úteis de similaridade entre segmentos de perfis de poços e reduzir o custo da etapa de comparação?

## Dados e recorte

Foram usados 12 segmentos reais do conjunto público da SEG: 3 fácies, 4 segmentos por fácies, 16 amostras por segmento e 3 curvas (GR, PHIND e PE).

Os rótulos de fácies foram usados **na seleção do recorte e na avaliação**, mas não entram como atributos nem como alvos do treinamento do SOM. O recorte é estratificado e deliberadamente pequeno. Vizinhos do mesmo poço são permitidos porque o objetivo é estudar a geometria deste conjunto fixo, não estimar desempenho em poços novos.

## Comparações

- Baseline: três curvas padronizadas + DTW multivariado dependente.
- Redução: SOM 1D com 8 neurônios + DTW 1D.
- O SOM é repetido 30 vezes, com sementes de 0 a 29.
- O DTW usa **custo acumulado**, sem normalização pelo comprimento do caminho.
- O tempo reportado mede **somente a construção da matriz de distâncias com as representações já disponíveis**; treinamento e transformação do SOM ficam fora do cronômetro.

## Resultados

| Métrica | Original | SOM reduzido — média ± DP |
|---|---:|---:|
| Acurácia do vizinho mais próximo por fácies (empates fracionados) | 50.0% | 49.6% ± 5.6% |
| Acordo com o primeiro vizinho do baseline (empates fracionados) | — | 60.7% ± 6.3% |
| Razão de separação inter/intrafácies | 1.301 | 1.954 ± 0.177 |
| Correlação de Pearson entre distâncias e baseline | 1,000 | 0.754 ± 0.074 |
| Tempo da matriz de distâncias | 249.3 ± 52.0 ms | 40.0 ± 5.2 ms |

Razão entre as médias de tempo (original/reduzido): **6.23×**.

## Interpretação

As métricas respondem a perguntas diferentes. A correlação de Pearson mede associação global entre as distâncias das duas representações; a acurácia por fácies mede se o conjunto de vizinhos empatados na menor distância pertence à mesma classe; e o acordo de primeiro vizinho mede a concordância com o baseline. Em empates, o crédito é fracionado uniformemente entre os candidatos mínimos, sem usar nomes ou rótulos para desempatar.

Nenhuma dessas medidas, isoladamente, demonstra preservação completa da similaridade. Este resultado é uma **prova de conceito descritiva em um conjunto fixo**, não uma estimativa de generalização para poços novos nem uma validação geológica ampla.

## Observação sobre agregação

Os resultados principais acima são a **média das métricas calculadas separadamente em cada uma das 30 sementes**. O arquivo distance_matrix_reduced_mean.csv contém, separadamente, a média elemento a elemento das 30 matrizes reduzidas e serve apenas como visualização agregada; ele não representa uma execução individual do SOM.

# Resultado do experimento

## Pergunta testada

Uma representação reduzida baseada em SOM consegue preservar relações úteis de similaridade entre segmentos de perfis de poços e reduzir o custo da comparação?

## Dados

Foram usados 12 segmentos reais retirados do conjunto público da SEG (2016):

- 3 fácies: 2, 3 e 6;
- 4 segmentos por fácies;
- 16 amostras por segmento;
- 3 curvas: GR, PHIND e PE;
- 192 linhas no total.

Os rótulos de fácies foram usados apenas como referência externa de avaliação.

## Comparações

**Baseline:** dados originais padronizados com três variáveis, comparados por DTW multivariado.

**Representação reduzida:** dados transformados por um SOM 1D de 8 neurônios e comparados por DTW 1D.

Como o SOM possui componentes aleatórios, foram realizadas **30 execuções com sementes diferentes (0 a 29)**. Para o espaço reduzido, os resultados são apresentados como média ± desvio-padrão.

## Resultados

| Métrica | Original | SOM reduzido — 30 execuções |
|---|---:|---:|
| Acurácia do vizinho mais próximo por fácies | 58,3% | 48,6% ± 8,2% |
| Razão de separação | 1,367 | 2,543 ± 0,360 |
| Correlação com a matriz original | 1,000 | 0,758 ± 0,062 |

Na última verificação de tempo:

- matriz original: aproximadamente **139 ms**;
- matriz reduzida: aproximadamente **93 ± 9 ms**;
- aceleração aproximada: **1,5×**.

Os tempos podem variar entre execuções e máquinas.

## Leitura do resultado

O experimento não sustenta a afirmação de que a redução por SOM melhora consistentemente a identificação do vizinho mais próximo. A acurácia média no espaço reduzido foi inferior à do baseline e apresentou variabilidade entre inicializações.

Por outro lado, a representação reduzida preservou parcialmente a estrutura global de distâncias, com correlação média de 0,758, e apresentou maior razão de separação entre fácies. A etapa de comparação também foi mais rápida nesta implementação.

Assim, a conclusão mais segura é:

> Neste recorte pequeno, a redução baseada em SOM preservou parte da estrutura de similaridade e reduziu o custo da comparação, mas introduziu variabilidade e perda de desempenho na métrica de vizinho mais próximo.

O resultado deve ser interpretado como **prova de conceito**. O conjunto é deliberadamente pequeno e não permite concluir que a mesma relação ocorrerá em qualquer bacia, poço ou conjunto de curvas.

## O que isso permite dizer na apresentação

> Em uma prova de conceito com 12 segmentos reais de poços e três curvas de perfilagem, uma codificação 1D baseada em SOM foi avaliada em 30 inicializações diferentes. A representação reduzida manteve correlação média de 0,758 com as distâncias originais e reduziu o custo computacional da comparação. Entretanto, a acurácia do vizinho mais próximo caiu em média, mostrando que a redução preserva apenas parcialmente a estrutura de similaridade e que seus resultados dependem da inicialização.

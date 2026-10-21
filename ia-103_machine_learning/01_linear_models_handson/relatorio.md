# Trabalho Prático 1 - Modelos Lineares

**Angel Ottoni | Unimontes | Especialização em Inteligência Artificial**
**IA-103 - Machine Learning | Outubro de 2026**

## Objetivo e organização

Neste trabalho, retomo os modelos lineares implementando as funções de custo, os gradientes e as etapas de predição. Meu objetivo foi compreender o comportamento dos algoritmos e relacionar os resultados às decisões de implementação. Organizei os experimentos nas quatro partes do roteiro. O notebook contém o código executado, os gráficos e as verificações; este relatório reúne as análises e conclusões.

Usei as quatro bases fornecidas na disciplina: food truck, imóveis, estudantes e microchips. Não encontrei valores ausentes nem linhas inteiramente duplicadas. Implementei o gradiente descendente batch com NumPy e utilizei o SciPy para otimizar os modelos logísticos. Conferi os gradientes logísticos por diferenças finitas e comparei as regressões lineares com mínimos quadrados. Nas divisões aleatórias, fixei a semente em 42.

## 1. Regressão linear simples

Na base de food truck, com 97 observações, modelei o lucro em função da população. Com parâmetros iniciais iguais a zero, o custo foi **32,072734**, conforme o valor de referência do roteiro. Após 2.000 atualizações com taxa de aprendizado 0,01, obtive **θ₀ = -3,788069** e **θ₁ = 1,182213**, com custo **4,478028**.

Comparei as taxas 0,001, 0,01 e 0,02 durante 2.000 atualizações. A taxa menor apresentou redução mais lenta do custo. A taxa 0,03 ultrapassou o limite de estabilidade calculado para essa matriz, aproximadamente 0,024324, e apresentou crescimento do erro. Limitei essa demonstração às primeiras 20 atualizações para evitar estouro numérico. Também comparei 100, 500, 2.000 e 15.000 atualizações com taxa 0,01. A execução mais longa se aproximou dos coeficientes de mínimos quadrados: -3,895781 e 1,193034. Isso explica por que os parâmetros não coincidem exatamente com os valores aproximados do roteiro em qualquer número de iterações.

Usando o ajuste de 2.000 atualizações e convertendo as unidades, o lucro previsto foi **3.496,76 unidades monetárias** para 35 mil habitantes e **80.340,59** para 100 mil. A primeira previsão extrapola a faixa observada, cujo mínimo é aproximadamente 50 mil habitantes. Portanto, precisa de mais cautela. A relação encontrada também não demonstra que população, isoladamente, determine lucro.

## 2. Regressão linear múltipla

Na base de imóveis, com 47 observações, utilizei área e quantidade de quartos. Padronizei os atributos pela média e pelo desvio padrão populacional, preservando o intercepto. Para prever o novo imóvel, reutilizei essas mesmas estatísticas. Essa etapa reduz a diferença de escala entre os atributos e facilita a convergência do gradiente; não determina sua importância no problema.

Com taxa 0,01 e 2.000 atualizações, a curva de custo diminuiu continuamente. Para o imóvel de **1.650 pés² e três quartos**, estimei **293.083,35 unidades monetárias**. O resultado de mínimos quadrados foi 293.081,46, indicando proximidade entre as soluções. O custo final, aproximadamente 2,04 bilhões, está em unidades monetárias ao quadrado e não é comparável diretamente ao custo da primeira base. O coeficiente de quartos ficou negativo quando a área é mantida constante, mas não interpreto isso como efeito causal. Os resultados descrevem ajuste na própria base, sem avaliação independente de imóveis novos.

## 3. Regressão logística

Para os 100 estudantes, implementei custo logístico, gradiente, predição com limiar 0,5 e acurácia. Usei sigmoide estável e `logaddexp` no custo para evitar problemas numéricos com probabilidades extremas. Com parâmetros nulos, o custo foi **0,693147** e o gradiente aproximado foi **[-0,1000; -12,0092; -11,2628]**. A verificação numérica da derivada apresentou erro absoluto de aproximadamente 1,21 × 10⁻⁵.

O método de gradientes conjugados de `scipy.optimize.minimize` convergiu e reduziu o custo para **0,203498**. Para notas **45 e 85**, a probabilidade estimada de aprovação foi **77,62%**, resultando em classificação positiva. A acurácia foi **89%**, medida nos próprios dados de treinamento. A fronteira é linear nas notas. Esse acerto não deve ser apresentado como desempenho em novos estudantes nem como evidência de calibração das probabilidades.

## 4. Regularização e seleção de modelos

Na base de 118 microchips, construí os termos polinomiais de grau total até 6, gerando 28 atributos, incluindo o intercepto. Adotei a penalização **λ/(2m) × Σθⱼ²**, excluindo o intercepto. Assim, a derivada é λθⱼ/m. Essa escolha corrige a inconsistência de fator entre custo e gradiente no roteiro; conferi a implementação por diferenças finitas. Antes da divisão, ajustei o modelo demonstrativo com λ = 1 na base completa e obtive 83,05% de acerto, apenas como medida de ajuste.

Nos dez treinamentos adicionais, mantive a mesma divisão estratificada: **82 amostras para treino e 36 para teste**, com semente 42. Os resultados foram:

| λ | Acurácia no treino | Acurácia no teste |
|---:|---:|---:|
| 0,0001 | 92,68% | 83,33% |
| 0,001 | 89,02% | 83,33% |
| 0,01 | 85,37% | 83,33% |
| 0,1 | 82,93% | 80,56% |
| 0,3 | 82,93% | 80,56% |
| 1 | 82,93% | 77,78% |
| 3 | 81,71% | 66,67% |
| 10 | 76,83% | 61,11% |
| 100 | 53,66% | 50,00% |
| 1000 | 51,22% | 50,00% |

Os valores **0,0001, 0,001 e 0,01 empataram no maior acerto do teste**. Porém, a mesma acurácia não significa comportamento equivalente. Em λ = 0,0001, a norma dos pesos foi aproximadamente 174,74, e a perda logística passou de 0,166 no treino para 1,840 no teste. Em λ = 0,01, essa perda no teste foi 0,538. A diferença evidencia erros probabilísticos mais severos no primeiro ajuste e, junto da fronteira mais complexa, é compatível com overfitting. Com λ = 1000, os acertos próximos de 50% em ambos os conjuntos indicam restrição excessiva e underfitting.

Como complemento, fiz validação cruzada estratificada em cinco partes somente nas 82 amostras de treinamento. A maior média foi **80,66%, com λ = 0,3**, mas λ = 0,001 alcançou 80,51%. A diferença é pequena diante da variação entre as partes; não demonstra superioridade robusta. O roteiro pede comparar λ pelo teste, e cumpri essa comparação. Entretanto, ao participar da escolha, esse conjunto perde a condição de avaliação final independente. A análise complementar não elimina essa limitação. Além disso, cada erro no teste altera sua acurácia em cerca de 2,78 pontos percentuais.

## Conclusões

O trabalho reforçou a importância de conferir unidades, derivadas e convergência antes de interpretar uma métrica. A padronização facilitou a otimização da regressão múltipla, enquanto os atributos polinomiais ampliaram a flexibilidade da logística e tornaram necessário controlar a complexidade. Também observei que configurações com a mesma acurácia podem apresentar perdas e probabilidades bastante diferentes.

As implementações passaram pelas verificações descritas e responderam às tarefas propostas. Para avançar além do exercício didático, eu reservaria um teste final fora da seleção de hiperparâmetros, repetiria a avaliação em outras divisões e examinaria os erros por classe. As bases pequenas e o uso exploratório do teste limitam as conclusões de generalização.

## Referências e autoria

- Roteiro e bases disponibilizados na disciplina IA-103, Unimontes. Conforme os créditos do roteiro, os exercícios adaptam atividades de Machine Learning de Andrew Ng, Stanford. Os materiais originais não são de minha autoria e permanecem fora do versionamento.
- SciPy. [Documentação de `scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). Consulta em 21/10/2026.
- Resolução preparada com apoio de IA na implementação, redação e verificação. Os resultados foram obtidos pela execução do notebook; a revisão e a compreensão antes da submissão cabem ao autor.

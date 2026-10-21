# TP1 - Modelos Lineares | IA-103

Resolução de Angel Ottoni para a Especialização em Inteligência Artificial da Unimontes.

O notebook reúne implementação, execução, gráficos e análise das quatro partes do trabalho. O relatório em Markdown sintetiza os resultados. A organização e a redação da resolução são próprias; o roteiro é uma adaptação de atividades de Andrew Ng/Stanford, conforme seus créditos. Os dados e os materiais fornecidos na disciplina não são de autoria do estudante.

## Arquivos

- `TP1_Modelos_Lineares.ipynb`: resolução executada, com nove células de código e verificações numéricas.
- `relatorio.md`: relatório de análises e conclusões. A paginação depende do renderizador; contém aproximadamente 1.200 palavras e uma tabela, visando o limite de 1 a 3 páginas com diagramação compacta. Entregue em Markdown por preferência do autor.
- `figuras/`: onze gráficos gerados pelo notebook, incluindo as dez fronteiras em uma figura.
- `resultados.json`: parâmetros, métricas, versões e índices da divisão de microchips.
- `experimentos_regularizacao.csv` e `validacao_cruzada.csv`: tabelas dos experimentos.
- `reproduzir.py`: execução integral e montagem opcional do ZIP.
- `requirements.txt`: versões utilizadas.
- `materiais_professores/`: cópia local do roteiro original, ignorada pelo Git.
- `entrega/TP1_Angel_Ottoni.zip`: pacote local de submissão, ignorado pelo Git porque inclui os dados fornecidos.

## Reproduzir

Utilize Python 3.12 e os quatro dados originais na mesma pasta do notebook:
`food_truck.txt`, `imoveis.txt`, `estudantes.txt` e `microchips.txt`.
Esses dados permanecem disponíveis localmente, mas são ignorados pelo Git. Após clonar o repositório, é necessário obtê-los pelo material da disciplina. O ZIP local de entrega já contém os quatro arquivos.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduzir.py --zip
```

A execução sobrescreve somente os artefatos gerados da própria resolução. O notebook também pode ser aberto em um ambiente Jupyter e executado do início ao fim.

## Cobertura do roteiro

| Parte | O que foi resolvido |
|---|---|
| Linear simples | Custo, gradiente batch, taxas, iterações, curva de erro, reta e previsões para 35 mil e 100 mil habitantes |
| Linear múltipla | Padronização, gradiente vetorizado, curva de custo e previsão para 1.650 pés² e três quartos |
| Logística | Sigmoide, custo, gradiente, resumo de `minimize`, ajuste CG, fronteira, probabilidade para notas 45/85, predição e acurácia |
| Regularização | Mapeamento de grau 6, custo e gradiente consistentes, ajuste inicial, dez λ na mesma divisão, dez fronteiras, métricas e interpretação |
| Verificação | Custos de referência, diferenças finitas, mínimos quadrados, convergência dos otimizadores e divisão sem sobreposição |

O histórico de custo inclui o estado inicial na posição zero. A demonstração de taxa instável usa 20 atualizações; as três taxas estáveis usam 2.000. A comparação principal de λ pelo teste atende ao roteiro e é identificada como exploratória. A validação cruzada complementar usa apenas o treinamento, mas não restaura a independência do teste já consultado.

## Material dos professores e versionamento

Os slides, roteiros originais, ementas e dados fornecidos são mantidos localmente e excluídos pelo `.gitignore`. O commit da resolução remove as antigas cópias do roteiro e dos dados do estado atual do Git; não reescreve os commits anteriores. Notebook de resolução, relatório e artefatos produzidos ficam versionados. Não se atribui aos professores a autoria das bases originais adaptadas; elas são tratadas aqui como material fornecido, sem redistribuição pública pelo commit.

## Autoria e revisão

Resolução preparada com apoio de IA na implementação, redação e verificação. A execução gerou os resultados incluídos. A leitura, compreensão e revisão pelo autor devem anteceder a submissão. A preferência por relatório em Markdown foi definida pelo autor; o enunciado limita sua extensão a 1 a 3 páginas, sem especificar formato de arquivo.

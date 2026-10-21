# TP1 - Modelos Lineares | IA-103

**Angelina de Meiras Ottoni | PPGMCS/Unimontes | 2026**

Resolução do Trabalho Prático 1 de Machine Learning, organizada no próprio notebook fornecido pelo professor.

## Arquivos

- [tp1_linear_models.ipynb](tp1_linear_models.ipynb): notebook original preenchido e executado, com implementações, gráficos e análises.
- [report.md](report.md): relatório de análises e conclusões, em redação impessoal.
- [submission.md](submission.md): arquivo com os endereços da resolução para envio no ambiente da disciplina.
- `food_truck.txt`, `imoveis.txt`, `estudantes.txt` e `microchips.txt`: datasets fornecidos para os exercícios. Os nomes originais foram mantidos para compatibilidade com o roteiro.
- [requirements.txt](requirements.txt): versões das dependências utilizadas.

## Execução

Utilizar Python 3.12 e instalar as dependências:

```bash
python -m pip install -r requirements.txt
```

Abrir `tp1_linear_models.ipynb` em um ambiente com suporte a notebooks, selecionar o ambiente Python e executar as células em ordem. Os quatro datasets devem permanecer na mesma pasta. Os gráficos estão incorporados às saídas do notebook; não há dependência de script auxiliar ou de imagens externas.

## Correspondência com o roteiro

As 101 células originais foram mantidas na mesma ordem, com preenchimento dos espaços de implementação e acréscimo de notas e análises identificadas. Foram preservados os nomes de funções e variáveis, incluindo `J_history`, com um custo por atualização.

As alterações técnicas foram explicadas junto ao código: preservação do intercepto na normalização, estabilidade da sigmoide e do custo logístico, consistência entre custo e gradiente da regularização e configuração dos otimizadores. As duas imagens de referência ausentes no material foram sinalizadas nos respectivos locais. Os gráficos da resolução são apresentados nas saídas das células.

Foram implementadas as quatro partes: regressão linear simples, regressão múltipla, regressão logística e regularização. Os dez experimentos de regularização utilizam a mesma divisão de treino e teste; a validação cruzada complementar utiliza somente o treinamento. As limitações da comparação pelo teste estão explicitadas.

## Materiais e entrega

Os datasets utilizados nos exercícios acompanham a resolução e não são de autoria da estudante. Os enunciados e créditos do notebook original foram preservados. Slides, ementas e demais materiais de aula ficam fora do versionamento.

A submissão será feita por arquivo Markdown contendo o link do repositório, conforme definido pela estudante. O relatório também foi mantido em Markdown.

## Créditos

Conforme o roteiro, o trabalho adapta atividades de Machine Learning de Andrew Ng, Stanford. Implementações, experimentos e análises preparados com apoio de IA. Os resultados apresentados correspondem à execução do notebook preenchido.

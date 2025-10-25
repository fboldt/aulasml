## 12. Testes de Significância Estatística

### 1. Conceitos Fundamentais

1.1. **O propósito dos testes de significância:**
Um teste de significância estatística é utilizado para avaliar se os resultados obtidos em uma amostra refletem um padrão real em toda a população ou se são apenas consequência do acaso. Esses testes formam a base de inferência estatística, permitindo fundamentar conclusões quantitativas e comparações objetivas entre modelos, métodos ou grupos.

***

1.2. **Hipótese nula (H₀):**
A hipótese nula estabelece uma condição de ausência de diferença, efeito ou relação estatística. É a suposição inicial que se busca testar, assumindo que o resultado observado decorre do acaso.
Exemplo em Machine Learning: testar se dois modelos (A e B) têm a mesma acurácia média — H₀: µA = µB.

***

1.3. **Hipótese alternativa (H₁ ou Hₐ):**
A hipótese alternativa representa o contraponto à hipótese nula, propondo que há, de fato, uma diferença estatística.
Ela pode ser **bicaudal** (duas direções possíveis de diferença) ou **unicaudal** (diferença em uma única direção esperada).
Exemplo:

- H₁ bicaudal: µA ≠ µB (diferença genérica entre dois modelos).
- H₁ unicaudal: µA > µB (modelo A tem desempenho superior ao modelo B).

***

1.4. **O significado do valor-p:**
O **valor-p** (probabilidade observada) mede a chance de observar resultados iguais ou mais extremos do que os obtidos, **assumindo que H₀ é verdadeira**.

- Um **valor-p pequeno** (geralmente menor que 0,05) sugere que o resultado dificilmente ocorreria por acaso, justificando a rejeição de H₀.
- Um **valor-p elevado** indica que não há evidência estatística suficiente para rejeitar H₀.

**Importante:** o valor-p não mede a magnitude da diferença nem a probabilidade de H₀ ser verdadeira; ele apenas quantifica a evidência contra ela.

***

1.5. **Nível de significância (α):**
Define o limiar de probabilidade para decidir se o resultado é considerado estatisticamente significativo.
Valores comuns são **α = 0,05** (95% de confiança) e **α = 0,01** (99% de confiança).
Se **p < α**, rejeita-se H₀, aceitando-se H₁ como mais plausível.
A escolha de α depende do contexto:

- Estudos exploratórios costumam aceitar α = 0,1.
- Aplicações críticas (medicina, segurança) exigem α < 0,01.

***

1.6. **Interpretação correta da significância:**
A significância estatística indica que o resultado é improvável sob H₀, mas **não garante relevância prática**.
Diferenças muito pequenas podem ser estatisticamente significativas em grandes amostras.
Em Machine Learning, é fundamental combinar testes de significância com métricas de **efeito prático** (ex.: diferença no F1-score, ganho percentual de acurácia) e **tamanho de efeito** (ex.: Cohen’s d).

***

1.7. **Contexto no fluxo de um projeto de Machine Learning:**
Os testes de significância são aplicados nas fases finais de avaliação de modelos, após:

- Definir o conjunto de validação;
- Executar treino e teste com múltiplas rodadas (ex.: k-fold cross-validation);
- Calcular métricas repetidas (acurácia, F1, ROC-AUC).
Esses resultados permitem verificar **se a diferença de desempenho entre modelos é consistente** e não apenas fruto da variação amostral.

***

1.8. **Exemplo prático de aplicação:**
Suponha dois classificadores avaliados em 10 folds de validação cruzada, com médias de acurácia 0,91 e 0,88.
O teste t pareado pode determinar se a diferença de 0,03 na média é estatisticamente significativa.
Caso o p-value retornado seja 0,012, por exemplo:

- Como p < 0,05 → rejeita-se H₀.
- Conclusão: há evidência de que o modelo A supera o B em desempenho.

***

1.9. **Resumo visual (interpretação dos testes):**


| Resultado | Valor-p observado | Decisão sobre H₀ | Interpretação |
| :-- | --: | :-- | :-- |
| Diferença não significativa | p > α | Não rejeita H₀ | Sem evidência contra H₀ |
| Diferença significativa | p < α | Rejeita H₀ | Evidência contra H₀ |
| Diferença altamente sig. | p < 0,01 | Rejeita fortemente H₀ | Muito improvável sob acaso |

***

### 2. Erros Estatísticos


***

2.1. **Introdução aos erros de decisão**
Os testes de hipóteses baseiam-se em uma decisão binária: rejeitar ou não rejeitar a hipótese nula (H₀).
Contudo, essa decisão está sujeita a incertezas, já que trabalhamos com amostras e não com a população completa.
Dessa limitação surgem dois tipos clássicos de erro: **Tipo I** e **Tipo II**.
Ambos são inevitáveis, mas podem ser controlados por meio de tamanho de amostra, nível de significância e poder estatístico.

***

2.2. **Erro Tipo I (α): o falso positivo**
O **Erro Tipo I** ocorre quando rejeitamos H₀ mesmo ela sendo verdadeira.
Em outras palavras, o teste indica uma diferença ou efeito inexistente — detectando um “falso alarme”.

Exemplo prático:
Ao comparar dois modelos de Machine Learning, concluímos que o modelo A é mais preciso que o B, quando na verdade ambos são equivalentes.

O nível de significância **α** representa a **probabilidade máxima de cometer esse erro**.
Valores comuns são 0,05 (5%) ou 0,01 (1%).
Isso significa aceitar o risco de 5% ou 1% de rejeitar H₀ indevidamente.

**Consequências:**

- Conclusões incorretas em experimentos.
- Risco de identificar “avanços” inexistentes em modelos ou métodos.

***

2.3. **Erro Tipo II (β): o falso negativo**
O **Erro Tipo II** ocorre quando não rejeitamos H₀, embora ela seja falsa.
Nesse caso, o teste **falha em detectar uma diferença real** entre grupos ou modelos.

Exemplo:
Dois classificadores têm desempenhos distintos, mas o teste não identifica diferença significativa devido a tamanho amostral pequeno ou grande variabilidade nos resultados.

A probabilidade de cometer esse erro é indicada por **β (beta)**.
O oposto de β é o **poder do teste (1 - β)**.

***

2.4. **Poder do teste (1 - β): a sensibilidade estatística**
O **poder estatístico** representa a probabilidade de **detectar corretamente um efeito real**.
Poder alto significa que o teste é mais sensível a diferenças reais.
Cientificamente, é desejável que o poder seja **pelo menos 0,8 (ou 80%)**.

**Fatores que aumentam o poder:**

- Tamanho de amostra maior.
- Maior diferença real entre médias (efeito mais forte).
- Menor variabilidade (dados mais consistentes).
- Escolha correta do teste estatístico.

**Em Machine Learning:**
Maior poder significa maior confiança ao afirmar que um modelo realmente supera outro.

***

2.5. **Relação entre α, β e poder**
Existe um **trade-off** fundamental: reduzir α (menos falsos positivos) geralmente aumenta β (mais falsos negativos).
Portanto, definir α exige considerar o contexto da análise.
Em pesquisas científicas com grandes implicações, é preferível minimizar o **Erro Tipo I**; em aplicações exploratórias, prioriza-se reduzir o **Erro Tipo II**.

Tabela-resumo:


| Decisão / Realidade | H₀ verdadeira | H₀ falsa |
| :-- | :-- | :-- |
| **Rejeitar H₀** | Erro Tipo I (α) | Decisão correta (1 - β) |
| **Não rejeitar H₀** | Decisão correta (1 - α) | Erro Tipo II (β) |


***

2.6. **Exemplo aplicado em Machine Learning**
Imagine que, após diversas execuções de validação cruzada, queremos determinar se um **modelo otimizado** realmente supera o modelo base.

- H₀: não há diferença média nas acurácias.
- H₁: o modelo otimizado tem acurácia superior.

Caso o valor-p obtido seja **0,12**, não rejeitamos H₀.
Mas talvez o poder do teste seja baixo, sugerindo que precisamos de mais repetições (folds) ou mais dados para aumentar a sensibilidade.

Em contraste, se p < 0,05 mas o efeito é pequeno (0,2%), a diferença, embora estatisticamente significativa, **pode ser irrelevante na prática**.

***

2.7. **Visualização conceitual (interpretação gráfica)**
Nos gráficos de teste de hipóteses:

- A distribuição sob H₀ define a região crítica (α).
- Sob H₁, parte da área se sobrepõe, correspondendo aos erros de tipo I e II.
Essas curvas ajudam a visualizar que **aumentar o tamanho da amostra “estreita” as distribuições**, reduzindo ambos os erros.

***

2.8. **Resumo prático e recomendações**

- Escolha **α** com base no contexto da análise.
- Calcule **β e poder** antecipadamente, se possível.
- Relate sempre **ambos os erros** em artigos e experimentos de Machine Learning.
- Priorize **poder ≥ 0,8**, quando viável.
- Use visualizações e simulações para compreender como variabilidade e amostragem impactam suas conclusões.

***

Em suma, o domínio dos **erros tipo I e II** é essencial para interpretar corretamente resultados de testes de significância. No contexto de Machine Learning, compreender e equilibrar esses erros garante conclusões mais confiáveis sobre o desempenho comparativo entre modelos.


***

### 3. Testes Paramétricos


***

3.1. **Definição e Intuição**
Os **testes paramétricos** são métodos estatísticos utilizados quando os dados atendem a certos pressupostos sobre a distribuição populacional, em geral, a **normalidade**.
Eles se baseiam diretamente em parâmetros da distribuição, como **média** e **variância**, e são apropriados para amostras obtidas de populações normalmente distribuídas.

Esses testes comparam médias ou variâncias entre grupos, assumindo que:

- As observações são independentes;
- As variâncias entre os grupos são semelhantes (homocedasticidade);
- As variáveis medidas são contínuas e aproximadamente normais.

***

3.2. **Principais pressupostos dos testes paramétricos**
Antes de aplicar testes paramétricos, é fundamental verificar os seguintes pontos:

- **Normalidade dos dados:** pode ser verificada com testes como *Shapiro-Wilk* ou *Kolmogorov–Smirnov*, bem como inspeção visual via *Q-Q plots* ou histogramas.
- **Homogeneidade de variâncias:** checada com o *teste de Levene* ou *Bartlett*.
- **Independência amostral:** assegurada quando as observações de um grupo não influenciam as de outro.

Se esses pressupostos não forem atendidos, deve-se recorrer a **testes não-paramétricos** (ver Tópico 4).

***

3.3. **Teste t de Student**
O **teste t** é o mais utilizado entre os testes paramétricos e tem várias variações:

- **Teste t de uma amostra:** compara a média de uma amostra com um valor hipotético (exemplo: testar se o desempenho médio de um modelo é 90%).
- **Teste t para duas amostras independentes:** analisa se duas populações têm médias diferentes (exemplo: comparar o desempenho médio de dois modelos em datasets distintos).
- **Teste t pareado (dependente):** usado quando as amostras são relacionadas, como os resultados de dois modelos aplicados sobre o mesmo conjunto de folds.

**Fórmula geral:**

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
$$

onde \$ S_p \$ é o desvio padrão combinado (pooled standard deviation), e \$ n_1, n_2 \$ são os tamanhos das amostras.

O resultado do teste inclui o valor-p e o intervalo de confiança da diferença de médias.

***

3.4. **ANOVA (Analysis of Variance)**
Quando existem **três ou mais grupos** a serem comparados, utiliza-se o **teste ANOVA (fatorial ou simples)**.
Ele avalia se **pelo menos uma média difere das demais**, sem precisar testar cada par individualmente inicialmente.

**Exemplo:** testar se modelos A, B e C diferem significativamente em acurácia média.

A estatística do teste é baseada na razão entre a variância **entre grupos** e a variância **dentro dos grupos**:

$$
F = \frac{MS_{entre}}{MS_{dentro}}
$$

- \$ MS_{entre} \$: média dos quadrados entre grupos (diferenças entre médias dos grupos).
- \$ MS_{dentro} \$: média dos quadrados dentro dos grupos (variação interna).

Se o valor de **F** for alto e o **p-value** < 0,05, há evidência de diferença significativa entre as médias.

**Extensões:**

- **ANOVA unidirecional (one-way):** um fator, como “tipo de modelo”.
- **ANOVA fatorial (two-way):** dois fatores, como “modelo” e “tipo de dataset”.
- **MANOVA:** múltiplas variáveis dependentes simultaneamente.

***

3.5. **Testes pós-hoc (correção de múltiplas comparações)**
Quando o ANOVA detecta diferença significativa, é necessário identificar **quais grupos diferem**. Para isso, aplicam-se testes pós-hoc:

- **Tukey’s HSD (Honest Significant Difference)**
- **Bonferroni correction**
- **Scheffé test**

Esses testes ajustam os valores-p para evitar excesso de falsos positivos resultantes das múltiplas comparações.

***

3.6. **Escolha entre testes t e ANOVA no contexto de Machine Learning**

- Use **teste t pareado** ao comparar dois modelos com resultados sobre os mesmos folds.
- Use **ANOVA** quando comparar mais de dois algoritmos (ex.: Random Forest, SVM e XGBoost).
- Sempre avalie o tamanho da amostra e, quando possível, complemente o teste com medidas de **tamanho de efeito** (Cohen’s d, η², ω²).

**Exemplo aplicado:**
Comparando três modelos com 10 repetições de cross-validation:

- ANOVA → detecta diferença global (p < 0,03).
- Pós-hoc Tukey → indica que o modelo A é significativamente melhor que o B, mas não difere de C.

***

3.7. **Implementação prática em Python**

```python
from scipy.stats import ttest_rel, f_oneway
import numpy as np

# Exemplo: comparação de dois modelos
modelo_A = np.array([0.88, 0.90, 0.89, 0.87, 0.91])
modelo_B = np.array([0.85, 0.82, 0.84, 0.83, 0.86])

t_stat, p_val = ttest_rel(modelo_A, modelo_B)
print(f'Teste t pareado: t={t_stat:.3f}, p={p_val:.4f}')

# Exemplo ANOVA
modelo_C = np.array([0.91, 0.92, 0.90, 0.93, 0.91])
f_stat, p_val = f_oneway(modelo_A, modelo_B, modelo_C)
print(f'ANOVA: F={f_stat:.3f}, p={p_val:.4f}')
```


***

3.8. **Boas práticas para aplicações paramétricas em Machine Learning**

- Avaliar pressupostos antes da aplicação.
- Considerar normalização das métricas (ex.: acurácia em diferentes folds).
- Interpretar conjuntamente valor-p, diferença de médias e tamanho do efeito.
- Em datasets menores, usar **bootstrap** para robustez dos resultados.

***

Os **testes paramétricos** são ferramentas essenciais para avaliar diferenças significativas em desempenho de modelos. Sua correta utilização reforça a confiabilidade experimental em projetos de Machine Learning, evitando conclusões precipitadas e promovendo análises baseadas em evidência estatística.

***

### 4. Testes Não-Paramétricos


***

4.1. **Visão geral e motivação**
Os **testes não-paramétricos** são métodos estatísticos utilizados quando as suposições fundamentais dos testes paramétricos — como normalidade, homogeneidade de variância ou escala intervalar — **não são atendidas**.
Esses testes são ditos “distribuição-livres” (ou *distribution-free*) porque não dependem de parâmetros como média ou variância populacional.

Em contextos comuns de Machine Learning e ciência de dados, os testes não-paramétricos são úteis quando:

- O tamanho da amostra é pequeno (n < 30);
- Os resultados (ex.: acurácia de modelos) apresentam assimetria ou outliers;
- As métricas são ordinais (como rankings de modelos por desempenho).

***

4.2. **Características gerais**

- Baseiam-se em *rankings* ou ordenações dos dados, em vez de valores absolutos.
- Menos sensíveis a outliers e distribuições não normais.
- Possuem menor poder estatístico (tendem a ser mais conservadores).
- Podem ser aplicados tanto para amostras independentes quanto pareadas.

Essas propriedades tornam-nos particularmente valiosos para **comparações robustas em validações cruzadas** de pequenos conjuntos de modelos.

***

4.3. **Principais testes não-paramétricos**

**4.3.1. Teste de Mann–Whitney U (Wilcoxon Rank-Sum)**
Usado como alternativa ao *t-test* para duas amostras independentes.
Ele verifica se uma distribuição tende a ter valores maiores do que a outra, **sem assumir normalidade**.
A ideia central é ordenar todas as observações dos dois grupos e comparar as somas dos postos (ranks).

**Exemplo:** comparação de desempenho (ROC-AUC) entre dois classificadores em diferentes subsets de dados, sem pressupor distribuição normal.

**Interpretação:**

- H₀: As duas distribuições são idênticas.
- H₁: Uma das distribuições tende a gerar valores maiores.

***

**4.3.2. Teste de Wilcoxon Signed-Rank**
Equivalente não-paramétrico do *t-test pareado*, aplicado a amostras dependentes (mesmos indivíduos/modelos avaliados em condições diferentes).
Analisa as diferenças entre pares de observações, classificando-as por magnitude e sinal.

**Aplicação:** comparar dois modelos de ML usando métricas obtidas por *k-fold cross-validation*, onde cada fold fornece pares correlacionados.

**Interpretação:**

- H₀: Não há diferença nas medianas.
- H₁: Existe diferença sistemática nas medianas.

O teste é apropriado quando as diferenças não seguem distribuição aproximadamente normal.

***

**4.3.3. Teste de Kruskal–Wallis**
Extensão do Mann–Whitney para mais de dois grupos independentes, similar ao ANOVA.
Ele compara as medianas de k grupos com base nos ranks combinados de todas as observações.

**Exemplo:** comparação simultânea de três ou mais algoritmos (ex.: LogReg, SVM e RandomForest) pelos resultados de 10 folds de validação cruzada.

**Interpretação:**

- H₀: Todas as populações têm a mesma distribuição.
- H₁: Pelo menos uma população é diferente.

Se o teste indica significância (p < 0,05), deve-se aplicar análises pós-hoc (ex.: *teste de Dunn ou Conover*) para identificar quais grupos se diferenciam.

***

4.4. **Testes pós-hoc em contexto não-paramétrico**
Após um Kruskal–Wallis significativo, os testes pós-hoc ajudam a identificar grupos diferentes, ajustando o erro tipo I (múltiplas comparações).
Os métodos mais comuns incluem:

- **Teste de Dunn com correção de Bonferroni ou Holm.**
- **Conover test**, mais robusto quando há diferenças de variância entre grupos.

Essas análises são comuns em benchmarks de modelos em Machine Learning, onde múltiplos algoritmos são comparados sob métricas repetidas.

***

4.5. **Escolha entre testes paramétricos e não-paramétricos**


| Situação | Teste adequado |
| :-- | :-- |
| Duas amostras independentes, distribuição normal | t-test independente |
| Duas amostras independentes, distribuição não normal | Mann–Whitney U |
| Duas amostras pareadas, distribuição normal | t-test pareado |
| Duas amostras pareadas, distribuição não normal | Wilcoxon |
| Três ou mais grupos independentes, distribuição normal | ANOVA |
| Três ou mais grupos independentes, distribuição não normal | Kruskal–Wallis |


***

4.6. **Exemplo prático em Python**

```python
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, kruskal

# Duas amostras independentes
modelo_A = np.array([0.91, 0.89, 0.92, 0.90, 0.88])
modelo_B = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
stat, p = mannwhitneyu(modelo_A, modelo_B)
print(f"Mann–Whitney U: U={stat:.3f}, p={p:.4f}")

# Duas amostras pareadas
stat, p = wilcoxon(modelo_A, modelo_B)
print(f"Wilcoxon: T={stat:.3f}, p={p:.4f}")

# Três grupos independentes
modelo_C = np.array([0.93, 0.91, 0.92, 0.94, 0.90])
stat, p = kruskal(modelo_A, modelo_B, modelo_C)
print(f"Kruskal–Wallis: H={stat:.3f}, p={p:.4f}")
```


***

4.7. **Vantagens dos testes não-paramétricos**

- Robustos frente a outliers e distribuições assimétricas.
- Podem ser aplicados a dados ordinais e métricas rankeadas.
- Simples de interpretar e implementar com poucas suposições.

***

4.8. **Limitações**

- Menor poder estatístico (exige amostras maiores para detectar o mesmo efeito).
- Resultados são expressos em termos de medianas e ranks, não médias.
- Difíceis de estender a modelos com múltiplos fatores (como o ANOVA fatorial).

***

4.9. **Aplicação em Machine Learning e benchmarking de modelos**
Em experimentos com resultados de validação cruzada entre modelos, testes não-paramétricos são amplamente recomendados por não assumir normalidade das métricas.
Em particular, combinações como:

- **Friedman test + Nemenyi post-hoc** são padrão de ouro em comparações de algoritmos sobre diversos datasets (ex.: papers da área de avaliação de meta-learning).
Esses métodos oferecem uma avaliação **estatisticamente robusta e reprodutível** da superioridade entre modelos.

***

Os **testes não-paramétricos** são ferramentas fundamentais para validar diferenças de desempenho quando as condições ideais de normalidade e homocedasticidade não são atendidas. No contexto de Machine Learning, eles garantem análises justas, robustas e replicáveis, especialmente em benchmarks de múltiplos modelos e datasets.

***

### 5. Interpretação de Resultados


***

5.1. **O propósito da interpretação estatística**
Interpretar resultados de testes de significância é o processo de traduzir números (valor-p, estatísticas de teste, tamanhos de efeito) em **conclusões significativas** dentro de um contexto prático.
No campo de Machine Learning, a interpretação é essencial para decidir se um modelo realmente apresenta desempenho melhor que outro ou se os resultados observados decorrem do acaso.

***

5.2. **O papel do valor-p (p-value)**
O **valor-p** é a probabilidade de observar um resultado igual ou mais extremo do que o obtido, **assumindo que a hipótese nula (H₀) é verdadeira**.

- Se **p < α**, rejeita-se H₀, indicando que o efeito observado é improvável de ocorrer por acaso.
- Se **p ≥ α**, não há evidência estatística suficiente para rejeitar H₀.

**Exemplo prático:**
Comparando dois modelos de classificação via teste t pareado, obtém-se p = 0,032 com α = 0,05.
Como 0,032 < 0,05, conclui-se que o modelo A tem desempenho **estatisticamente superior** ao modelo B com 95% de confiança.

**Importante:** o valor-p **não mede a magnitude da diferença** nem a probabilidade de H₀ ser verdadeira — ele apenas indica evidência contra H₀.

***

5.3. **Tamanho do efeito (effect size)**
O **tamanho do efeito** quantifica a magnitude da diferença entre grupos, complementando o valor-p.
Dois modelos podem ter p < 0,05 mas com diferença prática irrelevante (ex.: 0,2% de melhoria na acurácia).
Entre as medidas mais comuns:

- **Cohen’s d**: diferença de médias em unidades de desvio padrão;

$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}
$$

onde \$ s_p \$ é o desvio padrão combinado.
Convenção de interpretação:
    - d ≈ 0,2 → pequeno efeito
    - d ≈ 0,5 → efeito médio
    - d ≥ 0,8 → efeito grande
- **Eta quadrado (η²)**: usado em ANOVA, mede a proporção da variância explicada pelo fator avaliado:

$$
\eta^2 = \frac{SS_{entre}}{SS_{total}}
$$

onde \$ SS \$ representa a soma dos quadrados (sum of squares).

O tamanho do efeito é especialmente útil para justificar relevância prática de diferenças estatísticas, principalmente em experimentos com grandes amostras.

***

5.4. **Intervalos de confiança (IC)**
Os **intervalos de confiança** expressam a faixa de valores possíveis para a média ou diferença entre médias, considerando uma probabilidade específica (ex.: 95%).
Interpretar um IC de 95% para a diferença entre dois modelos como  significa que há alta confiança de que o modelo A é **entre 0,5% e 3,5% mais preciso** que o modelo B.

Recomenda-se sempre apresentar IC juntamente com p-value e tamanho de efeito, pois o intervalo:

- Indica a **precisão da estimativa** (intervalos amplos sugerem incerteza);
- Pode demonstrar **a relevância prática** mesmo quando o teste é inconclusivo.

***

5.5. **Complementariedade entre significância e relevância prática**
Um resultado estatisticamente significativo nem sempre implica uma melhoria relevante.
Por exemplo:

- Em um dataset com milhões de observações, até diferenças de 0,1% podem ser estatisticamente relevantes.
- Porém, do ponto de vista de negócios, essa diferença pode ser desprezível.

Portanto, análises devem ser interpretadas considerando também **impacto prático, custo computacional e contexto de aplicação**.

**Checklist recomendado:**

1. Valor-p indica se há diferença significativa.
2. Tamanho do efeito mostra o quanto os modelos realmente diferem.
3. Intervalos de confiança quantificam a precisão da estimativa.
4. Métricas de desempenho ajudam a contextualizar se a diferença vale a pena.

***

5.6. **Limitações do valor-p e abordagens complementares**
O p-value sofre críticas por ser frequentemente mal interpretado. Algumas limitações incluem:

- Dependência do tamanho da amostra (quanto maior a amostra, mais fácil obter p pequeno).
- Falta de informação sobre a magnitude da diferença.
- Interpretação binária simplista (p < 0,05 vs p ≥ 0,05).

Alternativas e complementos recomendados:

- **Tamanho de efeito**: já abordado acima.
- **Bayesian inference**: trabalha com probabilidades a posteriori de hipóteses.
- **Bootstrap confidence intervals**: estimativas empíricas de incerteza em amostras pequenas.

Essas abordagens são úteis em Machine Learning, especialmente quando se comparam modelos em datasets com alta variabilidade.

***

5.7. **Aplicação prática em experimentos de Machine Learning**
Em benchmarks, é comum comparar vários modelos por métricas como acurácia, F1, ou ROC-AUC.
Procedimento típico:

1. Coletar métricas via *k-fold cross-validation*.
2. Aplicar teste t pareado ou Wilcoxon para comparar dois modelos.
3. Calcular tamanho de efeito (ex.: Cohen’s d).
4. Exibir médias, desvios padrão e intervalos de confiança para interpretar desempenho.

Um exemplo típico em Python:

```python
from scipy.stats import ttest_rel
import numpy as np

modelo_A = np.array([0.91, 0.90, 0.89, 0.92, 0.91])
modelo_B = np.array([0.87, 0.86, 0.88, 0.85, 0.89])

t_stat, p_val = ttest_rel(modelo_A, modelo_B)
print(f"t={t_stat:.3f}, p={p_val:.4f}")

# diferença média e IC
diff = modelo_A - modelo_B
mean_diff = np.mean(diff)
conf_int = [mean_diff - 1.96*np.std(diff)/np.sqrt(len(diff)),
            mean_diff + 1.96*np.std(diff)/np.sqrt(len(diff))]
print(f"Diferença média: {mean_diff:.3f}")
print(f"Intervalo de confiança 95%: {conf_int}")
```


***

5.8. **Boas práticas de reporte de resultados estatísticos**

- Sempre mencionar **testes aplicados, α, p-value, tamanho de efeito e IC**.
- Não reportar apenas “significativo” ou “não significativo” — incluir números exatos.
- Visualizar resultados com **boxplots** e **violin plots**, destacando diferenças reais entre distribuições.
- Apresentar tanto o **resultado estatístico** quanto sua **interpretação de negócio ou científica**.

***

Em resumo, interpretar corretamente resultados de testes de significância implica ir além do valor-p, combinando estatística, contexto e julgamento crítico. Em Machine Learning, isso se traduz em decisões mais robustas sobre escolha de modelos, reprodutibilidade de resultados e confiabilidade experimental.

***

### 6. Aplicações em Machine Learning


***

6.1. **A importância da significância estatística em ML**
Em projetos de Machine Learning, o uso de testes de significância é essencial para **avaliar diferenças entre modelos, métodos ou abordagens** de maneira científica.
Comumente, pesquisadores e engenheiros de dados treinam múltiplos modelos e, ao comparar métricas como acurácia ou F1-score, enfrentam a pergunta: *essa diferença é realmente relevante ou apenas fruto do acaso?*
Os testes estatísticos resolvem isso ao quantificar **a probabilidade de uma diferença observada ser aleatória**, permitindo decisões mais confiáveis e replicáveis.

Aplicar significância estatística garante que as melhorias relatadas sejam **estatisticamente justificáveis**, especialmente em cenários onde pequenas variações de métrica podem induzir falsas conclusões de superioridade de modelo.

***

6.2. **Cenários comuns de aplicação**

- **Comparação entre dois modelos** (ex.: Logistic Regression vs. Random Forest) avaliados sobre os mesmos folds de cross-validation;
- **Comparação de vários algoritmos** (ex.: SVM, KNN, XGBoost) para escolher o melhor;
- **Avaliação de tuning de hiperparâmetros**, verificando se uma melhoria no GridSearch é significativa;
- **Estudos de ablação**, analisando o impacto de uma feature ou pré-processamento;
- **Benchmarks** de datasets variados, garantindo que modelos generalizem apropriadamente.

Nesses casos, o foco é determinar se diferenças nas métricas de validação **são consistentes o suficiente para rejeitar a hipótese de igualdade média de desempenho**.

***

6.3. **Pipeline estatístico completo para comparação de modelos**

1. **Obter métricas comparáveis:**
    - Executar *k-fold cross-validation* sob o mesmo particionamento de dados (garantindo independência do mesmo contexto de avaliação).
2. **Checar pressupostos:**
    - Se as métricas forem aproximadamente normais → usar **testes paramétricos** (ex.: t-test pareado).
    - Se as métricas não forem normais ou contiverem outliers → aplicar **testes não-paramétricos** (Wilcoxon, Friedman).
3. **Executar o teste apropriado:**
    - Gere o valor estatístico e calcule o **valor-p**.
4. **Interpretar o resultado:**
    - p < 0,05 → diferenças significativas (rejeita H₀);
    - p ≥ 0,05 → diferenças não significativas (não se rejeita H₀).
5. **Complementar com tamanho do efeito e intervalos de confiança:**
    - Avaliar relevância prática além do resultado binário de significância.

***

6.4. **Comparação de dois modelos: exemplo prático com teste t pareado**
Se dois modelos são avaliados em 10 folds de cross-validation, podemos aplicar o teste *t pareado* sobre suas acurácias.

```python
import numpy as np
from scipy.stats import ttest_rel

acc_model_A = np.array([0.91, 0.92, 0.90, 0.91, 0.93, 0.92, 0.91, 0.90, 0.93, 0.92])
acc_model_B = np.array([0.89, 0.89, 0.91, 0.90, 0.88, 0.90, 0.89, 0.91, 0.89, 0.90])

t_stat, p_val = ttest_rel(acc_model_A, acc_model_B)
print(f"t={t_stat:.3f}, p={p_val:.4f}")

if p_val < 0.05:
    print("Diferença significativa entre os modelos.")
else:
    print("Diferença não significativa.")
```

Nesse exemplo, se o valor-p = 0,012, rejeitamos H₀ e concluímos que o modelo A supera estatisticamente o modelo B.

**Observação:** ambas as distribuições devem ser aproximadamente normais — verifique com o *teste de Shapiro-Wilk*.

***

6.5. **Comparação de múltiplos modelos com ANOVA ou Kruskal–Wallis**
Quando três ou mais modelos são comparados, o teste t não é mais suficiente.
Aplica-se então o **ANOVA (paramétrico)** ou o **Kruskal–Wallis (não-paramétrico)**.

**Exemplo (ANOVA):**

```python
from scipy.stats import f_oneway

rf = np.array([0.92, 0.91, 0.93, 0.92, 0.91])
svm = np.array([0.89, 0.90, 0.88, 0.90, 0.89])
xgb = np.array([0.94, 0.93, 0.92, 0.94, 0.93])

f_stat, p_val = f_oneway(rf, svm, xgb)
print(f"F={f_stat:.3f}, p={p_val:.4f}")
```

Se p < 0,05, há pelo menos um modelo com desempenho diferente dos demais.
Em seguida, aplicam-se **testes pós-hoc** (p. ex. *Tukey HSD* ou *Dunn test*) para descobrir **quais pares diferem**.

***

6.6. **Testes recomendados por tipo de comparação**


| Situação | N° de Modelos | Teste Sugerido | Tipo |
| :-- | :-- | :-- | :-- |
| Dois modelos (mesmo conjunto de dados) | 2 | t-test pareado | Paramétrico |
| Dois modelos (sem relação entre dados) | 2 | t-test independente | Paramétrico |
| Dois modelos, não normalidade | 2 | Wilcoxon Signed-Rank | Não-paramétrico |
| Três ou mais modelos | 3+ | ANOVA | Paramétrico |
| Três ou mais modelos (sem normalidade) | 3+ | Kruskal–Wallis ou Friedman | Não-paramétrico |


***

6.7. **Friedman test e Nemenyi post-hoc para benchmarks**
Em estudos com múltiplos datasets (ex.: comparação de 5 algoritmos em 10 datasets), o **teste de Friedman** é amplamente utilizado.
Ele avalia se as diferenças médias de ranking entre algoritmos são significativas.
Se o resultado for significativo, o **Nemenyi post-hoc** detalha quais pares de modelos são diferentes.

**Esses procedimentos são padrão em pesquisa científica de Machine Learning**, pois se ajustam a cenários com várias repetições e datasets heterogêneos.

***

6.8. **Boas práticas ao comparar modelos estatisticamente**

- Assegurar que cada modelo foi treinado sob **condições idênticas** (mesma amostra, pre-processamento, hiperparâmetros ajustados).
- Evitar conclusões baseadas em uma única métrica (combine acurácia com F1-score, ROC-AUC etc.).
- Repetir experimentos e reportar variabilidade.
- Apresentar o **valor-p, tamanho do efeito e significância prática**, não apenas “ganhos médios”.
- Complementar análises quantitativas com **gráficos** (boxplots, violin plots) para ilustrar variações.

***

6.9. **Resumo conceitual em formato decisório**


| Objetivo | Teste preferencial | Verificação prévia |
| :-- | :-- | :-- |
| Comparar dois modelos | t pareado ou Wilcoxon | Teste de normalidade (Shapiro–Wilk) |
| Comparar três ou mais modelos | ANOVA ou Kruskal–Wallis | Homogeneidade de variância |
| Benchmark com vários datasets | Friedman + Nemenyi | Repetição equilibrada de execuções |


***

6.10. **Reflexão final**
O uso de testes de significância estatística em Machine Learning eleva a **rastreabilidade e robustez científica** das conclusões.
Em vez de confiar apenas em médias, os testes permitem **mensurar a confiabilidade** das diferenças observadas e fundamentar a escolha de modelos com base estatística sólida.
Essa prática é essencial para estudos comparativos, artigos científicos e validações em ambientes corporativos com exigência de evidências quantitativas rigorosas.

***

### 7. Implementação Prática no Python


***

7.1. **Objetivo da etapa prática**
A implementação prática tem como proposta consolidar os conceitos teóricos apresentados em aulas anteriores — hipótese nula, níveis de significância, testes paramétricos e não paramétricos —, além de desenvolver **a capacidade de aplicar testes estatísticos diretamente em experimentos de Machine Learning**.
O foco recai sobre a biblioteca **SciPy**, especialmente o módulo `scipy.stats`, que fornece funções para testes clássicos (t-test, ANOVA, Wilcoxon, Mann–Whitney, Kruskal–Wallis etc.).

A combinação de **NumPy** (para manipulação de amostras), **Pandas** (para análise tabular) e **Matplotlib/Seaborn** (para visualização) permite construir uma análise estatística completa e reprodutível.

***

7.2. **Estrutura básica do ambiente de análise**

Antes de aplicar testes, recomenda-se organizar as medições de desempenho dos modelos (ou outras métricas) em arrays ou *DataFrames*.

```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Resultados em 10 folds
model_A = np.array([0.91, 0.90, 0.92, 0.91, 0.93, 0.92, 0.91, 0.93, 0.94, 0.92])
model_B = np.array([0.88, 0.89, 0.87, 0.89, 0.90, 0.88, 0.87, 0.89, 0.88, 0.90])

df = pd.DataFrame({'Model A': model_A, 'Model B': model_B})
sns.boxplot(data=df)
plt.title("Distribuição das Acurácias em 10 Folds")
plt.show()
```

Visualizações ajudam a **identificar assimetrias, outliers** e garantir que o teste estatístico a ser aplicado esteja alinhado aos pressupostos dos dados.

***

7.3. **Testes de normalidade e homogeneidade**

Antes de escolher o teste principal, é importante examinar os pressupostos.

**(a) Teste de normalidade — Shapiro–Wilk:**

```python
_, p_A = stats.shapiro(model_A)
_, p_B = stats.shapiro(model_B)
print(f"p-value do Shapiro para o Modelo A: {p_A:.4f}")
print(f"p-value do Shapiro para o Modelo B: {p_B:.4f}")
```

- Se p ≥ 0,05 → não há evidência contra a normalidade.
- Se p < 0,05 → rejeita-se a hipótese de normalidade (usar teste não paramétrico).

**(b) Teste de homocedasticidade — Levene:**

```python
_, p_var = stats.levene(model_A, model_B)
print(f"p-value de Levene (homogeneidade de variâncias): {p_var:.4f}")
```


***

7.4. **Testes paramétricos em Python**

**(a) Teste t pareado:**
Compara médias de dois conjuntos correlacionados (por exemplo, acurácias em mesmo conjunto de folds).

```python
t_stat, p_val = stats.ttest_rel(model_A, model_B)
print(f"t={t_stat:.3f}, p={p_val:.4f}")
```

Interpretação:

- p < 0,05 → diferença significativa entre os modelos.
- p ≥ 0,05 → diferença não significativa.

**(b) ANOVA (três ou mais modelos):**

```python
model_C = np.array([0.93, 0.94, 0.92, 0.93, 0.94, 0.95, 0.93, 0.94, 0.92, 0.93])
f_stat, p_val = stats.f_oneway(model_A, model_B, model_C)
print(f"F={f_stat:.3f}, p={p_val:.4f}")
```

ANOVA permite verificar se **pelo menos um grupo difere** dos outros.

***

7.5. **Testes não-paramétricos em Python**

Quando os pressupostos de normalidade/homogeneidade são violados:

**(a) Wilcoxon Signed-Rank (amostras pareadas):**

```python
w_stat, p_val = stats.wilcoxon(model_A, model_B)
print(f"Wilcoxon: W={w_stat:.3f}, p={p_val:.4f}")
```

**(b) Mann–Whitney U (amostras independentes):**

```python
u_stat, p_val = stats.mannwhitneyu(model_A, model_B)
print(f"Mann–Whitney U: U={u_stat:.3f}, p={p_val:.4f}")
```

**(c) Kruskal–Wallis (três ou mais grupos):**

```python
h_stat, p_val = stats.kruskal(model_A, model_B, model_C)
print(f"Kruskal–Wallis: H={h_stat:.3f}, p={p_val:.4f}")
```

Esses testes comparam **ranks** das observações em vez de valores numéricos absolutos, sendo adequados para métricas não normalmente distribuídas ou amostras pequenas.

***

7.6. **Visualização e interpretação dos resultados**

Uma boa prática é sempre **visualizar as distribuições** das métricas além de interpretar p-values.

```python
sns.violinplot(data=df)
plt.title("Violin Plot - Distribuição das Acurácias")
plt.show()
```

Adicionalmente, utilizar **gráficos de densidade ou diferença de distribuições** ajuda na interpretação intuitiva da significância:

```python
sns.kdeplot(model_A, shade=True, label="Model A")
sns.kdeplot(model_B, shade=True, label="Model B")
plt.legend()
plt.title("Densidade das Distribuições de Acurácia")
plt.show()
```

Essas visualizações permitem interpretar se as diferenças de desempenho são apenas marginais (pequenos deslocamentos nas curvas) ou substanciais (distribuições separadas).

***

7.7. **Boas práticas de reprodutibilidade e automação**

- Sempre **fixar seeds aleatórias** (`np.random.seed()`) para garantir consistência de execução.
- Empregar **pipelines automatizados** usando `scikit-learn` com métricas coletadas sistematicamente.
- Agregar resultados de múltiplas execuções com **média e desvio padrão** antes dos testes.
- Para múltiplas comparações (mais de dois modelos), adotar **correções de Bonferroni** para controlar erros tipo I.

***

7.8. **Exemplo completo de workflow estatístico**

```python
import numpy as np
from scipy.stats import shapiro, ttest_rel

# Simulação de acurácias
np.random.seed(42)
model_A = np.random.normal(0.91, 0.01, 10)
model_B = np.random.normal(0.89, 0.01, 10)

# 1. Teste de normalidade
print("Normalidade A:", shapiro(model_A)[^1])
print("Normalidade B:", shapiro(model_B)[^1])

# 2. Teste de comparação pareada
t_stat, p_val = ttest_rel(model_A, model_B)
print(f"Resultado: t={t_stat:.3f}, p={p_val:.4f}")

# 3. Interpretação
if p_val < 0.05:
    print("O modelo A tem desempenho significativamente melhor que o B.")
else:
    print("Não há diferença significativa entre os modelos.")
```


***

7.9. **Conclusão do módulo prático**
A implementação computacional de testes estatísticos é uma parte vital do processo de validação experimental em Machine Learning.
Usar ferramentas como `scipy.stats` e `seaborn` não apenas facilita a execução dos testes, mas promove uma **cultura de rigor científico e transparência**.

A automatização dessas análises deve ser incorporada a pipelines de validação, especialmente em comparações de modelos, estudos de hiperparâmetros e relatórios de pesquisa.

***

### 8. Boas Práticas e Interpretação Crítica


***

8.1. **A importância da interpretação crítica**
Aplicar um teste estatístico de forma mecânica é insuficiente — o verdadeiro valor está na **capacidade de interpretar os resultados dentro do contexto experimental**.
Uma diferença de desempenho entre dois modelos pode ser estatisticamente significativa, mas **irrelevante do ponto de vista prático**.
Além disso, resultados significativos podem ser fruto de viés no processo de coleta de dados, amostragem insuficiente ou tuning incorreto.
Portanto, a análise crítica deve sempre considerar:

- A qualidade dos dados e a validade do experimento;
- A robustez estatística do teste (poder, tamanho de amostra, pressupostos);
- O significado prático (impacto no problema real).

***

8.2. **Distinção entre significância estatística e relevância prática**

- **Significância estatística**: indica que o resultado é improvável ao acaso (p < α).
- **Relevância prática**: indica que o resultado traz uma mudança substancial ou útil no contexto da aplicação.

Em Machine Learning, pequenas melhorias em métricas podem ser estatisticamente significativas se a base for grande, mas **não úteis** na prática.
Exemplo: um aumento de **0,2% na acurácia** pode ser significativo (p = 0,01), mas irrelevante num sistema produtivo com custo computacional muito maior.
Por isso, combina-se o **valor-p** com medidas como **tamanho de efeito (Cohen’s d)** e **intervalos de confiança (IC)** para criar uma visão mais balanceada.

***

8.3. **Importância da reprodutibilidade**
Um dos principais objetivos dos testes estatísticos é **garantir que os resultados sejam reproduzíveis**.
Em especial no contexto acadêmico e corporativo, um resultado estatístico sem reprodutibilidade não tem utilidade prática.

Boas práticas de reprodutibilidade incluem:

- Fixar *random seeds* em todas as execuções (`np.random.seed(42)`);
- Documentar claramente datasets, preprocessamentos e parâmetros de modelo;
- Manter controle de versão de experimentos (por exemplo, com ferramentas como **MLflow** ou **Weights \& Biases**);
- Publicar scripts e dados de forma aberta sempre que possível.

***

8.4. **Controle de erro tipo I em comparações múltiplas**
Em experimentos que comparam **vários modelos ou conjuntos de métricas**, o risco de erro tipo I (falso positivo) cresce rapidamente.
Por exemplo, comparar 10 modelos resulta em 45 pares possíveis — com α = 0,05, espera-se pelo menos dois falsos positivos por acaso.
Para mitigar isso, usa-se **correções para comparações múltiplas**, como:

- **Correção de Bonferroni**: divide o nível α pelo número de comparações;
- **Correção de Holm-Bonferroni**: mais flexível e poderosa;
- **False Discovery Rate (FDR)** de Benjamini-Hochberg: controla o percentual esperado de falsos positivos em conjunto.

Essas técnicas ajudam a **preservar a validade global dos resultados**.

***

8.5. **Validação cruzada e incerteza**
Para garantir que o resultado de um teste estatístico não dependa de uma partição específica dos dados, é recomendado realizar **testes sobre métricas obtidas em validação cruzada (k-folds)**.
Dessa forma, cada fold fornece uma estimativa independente do desempenho, e a comparação entre modelos considera a **variabilidade inerente** dos resultados.

Exemplo: coletar acurácias de 10 folds e usar o **teste t pareado** para avaliar se o modelo A é consistentemente melhor que o modelo B.
Esse procedimento reduz ruído e fornece maior estabilidade estatística.

***

8.6. **Integração entre visualização e estatística**
Ferramentas gráficas complementam a interpretação numérica dos testes:

- **Boxplots** e **violin plots** revelam medianas, dispersões e outliers;
- **Histograms** mostram simetrias ou desvios das distribuições;
- **Gráficos de densidade (KDE)** permitem observar sobreposição entre distribuições.

Exemplo prático:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(data=[model_A, model_B])
plt.xticks([0, 1], ['Model A', 'Model B'])
plt.title('Distribuição de Desempenhos')
plt.show()
```

Visualizar as distribuições facilita a comunicação científica e reduz a chance de interpretações erradas baseadas apenas em valores-p.

***

8.7. **Boas práticas para redação e comunicação dos resultados**
Os resultados estatísticos devem ser comunicados de forma precisa, transparente e responsável. Recomenda-se:

1. Reportar valores exatos de p (ex.: p = 0,037, e não apenas “p < 0,05”);
2. Incluir tamanho de efeito, intervalos de confiança e medidas descritivas (média, desvio-padrão);
3. Explicitar o método usado (teste t, Wilcoxon, ANOVA, Kruskal–Wallis, etc.) e o contexto da análise;
4. Evitar afirmações absolutas — deve-se escrever “há evidência de diferença” em vez de “os modelos são diferentes”;
5. Complementar estatísticas com uma interpretação qualitativa, relacionando os resultados aos objetivos do experimento.

***

8.8. **Erros comuns e como evitá-los**

- **Confundir correlação com causalidade:** um resultado estatisticamente significativo não implica causa e efeito.
- **Ignorar pressupostos do teste:** aplicar teste t sem verificar normalidade pode invalidar as conclusões.
- **Desconsiderar outliers:** podem influenciar indevidamente médias e variâncias.
- **Overfitting em estatísticas:** repetir testes até encontrar p < 0,05 caracteriza *p-hacking*, prática cientificamente antiética.

Esses erros são frequentes e comprometem tanto a validade quanto a credibilidade do estudo.

***

8.9. **Recomendações finais para uso em Machine Learning**

- Prefira **comparações pareadas** sempre que possível, pois reduzem a variabilidade interamostral.
- Combine métricas diversas (ex.: F1, ROC-AUC, log-loss) em vez de depender de uma única.
- Controle a **aleatoriedade de split dos dados** e de inicialização de modelos.
- Documente metodologias, testes aplicados, e justificativas estatísticas em relatórios técnicos.
- Considere replicar experimentos sob diferentes condições (datasets, seeds, arquiteturas).

Essas práticas cultivam **rastreabilidade e confiança dos resultados**, fundamentais em pesquisa científica e aplicações industriais.

***

8.10. **Síntese final**
Testes de significância apenas fazem sentido quando aliados à **consistência metodológica e interpretação cuidadosa**.
O rigor estatístico deve sempre vir acompanhado de clareza de comunicação, consciência das limitações e respeito ao contexto experimental.
Em Machine Learning, seguir essas boas práticas fortalece a credibilidade dos resultados e estabelece uma base sólida para comparações, replicações e publicações científicas.

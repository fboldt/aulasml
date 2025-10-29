## 11. Dados Desbalanceados

### 1. Problema do desbalanceamento 

O **problema do desbalanceamento de dados** ocorre quando a proporção entre as classes em um dataset não é uniforme — por exemplo, em um conjunto de 10.000 amostras, se apenas 200 forem positivas (2%), o modelo tende a “aprender” que a classe negativa é a mais provável e, portanto, ignorar os casos positivos. Esse cenário desequilibrado é comum e impacta diretamente o desempenho dos modelos de aprendizado supervisionado.

#### 1.1. Causas típicas do desbalanceamento

O desbalanceamento frequentemente surge de fatores inerentes ao domínio da aplicação:

- **Raridade do evento:** fraudes, doenças raras, falhas em sistemas, etc. são naturalmente escassas.
- **Problemas de coleta:** dados não amostrados de forma aleatória ou com viés de seleção.
- **Erros de anotação:** rótulos incorretos ou omissões em registros históricos.
- **Corte temporal inadequado:** janelas de tempo desiguais podem reduzir observações da classe minoritária.


#### 1.2. Impactos no aprendizado de máquina

O desbalanceamento afeta tanto o treinamento quanto a avaliação:

- **Durante o treinamento**, o modelo é otimizado para minimizar uma função de perda considerando todas as amostras igualmente importantes. Isso faz com que erros sobre a classe minoritária tenham impacto mínimo na atualização dos pesos.
- **Durante a avaliação**, métricas globais como acurácia e erro médio falham em representar o desempenho real. Por exemplo, um classificador que sempre prediz a classe negativa pode ter alta acurácia, mas recall nulo na positiva.


#### 1.3. Exemplos práticos

- **Detecção de fraudes financeiras:** menos de 1% das transações são fraudulentas; detectar fraudes exige maximizar recall da classe positiva, mesmo com aumento de falsos positivos.
- **Diagnóstico médico:** doenças raras apresentam poucas ocorrências, e falsos negativos podem ser críticos.
- **Recomendação de churn:** clientes que cancelam são minoria, e detectar corretamente essa classe é vital para o negócio.


#### 1.4. Análise exploratória em datasets desbalanceados

Antes de aplicar técnicas de correção, é essencial realizar uma análise exploratória detalhada:

- Verificar a **proporção entre classes** usando gráficos de barras ou `value_counts()`.
- Avaliar **distribuições de features por classe**, para entender se há separabilidade.
- Observar **padrões temporais** (em dados temporais, o desbalanceamento pode variar ao longo do tempo).
- Calcular **correlações entre features e classe minoritária**, buscando possíveis preditores.

Exemplo de diagnóstico em Python:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='classe', data=df)
plt.title('Distribuição de Classes')
plt.show()

print(df['classe'].value_counts(normalize=True))
```


#### 1.5. Desafios adicionais

- **Overfitting após oversampling:** gerar duplicatas ou exemplos sintéticos pode causar sobreajuste.
- **Generalização fraca:** em casos extremos (ex: 0,1% de positivos), modelos supervisionados podem falhar totalmente.
- **Avaliação enviesada:** técnicas de cross-validation padrão podem não preservar a proporção de classes, devendo usar *Stratified K-Fold*.
- **Dependência de contexto:** a importância das métricas varia conforme o domínio — em saúde, recall é o mais crítico; em segurança bancária, PR-AUC é mais informativo.


#### 1.6. Relação com aprendizado de custos diferenciados

O desbalanceamento está intimamente ligado à ideia de **aprendizado sensível ao custo**, onde o erro em uma classe é penalizado mais fortemente. Em vez de corrigir os dados, ajusta-se a função de perda. Essa abordagem é eficaz em modelos como regressão logística, SVM e redes neurais, utilizando parâmetros como `class_weight='balanced'` no Scikit-learn ou ajustes de loss manual no TensorFlow e PyTorch.


### 2. Estratégias de balanceamento 

O tratamento de **dados desbalanceados** envolve o uso de estratégias que buscam equilibrar a distribuição das classes sem comprometer a integridade dos dados. A escolha da técnica depende do tamanho do conjunto, da natureza do problema e do impacto esperado sobre o modelo. As estratégias principais estão agrupadas em **reamostragem dos dados**, **aprendizado sensível ao custo** e **geração de dados sintéticos**.

***

#### 2.1. Reamostragem dos dados

A reamostragem busca modificar o conjunto de treinamento para tornar as classes mais equilibradas. Existem dois tipos principais:

##### a) Oversampling (aumento da classe minoritária)

Consiste em **aumentar o número de instâncias** da classe minoritária.
Técnicas:

- **Random Oversampling:** duplica aleatoriamente exemplos minoritários até atingir o equilíbrio.
- **SMOTE (Synthetic Minority Oversampling Technique):** gera exemplos sintéticos interpolando amostras reais e seus vizinhos próximos.
Exemplo: Se $x_1$ e $x_2$ são amostras próximas, cria-se uma nova amostra $$ x_{\text{new}} = x_1 + \lambda (x_2 - x_1) $$ onde $\lambda \in $.
- **ADASYN (Adaptive Synthetic Sampling):** dá mais peso a regiões difíceis, isto é, onde há poucos exemplos minoritários rodeados por instâncias da classe majoritária.

**Vantagens:** preserva todos os exemplos existentes e melhora a diversidade na classe minoritária.
**Desvantagens:** risco de overfitting, principalmente em dados de baixa variabilidade.

##### b) Undersampling (redução da classe majoritária)

Remove instâncias da classe majoritária para balancear o conjunto.
Técnicas:

- **Random Undersampling:** elimina exemplos aleatoriamente.
- **Tomek Links:** identifica pares de instâncias de classes diferentes que são vizinhas imediatas e remove a majoritária, limpando fronteiras ambíguas.
- **NearMiss:** seleciona exemplos da classe majoritária que estão mais próximos das amostras minoritárias para preservar a estrutura da fronteira de decisão.

**Vantagens:** reduz o custo computacional e acelera o treinamento.
**Desvantagens:** possível perda de informação relevante e distorção da distribuição real.

##### c) Métodos híbridos

Combinam oversampling e undersampling, buscando equilibrar diversidade e representatividade.

- **SMOTEENN:** aplicação de SMOTE seguida da limpeza de pares usando Edited Nearest Neighbors (ENN).
- **SMOTETomek:** combinação de geração sintética com remoção de pontos redundantes segundo o critério de Tomek Links.

Essas abordagens são particularmente úteis em conjuntos grandes, onde oversampling puro pode ser redundante e undersampling puro causaria perda de dados.

***

#### 2.2. Aprendizado sensível ao custo (*Cost-Sensitive Learning*)

Em vez de alterar os dados, o modelo é treinado **levando em conta diferentes custos de erro**.
Cada classe recebe um peso proporcional à sua importância ou rareza. Assim, o erro ao classificar a classe minoritária incorretamente tem maior penalização na função de perda.

Exemplo prático com **Scikit-learn**:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

O parâmetro `class_weight='balanced'` ajusta os pesos automaticamente com base na frequência relativa das classes.

Modelos avançados, como **XGBoost**, **LightGBM** e **SVMs**, possuem parâmetros equivalentes (`scale_pos_weight`, `class_weight`).
Essa técnica é apropriada quando se deseja preservar o conjunto original e minimizar distorções estatísticas.

***

#### 2.3. Geração de dados sintéticos

Além do SMOTE e ADASYN, abordagens modernas exploram **modelos generativos** para sintetizar instâncias artificiais da classe minoritária:

- **Variational Autoencoders (VAEs):** aprendem a distribuição latente dos dados e geram novos exemplos.
- **GANs (Generative Adversarial Networks):** treinam um gerador e um discriminador, produzindo exemplos sintéticos altamente realistas.

Essas abordagens são especialmente úteis em aplicações complexas, como imagens médicas e séries temporais, onde os padrões da classe minoritária são muito específicos.

***

#### 2.4. Considerações práticas

- O **balanceamento deve ser aplicado apenas sobre o conjunto de treino**, nunca sobre o conjunto de teste, para evitar vazamento de informação.
- Nem sempre é desejável atingir um equilíbrio perfeito — é preferível otimizar métricas como F1 ou PR-AUC.
- Testar várias combinações de técnicas é recomendável, especialmente com validação estratificada.

***

#### 2.5. Comparativo das técnicas

| Abordagem | Tipo | Vantagem principal | Risco ou Limitação |
| :-- | :-- | :-- | :-- |
| Random Oversampling | Dados | Simples e eficaz em conjuntos pequenos | Overfitting |
| SMOTE / ADASYN | Dados | Cria exemplos realistas | Pode gerar outliers |
| Random Undersampling | Dados | Reduz tempo de treino | Perda de informação |
| Tomek / NearMiss | Dados | Melhora fronteiras de decisão | Balanceamento parcial |
| Cost-Sensitive Learning | Modelo | Preserva dados originais | Necessita calibração dos custos |
| GAN / VAE Synthetic | Dados | Geração sofisticada | Complexidade e custo de treino |





### 3. Métricas apropriadas 

Em **dados desbalanceados**, a escolha das métricas de avaliação é um dos fatores mais críticos no sucesso do modelo. A acurácia, amplamente utilizada em contextos balanceados, pode ser enganosa — um modelo que prevê sempre a classe majoritária pode atingir alta acurácia, mas ser ineficaz para o objetivo real (detectar a classe minoritária). Por isso, métricas que analisam o desempenho por classe ou ponderam erros de forma assimétrica são essenciais.

***

#### 3.1. Acurácia e suas limitações

A acurácia ($ accuracy $) mede a proporção de predições corretas: $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$ onde:
- $ TP $: verdadeiros positivos,
- $ TN $: verdadeiros negativos,
- $ FP $: falsos positivos,
- $ FN $: falsos negativos.

Em datasets desbalanceados, essa métrica tende a refletir principalmente o desempenho na classe majoritária. Por exemplo, em um dataset com 95% de negativos, prever “negativo” para tudo dá 95% de acurácia, mas recall nulo para a classe positiva.

***

#### 3.2. Métricas baseadas em classes positivas

Essas métricas focam na classe minoritária, geralmente a de maior interesse:

- **Precision (precisão):**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Mede o quanto das predições positivas são corretas — importante em contextos onde falsos positivos são custosos (ex: diagnóstico médico).
- **Recall (sensibilidade):**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Mede a proporção de casos positivos detectados — crítico em aplicações que não toleram a omissão de verdadeiros positivos, como segurança e saúde.
- **F1-Score:**

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

É a média harmônica entre precisão e recall, equilibrando ambos os aspectos.

Essas métricas são particularmente úteis em classificadores binários, mas podem ser estendidas para multiclasses com **médias macro** (equilíbrio entre classes) ou **médias ponderadas** (ponderadas pelo suporte de cada classe).

***

#### 3.3. Métricas agregadas por amostra

- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
Mede a capacidade do modelo de distinguir entre as classes, mostrando a relação entre **taxa de verdadeiros positivos** (TPR) e **taxa de falsos positivos** (FPR).
Um classificador aleatório produz AUC = 0,5; perfeito, AUC = 1.
- **Precision-Recall AUC (PR-AUC):**
É mais informativa que ROC-AUC em casos de alto desbalanceamento. Representa a área sob a curva que relaciona precisão e recall. Quando há poucos positivos, PR-AUC tende a oferecer uma leitura mais confiável da efetividade do modelo.
- **G-mean (Geometric Mean):** $$ G = \sqrt{\text{TPR} \times \text{TNR}} $$ 
Onde TNR (True Negative Rate) = $ \frac{TN}{TN + FP} $.
Essa métrica busca o equilíbrio entre sensibilidade e especificidade.

***

#### 3.4. Matriz de confusão e análise detalhada

A **matriz de confusão** permite visualizar onde o modelo erra:


|  | Previsto Positivo | Previsto Negativo |
| :-- | :-- | :-- |
| **Real Positivo** | TP | FN |
| **Real Negativo** | FP | TN |

Ela fornece uma base para derivar todas as métricas anteriores. Em problemas desbalanceados, deve-se analisar cuidadosamente as células TP e FN, já que a classe minoritária (positiva) é a mais afetada por falsos negativos.

Ferramentas como `ConfusionMatrixDisplay` do scikit-learn ajudam a avaliar visualmente o desempenho:

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize='true')
```


***

#### 3.5. Threshold tuning e curvas de decisão

Modelos probabilísticos, como regressão logística ou árvores, podem ajustar o **threshold de decisão** (padrão 0.5) para otimizar métricas específicas.
Por exemplo:

- Aumentar o threshold reduz falsos positivos (↑ precisão, ↓ recall);
- Diminuí-lo reduz falsos negativos (↑ recall, ↓ precisão).

Com a função `precision_recall_curve` é possível encontrar o ponto de equilíbrio ideal. Esse controle fino é crucial em pipelines sensíveis, como detecção de falhas ou segurança.

***

#### 3.6. Métricas de classes com múltiplas categorias

Em cenários multiclasse, escolhe-se como combinar métricas:

- **Macro average:** média simples entre classes (trata todas igualmente);
- **Weighted average:** média ponderada pelo número de amostras por classe;
- **Micro average:** agrega todos os dados antes do cálculo (boa para classes desbalanceadas moderadas).

Exemplo em Python:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=3))
```


***

#### 3.7. Boas práticas de avaliação

- Preferir **PR-AUC** e **F1** para datasets severamente desbalanceados.
- Comparar resultados com **baselines** simples (ex: modelo aleatório, classe majoritária).
- Usar **validação estratificada** para preservar proporções de classe.
- Complementar métricas com **interpretação de importância de features**, ajudando a entender vieses no modelo.

***

Essas métricas, combinadas com técnicas de reamostragem e aprendizado sensível ao custo, permitem avaliar modelos de forma justa e informativa — garantindo que a performance seja medida sob as condições reais do problema.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: conteudo-machine-learning.md


### 4. Pipeline experimental 

O **pipeline experimental para dados desbalanceados** tem como objetivo estruturar o processo de modelagem de forma sistemática, garantindo **reprodutibilidade, validação justa e comparabilidade entre técnicas de balanceamento e algoritmos**. Ele envolve desde o pré-processamento até a avaliação final com métricas adequadas.

***

#### 4.1. Estrutura geral do pipeline

Um pipeline típico de experimentação em machine learning com classes desbalanceadas envolve as seguintes etapas sequenciais:

1. **Preparação dos dados:**
    - Leitura e limpeza de dados (tratamento de outliers, valores faltantes, encoding).
    - Análise de distribuição de classes (`value_counts`).
    - Separação dos conjuntos **treino** e **teste**, normalmente com `StratifiedSplit` para preservar proporções.
2. **Reamostragem:**
    - Aplicação de métodos como **SMOTE**, **ADASYN** ou **undersampling** somente sobre o conjunto de treino.
    - Justificativa: evita vazamento de informação do teste para o modelo.
    - Ferramentas: `imblearn.over_sampling.SMOTE`, `imblearn.under_sampling.TomekLinks`, ou pipelines de `imblearn`.
3. **Treinamento do modelo:**
    - Seleção de algoritmos adequados (ex: `RandomForest`, `XGBoost`, `LogisticRegression` com `class_weight`).
    - Ajuste de hiperparâmetros e validação cruzada estratificada.
    - Teste de diferentes representações de dados para avaliar robustez.
4. **Avaliação do modelo:**
    - Uso de **métricas robustas** (F1, ROC-AUC, PR-AUC, G-mean).
    - Visualização da **matriz de confusão normalizada**.
    - Curvas ROC e Precision-Recall para inspecionar trade-offs entre recall e precisão.
5. **Comparação e seleção:**
    - Comparar modelos com e sem balanceamento.
    - Analisar qual combinação de técnica de reamostragem e algoritmo atinge melhor desempenho.
    - Opcionalmente, aplicar **testes estatísticos** para verificar diferenças significativas entre performances (explicado na Aula 12).

***

#### 4.2. Integração no Scikit-learn/Imbalanced-learn

O pacote **`imbalanced-learn`** integra-se com **`scikit-learn`** por meio de pipelines combinados:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipe, X, y, scoring=['f1', 'roc_auc'], cv=cv)
print(scores)
```

Essa configuração garante que a reamostragem ocorra **dentro de cada fold** da validação cruzada, evitando vazamento de dados e viés na avaliação.

***

#### 4.3. Escolha de algoritmos

Em contextos desbalanceados, alguns algoritmos se adaptam melhor devido à sua capacidade de lidar com pesos de classe:

- **Modelos Lineares:** Regressão logística, SVM com `class_weight='balanced'`.
- **Árvores e ensembles:** Random Forest e Gradient Boosting (XGBoost, LightGBM) permitem ajuste via `scale_pos_weight`.
- **Métodos probabilísticos e bayesianos:** úteis para priorização de riscos (ex: diagnóstico médico).
- **Modelos de deep learning:** podem incorporar **funções de perda ponderadas**, como *weighted binary cross-entropy*.

Em redes neurais, uma função de perda ponderada pode ser expressa como:

$$
L = -w_1 y \log(p) - w_0 (1 - y) \log(1 - p)
$$

onde $ w_1 $ e $ w_0 $ são pesos inversamente proporcionais à frequência das classes.

***

#### 4.4. Controle de thresholds

A etapa de **threshold tuning** é essencial após o treino de modelos probabilísticos. Ajustar o limite de decisão (normalmente 0,5) pode melhorar significativamente o recall ou F1-score:

```python
from sklearn.metrics import f1_score
y_probas = model.predict_proba(X_test)[:, 1]

for t in [0.3, 0.4, 0.5, 0.6]:
    y_pred = (y_probas >= t).astype(int)
    print(f'Threshold={t}, F1={f1_score(y_test, y_pred):.3f}')
```

Essa análise ajuda a escolher o ponto ótimo entre falso positivo e falso negativo, conforme o contexto do problema (segurança, saúde, finanças etc.).

***

#### 4.5. Avaliação e documentação

Para que os resultados sejam confiáveis e replicáveis:

- Use **validação estratificada** em todos os experimentos.
- Fixe *random seeds* para reprodutibilidade.
- Armazene resultados (ex: métricas, hiperparâmetros, tempo de execução) em planilhas ou bancos experimentais (`mlflow`, `wandb`).
- Registre as versões de pacotes e configuração do ambiente.

***

#### 4.6. Visualização e análise final

Curvas e relatórios ajudam a interpretar a performance do modelo:

- **Curva ROC:** Avalia discriminabilidade global;
- **Curva Precision-Recall:** Indica desempenho na classe minoritária;
- **Lift chart e Gain chart:** Úteis para priorização (por exemplo, top 10% mais prováveis de fraude).

Exemplo de plotagem:

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
```


***

#### 4.7. Conclusões práticas

Um pipeline experimental bem estruturado para dados desbalanceados deve:

- Isolar o balanceamento no conjunto de treino.
- Usar validação estratificada e métricas robustas.
- Testar pipelines variados de dados e modelos.
- Documentar sistematicamente as decisões e resultados.

Com esse processo disciplinado, o aluno adquire uma base sólida para **analisar problemas desbalanceados de forma justa, interpretável e reprodutível**, alinhada às boas práticas científicas em machine learning.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: conteudo-machine-learning.md


### 5. Estudos de caso 

Os **estudos de caso** em dados desbalanceados são essenciais para conectar teoria e prática, permitindo compreender como técnicas e métricas se aplicam a contextos reais. A seguir, cada caso ilustra um tipo de desafio frequente na indústria e na pesquisa, enfatizando estratégias específicas para lidar com o desbalanceamento de classes.

***

#### 5.1. Estudo de caso: detecção de fraude financeira

**Contexto:**
Em transações financeiras online, menos de 1% dos registros são fraudulentos, gerando um severo desbalanceamento. O objetivo é maximizar a detecção de fraudes (recall) sem causar excesso de falsos positivos.

**Pipeline aplicado:**

1. **Pré-processamento:**
    - Feature engineering com variáveis derivadas (valores de transação, tempo entre transações, geolocalização) e codificação categórica.
2. **Balanceamento:**
    - Aplicação de **SMOTE** combinada com **undersampling** da classe majoritária para preservar variabilidade.
3. **Modelagem:**
    - Modelos testados: Random Forest, XGBoost e Regressão Logística com `class_weight='balanced'`.
4. **Avaliação:**
    - Métricas principais: **PR-AUC** e **F1-score**.
    - Aumento de **20%-30%** no F1 quando comparado ao treino sem reamostragem.
    - Ajuste de threshold baseado em curva Precision-Recall para controlar custo operacional.

**Principais aprendizados:**
A combinação de **validação estratificada**, métricas adaptadas e reamostragem sintética pode reduzir significativamente a taxa de falsos negativos, sem comprometer demais a precisão. O uso de ROC-AUC isoladamente mascarava o baixo desempenho real em fraudes.

***

#### 5.2. Estudo de caso: churn de clientes (telecomunicações)

**Contexto:**
A retenção de clientes em serviços de assinatura (telefonia, internet, streaming) é um clássico problema desbalanceado: apenas 10–15% cancelam o serviço a cada período.

**Pipeline aplicado:**

1. **Engenharia de features:**
    - Variáveis de uso (minutos, tráfego de dados, suporte técnico), além de histórico de pagamento e satisfação.
2. **Balanceamento:**
    - Uso de **RandomOverSampler** e **SMOTETomek** para reforçar a classe “churn”.
3. **Algoritmos:**
    - Gradient Boosting e CatBoost com otimização de hiperparâmetros via validação cruzada estratificada.
4. **Avaliação:**
    - **Precision ≥ 0.7**, **Recall ≥ 0.8**, **G-mean ≈ 0.75** — resultado satisfatório para campanhas preventivas.
    - Interpretação via SHAP para identificar fatores decisivos: queda de uso e tempo de contrato foram as variáveis mais correlatas.

**Principais aprendizados:**
O foco em recall é mais adequado para ações de retenção, e técnicas de importância de features ajudam a compreender quais fatores preveem cancelamentos, possibilitando intervenções direcionadas.

***

#### 5.3. Estudo de caso: diagnóstico médico de doenças raras

**Contexto:**
Em aplicações médicas supervisionadas (ex: detecção de doenças raras como esclerose lateral amiotrófica), as classes positivas costumam ter menos de 5% dos exemplos.

**Pipeline aplicado:**

1. **Pré-processamento:**
    - Normalização de dados e imputação de valores faltantes críticos (exames laboratoriais).
2. **Balanceamento:**
    - Aplicação de **ADASYN**, priorizando amostras minoritárias próximas a fronteiras de decisão.
    - Envolvimento de especialistas clínicos para validação das novas instâncias sintéticas (essencial para evitar erros médicos).
3. **Modelagem:**
    - SVM com kernel RBF e `class_weight`, comparado a redes neurais com *loss* ponderada.
4. **Avaliação:**
    - **Recall (sensibilidade)** e **F1-score** como métricas principais.
    - A análise de falsos negativos foi usada como indicador de risco clínico.
5. **Resultado final:**
    - Melhor recall com o modelo neural ponderado (≈ 0.92) e leve queda na precisão (≈ 0.70), considerado aceitável em contexto de triagem médica.

**Principais aprendizados:**
Em problemas críticos, o custo de falsos negativos supera largamente o de falsos positivos; métricas devem refletir as consequências práticas, e o envolvimento interdisciplinar é indispensável.

***

#### 5.4. Comparativo dos três contextos

| Domínio | Fraude Financeira | Churn de Clientes | Saúde |
| :-- | :-- | :-- | :-- |
| Proporção minoritária | <1% | 10–15% | 3–5% |
| Técnica de reamostragem | SMOTE + Undersampling | SMOTETomek | ADASYN |
| Algoritmo principal | XGBoost | CatBoost | SVM / Rede Neural |
| Métrica-chave | PR-AUC | Recall / G-mean | Sensibilidade (Recall) |
| Risco crítico | Falsos negativos | Falsos negativos | Falsos negativos |
| Benefício adicional | Redução de perdas financeiras | Ação preventiva de retenção | Suporte à decisão clínica |


***

#### 5.5. Síntese conceitual dos estudos

Esses casos demonstram que **o problema do desbalanceamento não é técnico apenas, mas estratégico**. Cada domínio exige priorizar diferentes métricas e métodos conforme o impacto de erros:

- **Financeiro:** otimização de custo e precisão de triagem.
- **Marketing:** balancear recall e custo de retenção.
- **Saúde:** maximizar sensibilidade para salvar vidas.

Independentemente da aplicação, é crucial:

- Combinar técnicas de reamostragem com validação estratificada.
- Escolher métricas que reflitam o contexto do problema.
- Monitorar o modelo em produção, pois o desbalanceamento pode mudar no tempo (conceito de **data drift**).



### 6. Leituras recomendadas 

As **leituras recomendadas** da Aula 11 fornecem uma base teórica sólida e aplicada sobre o tratamento de **dados desbalanceados** e **métricas avançadas de avaliação**. Cada obra complementa um aspecto do conteúdo abordado — desde fundamentos matemáticos até práticas modernas de implementação.

***

#### 6.1. Kapelner \& Toth (2020) — *Classification in Imbalanced Datasets*

**Foco:** fundamentos estatísticos e experimentais do desbalanceamento de classes.
**Contribuições principais:**

- Apresenta a natureza probabilística do desbalanceamento e discute como os estimadores se tornam enviesados quando as proporções das classes não refletem a realidade populacional.
- Propõe análises sobre o efeito da reamostragem (oversampling e undersampling) em estimadores de probabilidade.
- Explora métodos de *reweighting* e *recalibration* pós-treinamento.
- Discute a relação entre prior de classe, Bayes Risk e custo esperado de decisão — essencial para entender o aprendizado sensível ao custo.

**Aplicação recomendada:** referência indispensável para alunos interessados em compreender as consequências formais do desbalanceamento nas funções de perda e inferência estatística.

***

#### 6.2. Charu C. Aggarwal — *Neural Networks and Deep Learning* (Capítulo 8: Data Challenges)

**Foco:** desafios estruturais de dados em redes neurais.
**Contribuições principais:**

- Detalha o impacto de datasets desbalanceados na **propagação de gradiente** e na otimização de funções de perda.
- Introduz o conceito de **class weighting** em redes neurais, demonstrando matematicamente como pesos ajudam a equilibrar gradientes.
- Apresenta técnicas modernas de regularização supervisionadas pela frequência das classes.
- Inclui estratégias de amostragem em mini-batches balanceados e métodos de *focal loss*, amplamente usados em visão computacional.

**Aplicação recomendada:** leitura avançada para estudantes que investigam modelos de *deep learning* aplicados a domínios como saúde e detecção de eventos raros.

***

#### 6.3. François Chollet — *Deep Learning with Python (2ª Edição)*, Capítulo 4 (“Model Evaluation and Metrics”)

**Foco:** práticas de modelagem e avaliação no contexto de redes neurais e aprendizado profundo.
**Contribuições principais:**

- Explica como definir e interpretar métricas sensíveis ao desbalanceamento (como F1 e PR-AUC) em frameworks como Keras e TensorFlow.
- Apresenta exemplos práticos de gráficos de curvas ROC e Precision-Recall integrados a *callbacks* de treinamento.
- Demonstra a importância de monitorar múltiplas métricas simultaneamente para evitar overfitting à métrica incorreta.
- Introduz o uso de *custom metrics* e *class weights* integrados às funções de perda (`model.compile(loss='binary_crossentropy', metrics=[...])`).

**Aplicação recomendada:** indicado para quem deseja dominar o controle de métricas e validação no ambiente Keras/TensorFlow.

***

#### 6.4. Seth Weidman — *Deep Learning from Scratch* (Capítulo 6: Improving Model Robustness)

**Foco:** melhoria da robustez de modelos supervisionados frente a desafios reais.
**Contribuições principais:**

- Descreve o impacto do desbalanceamento em gradientes e distribuições de saída.
- Explica técnicas de *data augmentation*, *bootstrapping* e regularização sob cenários desbalanceados.
- Discute ajustes em *batch normalization* e *learning rates* em presença de classes raras.
- Destaca práticas experimentais e comparações com validações estratificadas.

**Aplicação recomendada:** útil para quem quer projetar modelos consistentes e avaliar experimentalmente a sensibilidade a variações de distribuição dos dados.

***

#### 6.5. Leituras complementares sugeridas

Além das obras principais, recomenda-se:

- **He \& Garcia (2009). "Learning from Imbalanced Data"**, *IEEE Transactions on Knowledge and Data Engineering.*
Clássico artigo que estruturou as bases da pesquisa moderna sobre aprendizado com dados desbalanceados e deu origem ao SMOTE.
- **Buda et al. (2018). "A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks"**, *Neural Networks Journal.*
Analisa empiricamente o comportamento de CNNs sob diferentes desbalanceamentos, com recomendações práticas de normalização e perda focal.
- **Fernández et al. (2018). "Learning from Imbalanced Data Sets: Data, Algorithms and Applications"**, *Springer.*
Uma visão abrangente de métodos heurísticos e híbridos de reamostragem, incluindo abordagens para dados estruturados e textuais.

***

#### 6.6. Síntese das leituras

| Fonte | Enfoque principal | Aplicação |
| :-- | :-- | :-- |
| Kapelner \& Toth (2020) | Estatística e riscos de decisão | Modelagem teórica e análise de métricas |
| Aggarwal (2018) | Neural networks e weighting | Redes neurais e balanceamento interno |
| Chollet (2021) | Métricas e práticas Keras | Implementação experimental de métricas |
| Weidman (2020) | Robustez experimental | Avaliação prática e tuning de modelos |
| He \& Garcia (2009) | Fundamentos clássicos | Introdução e SMOTE |
| Buda et al. (2018) | CNNs e estudo empírico | Visão aplicada a deep learning moderno |


***

#### 6.7. Relação com o conteúdo da Aula 11

Essas leituras complementam cada parte da aula:

- **Tópico 1 (problema do desbalanceamento):** Kapelner \& Toth oferecem o embasamento estatístico.
- **Tópico 2 (estratégias de balanceamento):** He \& Garcia e Fernández et al. detalham variações e aplicações.
- **Tópico 3 (métricas avançadas):** Chollet enfatiza PR-AUC e F1-score em implementações práticas.
- **Tópico 4 (pipeline experimental):** Weidman demonstra ajustes robustos em pipelines de treino.
- **Tópico 5 (estudos de caso):** Aggarwal fornece exemplos de redes neurais aplicadas a contextos críticos.

***

Cada uma dessas obras contribui para formar um **arcabouço teórico-prático completo**, capacitando o aluno a compreender, implementar e avaliar soluções robustas para problemas de aprendizado supervisionado com dados desbalanceados — o principal objetivo formativo da **Aula 11**.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: conteudo-machine-learning.md


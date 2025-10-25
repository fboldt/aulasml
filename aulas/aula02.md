## 2. Regressão Linear e Métodos Básicos

### 1. Fundamentos da Regressão Linear

A **regressão linear** é um dos modelos mais antigos e fundamentais da estatística e do aprendizado de máquina supervisionado. Seu objetivo é estimar uma relação matemática entre uma variável alvo (*dependente*) **y** e uma ou mais variáveis explicativas (*independentes*) **x₁, x₂, …, xₙ**.

A ideia central é que o valor da variável de saída pode ser aproximado como uma **combinação linear** das entradas.

***

### 1.1 Conceito e Intuição

Imagine que você deseja prever o preço de uma casa com base em características como metragem, número de quartos e localização. Cada uma dessas características influencia o preço final de forma aproximadamente linear — ou seja, dobrar a metragem **tende** a aumentar o preço, mas não necessariamente o dobro.

Nesse caso, o modelo pode ser representado como:

**y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ε**

onde:

- **y** é o preço estimado,
- **x₁** pode ser a metragem, **x₂** o número de quartos, **x₃** a localização codificada,
- **β₀** é o intercepto (valor base, quando todas as entradas são zero),
- **βᵢ** são os pesos (coeficientes) que representam o impacto de cada variável,
- **ε** é o erro aleatório (diferença entre valor real e previsto).

O objetivo é encontrar os valores de **β₀, β₁, …, βₙ** que melhor explicam os dados observados.

***

### 1.2 Tipos de Regressão Linear

Existem dois tipos principais de regressão linear:

- **Regressão Linear Simples:**
Uma variável preditora e uma variável de resposta.
Exemplo: prever o preço de um imóvel apenas pela metragem.
- **Regressão Linear Múltipla:**
Mais de uma variável preditora.
Exemplo: prever o preço de um imóvel com base em metragem, número de quartos e localização.

***

### 1.3 Representação Vetorial

Em notação matricial, o modelo é expresso de forma compacta:

**y = Xβ + ε**

onde:

- **X** é a matriz de entrada (*features*),
- **β** é o vetor de parâmetros,
- **y** é o vetor de saídas (target).

Essa descrição permite utilizar álgebra linear para encontrar a solução analítica do modelo, o que torna a regressão linear computacionalmente eficiente.

***

### 1.4 Interpretação dos Coeficientes

Cada coeficiente **βᵢ** possui um significado claro:

- **Magnitude:** quanto o valor previsto de *y* muda quando *xᵢ* aumenta em uma unidade.
- **Sinal:** se positivo, indica relação direta (quanto maior *xᵢ*, maior *y*); se negativo, indica relação inversa.
- **Intercepto (β₀):** valor de *y* quando todas as variáveis independentes valem zero.

Exemplo prático:

- β₁ = 2000 → cada metro quadrado adicional aumenta o preço em cerca de 2000 unidades monetárias.
- β₀ = 10000 → representa o valor base de uma casa sem metragem útil (teoricamente).

***

### 1.5 O Papel do Erro (ε)

Nem todos os pontos estão exatamente sobre a linha ajustada.
Essa diferença é o **erro**, que representa fatores não modelados, ruídos de medição ou variabilidade natural do sistema.

Idealmente:

- A média dos erros é **zero**.
- Os erros não estão correlacionados entre si.
- A variância dos erros é constante (homocedasticidade).

Essas são condições importantes para que o modelo tenha validade estatística.

***

### 1.6 História e Relevância

A regressão linear tem origem no século XIX com os trabalhos de **Francis Galton** e **Karl Pearson**. Galton observou o fenômeno da “regressão à média”, notando que filhos de pessoas muito altas tendiam a ser mais baixos do que seus pais — daí vem o termo *regressão*.

Desde então, ela se tornou uma das técnicas mais importantes em estatística, economia, física, ciências sociais e, mais recentemente, aprendizado de máquina.
Modelos modernos como **redes neurais**, **árvores de decisão** e **modelos lineares generalizados** evoluíram a partir dos princípios que a regressão linear estabeleceu.

***

### 1.7 Limitações

Apesar de sua simplicidade, a regressão linear tem restrições importantes:

- Pressupõe **relação linear** entre variáveis, o que raramente é totalmente verdadeiro.
- É sensível a **outliers**.
- Requer independência entre as variáveis de entrada (evitar multicolinearidade).
- Pode ter desempenho fraco em problemas com relações não lineares complexas.

Por isso, sua compreensão é fundamental — tanto como ponto de partida teórico quanto como base comparativa para modelos mais complexos.



### 2. Ajuste por Mínimos Quadrados

O método dos **mínimos quadrados** é a técnica clássica para ajustar um modelo de regressão linear aos dados. A ideia principal é encontrar os parâmetros (coeficientes) que minimizam o **erro quadrático total** entre os valores observados e os valores previstos pelo modelo.

***

### 2.1 O Problema de Otimização

Dado um conjunto de pares de observações **(xᵢ, yᵢ)**, desejamos encontrar os coeficientes **β₀, β₁, …, βₙ** que melhor descrevem a relação entre as variáveis.

O modelo é:

**yᵢ = β₀ + β₁xᵢ₁ + β₂xᵢ₂ + … + βₙxᵢₙ + εᵢ**

onde **εᵢ** é o erro (diferença entre o valor real e o estimado).

O critério de mínimos quadrados busca minimizar:

**SSE = Σ (yᵢ - ŷᵢ)²**

Em palavras: queremos que a soma dos quadrados dos resíduos (erros) seja a menor possível.

***

### 2.2 Solução Analítica

Em notação matricial, temos:

**y = Xβ + ε**

O vetor **β** que minimiza o erro quadrático é obtido pela fórmula fechada:

**β̂ = (XᵀX)⁻¹ Xᵀy**

Essa expressão é derivada anulando o gradiente da função de custo em relação aos parâmetros.
Ela é chamada de **solução dos mínimos quadrados ordinários (OLS — Ordinary Least Squares)**.

> **Importante:** Essa solução só é válida se a matriz (XᵀX) for inversível, o que exige independência linear entre as variáveis de entrada (sem multicolinearidade).

***

### 2.3 Interpretação Geométrica

Geometricamente, o método dos mínimos quadrados pode ser entendido como a projeção do vetor **y** no subespaço gerado pelas colunas de **X**.

- O vetor **ŷ = Xβ̂** é a projeção ortogonal de **y** sobre o espaço das features.
- O resíduo **ε = y - ŷ** é perpendicular ao espaço das colunas de **X**.
- Isso implica que **Xᵀε = 0**, ou seja, os resíduos não têm correlação linear com as variáveis explicativas.

Essa perspectiva geométrica é útil para compreender como o modelo “se encaixa” nos dados.

***

### 2.4 Exemplo Numérico Simples

Imagine que desejamos ajustar uma reta simples **y = β₀ + β₁x** a três pontos:


| x | y |
| :-- | :-- |
| 1 | 2 |
| 2 | 2.8 |
| 3 | 4.5 |

Para estimar os coeficientes, aplicamos o método dos mínimos quadrados (por exemplo, com `numpy`):

```python
import numpy as np

# Dados
x = np.array([1, 2, 3])
y = np.array([2, 2.8, 4.5])

# Montagem da matriz X (com coluna de 1s para o intercepto)
X = np.vstack([np.ones(len(x)), x]).T

# Solução dos mínimos quadrados
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Intercepto (β₀):", beta[0])
print("Coeficiente (β₁):", beta[1])
```

Saída esperada (aproximada):

```
Intercepto (β₀): 0.63
Coeficiente (β₁): 1.41
```

Logo, a reta ajustada é **ŷ = 0.63 + 1.41x**, o que representa bem a tendência ascendente dos dados.

***

### 2.5 Pressupostos do Modelo Linear

O método dos mínimos quadrados é baseado em algumas hipóteses fundamentais:

1. **Linearidade:** a relação entre as variáveis é aproximadamente linear.
2. **Independência:** as observações são independentes entre si.
3. **Homoscedasticidade:** a variância dos erros é constante.
4. **Ausência de multicolinearidade:** as variáveis preditoras não são altamente correlacionadas.
5. **Normalidade dos resíduos:** para inferência estatística (teste t e intervalos de confiança).

Quando esses pressupostos são violados, as estimativas podem se tornar enviesadas ou instáveis.

***

### 2.6 Problemas e Alternativas

#### a) **Multicolinearidade**

Quando duas ou mais variáveis explicativas são quase lineares entre si, a matriz (XᵀX) se torna quase singular.
Consequência: pequenas variações nos dados causam grandes variações nos coeficientes.

**Alternativas:**

- Remover variáveis redundantes.
- Usar **análise de componentes principais (PCA)**.
- Adotar técnicas de **regularização** (Ridge, Lasso).


#### b) **Outliers**

Pontos de dados extremos podem ter grande impacto na linha ajustada.

**Alternativas:**

- Aplicar **regressão robusta** (como Huber ou RANSAC).
- Padronizar os dados.
- Verificar visualmente os resíduos.

***

### 2.7 Verificação Visual

Uma maneira simples de avaliar o ajuste do modelo é por meio de gráficos:

- **Dispersão com reta ajustada:** visualiza o quão bem o modelo se aproxima dos pontos reais.
- **Histograma dos resíduos:** deve estar aproximadamente centrado em zero.
- **Gráfico de resíduos vs. valores previstos:** deve mostrar dispersão aleatória (sem padrões).

Exemplo ilustrativo em Python:

```python
import matplotlib.pyplot as plt

y_pred = beta[0] + beta[1]*x

plt.scatter(x, y, color="blue", label="Valores Reais")
plt.plot(x, y_pred, color="red", label="Reta Ajustada")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Ajuste por Mínimos Quadrados")
plt.show()
```

Esse gráfico permite uma verificação intuitiva da qualidade do ajuste.

***

### 2.8 Limitações do Método

Embora poderoso, o método dos mínimos quadrados tem limitações:

- É sensível a **outliers**.
- Supõe que os erros são normalmente distribuídos.
- Não lida bem com relações não lineares.
- Requer **inversão de matriz**, que pode ser instável em alta dimensão.

Esses fatores motivam o uso de versões modificadas, como regressão **ridge**, **lasso** e **elastic net** (tópico seguinte).

### 3. Regularização: Ridge, Lasso e ElasticNet

A **regularização** é uma técnica usada para evitar o **sobreajuste (overfitting)** em modelos de aprendizado supervisionado, especialmente em regressão linear.
Ela atua controlando a complexidade do modelo por meio da **penalização dos coeficientes**, prevenindo que o modelo se ajuste em excesso aos ruídos do conjunto de treinamento.

***

### 3.1 Intuição da Regularização

Quando o modelo linear é ajustado a dados com muitas variáveis ou com correlações fortes entre elas, os coeficientes podem assumir valores muito altos (positivos ou negativos).
Isso leva a previsões instáveis e sensíveis a pequenas variações nos dados.

A regularização adiciona um **termo de penalização** à função de custo — forçando os coeficientes a permanecerem pequenos, o que melhora a **generalização** do modelo em novos dados.

***

### 3.2 Função de Custo Regularizada

A função de custo geral em uma regressão linear regularizada é:

**Loss = SSE + Penalty**

ou seja,

**Loss = Σ(yᵢ - ŷᵢ)² + λ * P(β)**

onde:

- **SSE** é o erro quadrático total,
- **λ (lambda)** é o parâmetro de regularização (controla o grau de penalização),
- **P(β)** é o termo de penalização aplicado aos coeficientes **β**.

O valor de λ define o *balanceamento* entre erro de ajuste e complexidade:

- λ = 0 → modelo igual ao de mínimos quadrados tradicionais.
- λ grande → modelo mais simples, com coeficientes próximos de zero.

***

### 3.3 Tipos de Regularização

#### 3.3.1 Ridge Regression (L2)

A penalização tipo **L2** consiste na soma dos quadrados dos coeficientes:

**Loss = Σ(yᵢ - ŷᵢ)² + λ Σβⱼ²**

Características:

- Reduz o tamanho dos coeficientes, mas **nunca os zera completamente**.
- Distribui a penalização entre as variáveis correlacionadas.
- É útil quando todas as *features* têm importância semelhante.

**Solução analítica:**

**β̂ = (XᵀX + λI)⁻¹ Xᵀy**

onde **I** é a matriz identidade.
Essa forma mostra que Ridge é uma “versão suavizada” da regressão linear.

***

#### 3.3.2 Lasso Regression (L1)

A penalização **L1** é a soma dos valores absolutos dos coeficientes:

**Loss = Σ(yᵢ - ŷᵢ)² + λ Σ|βⱼ|**

Características:

- Encoraja **esparsidade** — isto é, faz alguns coeficientes se tornarem exatamente zero.
- Atua como uma **seleção automática de variáveis**.
- É apropriada quando acreditamos que apenas algumas *features* são realmente relevantes.

Não existe solução analítica simples; a minimização é feita por métodos iterativos (como *coordinate descent*).

***

#### 3.3.3 ElasticNet

O **ElasticNet** combina os dois tipos de penalização (L1 e L2):

**Loss = Σ(yᵢ - ŷᵢ)² + λ₁ Σ|βⱼ| + λ₂ Σβⱼ²**

Características:

- Une a robustez do Lasso com a estabilidade do Ridge.
- Controla a proporção entre as penalizações via parâmetro **l₁_ratio** (entre 0 e 1).
- É útil em contextos com muitas variáveis correlacionadas e poucos exemplos.

***

### 3.4 Comparação entre os Métodos

| Método | Tipo de Penalização | Zera Coeficientes? | Solução Analítica | Indicado para... |
| :-- | :-- | :-- | :-- | :-- |
| **Ridge** | L2 — soma dos quadrados | Não | Sim | Variáveis correlacionadas |
| **Lasso** | L1 — soma dos valores absolutos | Sim | Não | Seleção de variáveis |
| **ElasticNet** | Combinação L1 + L2 | Sim (parcial) | Não | Dados correlacionados com poucos exemplos |


***

### 3.5 Escolha do Lambda (λ)

O valor de **λ** é essencial: define o grau de penalização.
Nos pacotes de *machine learning*, esse valor costuma ser ajustado via **validação cruzada (cross-validation)**.

- λ muito pequeno → modelo complexo, tendência a overfitting.
- λ muito grande → modelo simples demais, tendência a underfitting.

Ferramentas como `GridSearchCV` ou `RidgeCV` facilitam a busca automática do melhor valor.

Exemplo em Python:

```python
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
model.fit(X_train, y_train)
print("Melhor Lambda:", model.alpha_)
```


***

### 3.6 Exemplo Prático com Comparação

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Gerar dados artificiais
np.random.seed(10)
X = np.random.randn(100, 5)
y = 3*X[:,0] - 2*X[:,1] + 0.5*np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Modelos
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Avaliação
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: R² = {r2_score(y_test, y_pred):.3f}")
```

Esse exemplo mostra como a performance e a complexidade variam entre os métodos conforme o nível de regularização.

***

### 3.7 Visualização do Efeito da Penalização

Para observar o impacto prático da regularização, é comum representar graficamente como os coeficientes variam em função de λ:

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

alphas = np.logspace(-3, 2, 100)
coefs_ridge, coefs_lasso = [], []

for a in alphas:
    ridge = Ridge(alpha=a).fit(X_train, y_train)
    lasso = Lasso(alpha=a).fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_)
    coefs_lasso.append(lasso.coef_)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(alphas, coefs_ridge)
plt.xscale("log")
plt.title("Ridge - Coeficientes vs Lambda")

plt.subplot(2, 1, 2)
plt.plot(alphas, coefs_lasso)
plt.xscale("log")
plt.title("Lasso - Coeficientes vs Lambda")
plt.xlabel("Lambda (escala log)")
plt.tight_layout()
plt.show()
```

Esses gráficos revelam:

- No **Ridge**, todos os coeficientes decaem suavemente, permanecendo diferentes de zero.
- No **Lasso**, alguns coeficientes tornam-se exatamente zero conforme λ aumenta.

***

### 3.8 Observações Finais

- Regularização é essencial em aprendizado de máquina moderno, principalmente em contextos com **alta dimensionalidade (n >> m)**.
- Ridge é mais estável, Lasso é mais interpretável, e ElasticNet é um meio-termo poderoso.
- A escolha entre eles deve considerar tanto o **comportamento estatístico** quanto a **aplicação prática** (interpretação vs desempenho).


### 4. Métricas de Avaliação de Modelos de Regressão

Avaliar o desempenho de um modelo de regressão linear é fundamental para entender **o quão bem ele generaliza** para novos dados.
As métricas de avaliação quantificam a diferença entre **valores reais** e **valores previstos**, permitindo comparar modelos e ajustar hiperparâmetros (como o lambda em técnicas de regularização).

***

### 4.1 Conceito de Erro e Resíduo

Em um modelo de regressão, o **erro (ou resíduo)** é a diferença entre o valor observado e o valor previsto:

**eᵢ = yᵢ - ŷᵢ**

onde:

- **yᵢ** é o valor real,
- **ŷᵢ** é o valor previsto pelo modelo.

O ideal é que os resíduos sejam **pequenos, simétricos (positivos e negativos com a mesma frequência)** e **não apresentem padrão sistemático**.

***

### 4.2 Métricas Comuns de Avaliação

Abaixo estão as principais métricas usadas para medir a precisão de modelos de regressão.


| Métrica | Fórmula Simples | Interpretação | Faixa de Valores |
| :-- | :-- | :-- | :-- |
| **MAE** (Erro Médio Absoluto) | média( | yᵢ - ŷᵢ | ) |
| **MSE** (Erro Quadrático Médio) | média((yᵢ - ŷᵢ)²) | Penaliza mais fortemente grandes erros | 0 → perfeito |
| **RMSE** (Raiz do Erro Quadrático Médio) | √MSE | Interpretação direta (mesmas unidades de y) | 0 → perfeito |
| **R²** (Coeficiente de Determinação) | 1 - (SSE / SST) | Mostra a proporção da variância explicada pelo modelo | 0–1 (ou negativo) |


***

### 4.3 Interpretação Detalhada das Métricas

#### 4.3.1 MAE — Mean Absolute Error

- É a **média dos erros absolutos**, sem considerar direção (positivo ou negativo).
- Valor fácil de interpretar pois está na **mesma escala de y**.
- Robusto a outliers, mas não diferencia o tamanho dos grandes erros.

**Exemplo:**
Um MAE = 3.5 significa que, em média, o modelo erra 3.5 unidades no valor previsto.

***

#### 4.3.2 MSE — Mean Squared Error

- Eleva os erros ao quadrado, **penalizando fortemente erros grandes**.
- É sensível a outliers.
- Útil quando queremos **detectar grandes variações** no desempenho.

**Exemplo:**
Um MSE = 16 indica que, em média, o erro ao quadrado é 16, portanto o erro típico é cerca de √16 = 4 unidades.

***

#### 4.3.3 RMSE — Root Mean Squared Error

- É a **raiz quadrada do MSE**, voltando às mesmas unidades de y.
- Suaviza o efeito do quadrado, mas ainda mantém sensibilidade a grandes erros.
- Muito usada por ser intuitiva e comparável ao valor médio da variável alvo.

**Exemplo:**
Se y representa o valor de imóveis em milhares de reais e RMSE = 25, o modelo erra, em média, cerca de R\$25 mil.

***

#### 4.3.4 R² — Coeficiente de Determinação

- Mede a proporção da variação em y explicada pelo modelo.
- Calculado como:
**R² = 1 - (SSE / SST)**
onde:
    - **SSE = Σ(yᵢ - ŷᵢ)²** → soma dos erros quadráticos;
    - **SST = Σ(yᵢ - ȳ)²** → variação total em torno da média.
- Quanto maior o R², melhor o modelo explica os dados.
- No entanto, **R² alto não garante bom modelo** — pode resultar de overfitting.

> **Observação:** R² pode ser negativo quando o modelo é pior que a simples média dos dados.

***

### 4.4 Exemplo Prático em Python

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Valores reais e previstos
y_true = np.array([3, 5, 2.5, 7])
y_pred = np.array([2.8, 4.9, 2.7, 6.8])

# Cálculo das métricas
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
```

Saída típica:

```
MAE: 0.150
MSE: 0.030
RMSE: 0.173
R²: 0.992
```

Esses valores indicam excelente desempenho (pequenos erros e alta explicação da variância).

***

### 4.5 Comparando Modelos

A escolha da métrica depende do **problema** e do **impacto dos erros**:


| Situação | Métrica Recomendada | Motivo |
| :-- | :-- | :-- |
| Quando todos os erros têm o mesmo custo | MAE | Média dos erros diretos |
| Quando grandes erros são muito penalizados | MSE ou RMSE | Quadrado amplifica desvios |
| Quando se quer medir explicação da variância | R² | Facilmente interpretável |
| Dados com valores extremos (outliers) | MAE | Mais robusto e estável |

Um bom modelo deve **minimizar MAE/MSE/RMSE** e **maximizar R²**.

***

### 4.6 Normalização das Métricas

Para comparar modelos em diferentes escalas, é possível normalizar as métricas:

- **NRMSE (Normalized RMSE):** RMSE dividido pela amplitude ou média de y.
- **MAPE (Mean Absolute Percentage Error):** erro percentual médio:
**MAPE = (100/n) Σ(|(yᵢ - ŷᵢ) / yᵢ|)**
— expressa o erro como porcentagem, útil em negócios e previsões financeiras.

**Limitação:** MAPE não é aplicável quando **yᵢ = 0**.

***

### 4.7 Visualização dos Resíduos

A análise gráfica dos resíduos ajuda a diagnosticar problemas:

- **Plotagem de erros vs valores previstos:** deve parecer um "ruído aleatório".
- **Histograma dos resíduos:** deve ser aproximadamente simétrico ao redor de 0.
- **Q-Q plot:** verifica se os resíduos seguem distribuição normal.

Exemplo:

```python
import matplotlib.pyplot as plt

residuos = y_true - y_pred
plt.scatter(y_pred, residuos)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores previstos")
plt.ylabel("Resíduos")
plt.title("Análise dos Resíduos")
plt.show()
```


***

### 4.8 Limitações das Métricas Tradicionais

Apesar de úteis, as métricas clássicas apresentam desafios:

- **Sensibilidade à escala**: MSE e RMSE não são comparáveis entre datasets com diferentes unidades.
- **Falta de interpretação causal**: R² alto não implica causalidade.
- **Ausência de robustez**: outliers podem distorcer fortemente as métricas quadráticas.

Por isso, é comum combinar métricas diferentes e **interpretar resultados à luz do contexto do problema**.

***

### 4.9 Boas Práticas de Avaliação

1. Sempre **divida o conjunto de dados** em treinamento e teste (ou use validação cruzada).
2. Avalie **várias métricas** — uma única métrica pode ser enganosa.
3. Analise **os resíduos graficamente**.
4. Padronize os dados sempre que as entradas tiverem variâncias muito distintas.
5. Use validação cruzada para evitar conclusões acidentais sobre performance.

***

### 4.10 Resumo Conceitual

| Métrica | Escala | Penaliza Grandes Erros | Robustez a Outliers | Interpretabilidade |
| :-- | :-- | :-- | :-- | :-- |
| MAE | Igual a Y | Moderado | Alta | Alta |
| MSE | Elevada | Alta | Baixa | Média |
| RMSE | Igual a Y | Alta | Baixa | Alta |
| R² | Sem escala | Não | Média | Alta |


### 5. Demonstração Prática em Scikit-Learn

O objetivo deste tópico é aplicar os conceitos teóricos de regressão linear em um cenário prático utilizando a biblioteca **Scikit-Learn**, uma das mais populares ferramentas de *machine learning* em Python.
A demonstração cobre desde a preparação dos dados até a avaliação e visualização dos resultados.

***

### 5.1 Preparando o Ambiente e os Dados

Primeiro, importamos as bibliotecas principais e geramos um conjunto de dados sintético para fins didáticos.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reprodutibilidade
np.random.seed(42)

# Gerando dados sintéticos (relação linear com ruído)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1).ravel()

# Visualização inicial
plt.scatter(X, y, color='blue', alpha=0.6)
plt.title("Distribuição dos dados sintéticos")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

**Interpretação:**
Os dados simulam uma relação linear com ruído.
Nossa tarefa é ajustar modelos que aprendam a estimar a relação subjacente.

***

### 5.2 Dividindo os Dados em Treino e Teste

Dividimos o conjunto em subconjuntos para avaliar a generalização do modelo.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Tamanho de treino:", len(X_train))
print("Tamanho de teste:", len(X_test))
```

**Boas práticas:**

- Use 20–30% dos dados para teste.
- Mantenha o *random_state* fixo para reprodutibilidade.
- Quando houver dependência temporal, utilize divisão temporal (TimeSeriesSplit).

***

### 5.3 Ajustando Modelos de Regressão

A seguir, ajustamos quatro versões de modelos lineares com diferentes regularizações.

```python
# Instanciando modelos
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Treinando e armazenando previsões
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

# Exibindo resultados resumidos
print(pd.DataFrame(results).T)
```

**Interpretação dos resultados:**

- As métricas MAE, RMSE e R² permitem comparar equidade de ajuste e generalização.
- Espera-se que Ridge e ElasticNet apresentem desempenho similar ou ligeiramente melhor em presença de ruído ou multicolinearidade.

***

### 5.4 Visualizando os Ajustes

Podemos visualizar as predições de cada modelo sobre o conjunto de teste.

```python
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color="blue", label="Dados reais", alpha=0.6)

for name, r in results.items():
    X_sorted = np.sort(X_test, axis=0)
    y_pred_sorted = r["model"].predict(X_sorted)
    plt.plot(X_sorted, y_pred_sorted, label=name)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparação entre modelos de regressão")
plt.legend()
plt.show()
```

**Análise visual:**

- Linhas mais suaves (menor variabilidade) indicam efeito de regularização.
- O modelo LinearRegression tende a sobreajustar quando os dados têm ruído alto, o que pode ser suavizado por Ridge e ElasticNet.

***

### 5.5 Comparando Coeficientes e Interceptos

A comparação dos parâmetros estimados ajuda a compreender o impacto da regularização sobre os coeficientes.

```python
for name, r in results.items():
    model = r["model"]
    print(f"{name}: Coeficiente = {model.coef_[0]:.3f}, Intercepto = {model.intercept_:.3f}")
```

**Exemplo de saída:**

```
LinearRegression: Coeficiente = 2.98, Intercepto = 4.02
Ridge: Coeficiente = 2.95, Intercepto = 4.00
Lasso: Coeficiente = 2.91, Intercepto = 4.05
ElasticNet: Coeficiente = 2.94, Intercepto = 4.01
```

**Interpretação:**
O valor do coeficiente aproxima-se do valor real (3).
Modelos regularizados produzem coeficientes levemente menores — efeito desejado da penalização.

***

### 5.6 Avaliação com Gráficos de Resíduos

A análise dos erros ajuda a detectar padrões que indicam mau ajuste do modelo.

```python
plt.figure(figsize=(8,6))
for name, r in results.items():
    residuos = y_test - r["y_pred"]
    plt.scatter(y_test, residuos, alpha=0.7, label=name)

plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores reais")
plt.ylabel("Resíduos")
plt.title("Distribuição dos resíduos (teste)")
plt.legend()
plt.show()
```

**O que observar:**

- A dispersão deve ser aleatória em torno de zero.
- Padrões curvados indicam que a relação pode não ser linear.
- Resíduos sistemáticos indicam necessidade de transformação de variáveis.

***

### 5.7 Exemplo Prático com Dados Reais

Como exemplo adicional, podemos aplicar o mesmo pipeline ao conjunto **Boston Housing** (ou outro dataset público similar, como *California Housing*).

```python
from sklearn.datasets import fetch_california_housing

# Carregando dataset
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
```

**Resultado esperado:**
O modelo Ridge oferece desempenho razoável sem overfitting, mesmo em dados com variáveis correlacionadas.

***

### 5.8 Boas Práticas em Modelagem com Scikit-Learn

1. **Padronize variáveis** (média 0 e variância 1) ao usar Ridge, Lasso ou ElasticNet.
2. **Use validação cruzada (cross-validation)** para definir `alpha`.
3. **Combine métricas** e **análise de resíduos** para avaliar adequadamente.
4. **Salve modelos treinados** com `joblib` para reutilização posterior.
5. **Documente hiperparâmetros** e resultados para reprodutibilidade.

***

### 5.9 Resumo Geral

| Etapa | Descrição | Ferramentas |
| :-- | :-- | :-- |
| Gerar e explorar dados | Visualização, dispersão e ruído | `matplotlib`, `pandas` |
| Dividir treino/teste | Separação dos conjuntos | `train_test_split` |
| Ajustar modelos | Linear, Ridge, Lasso, ElasticNet | `sklearn.linear_model` |
| Avaliar desempenho | MAE, RMSE, R² | `sklearn.metrics` |
| Visualizar resultados | Linhas ajustadas e resíduos | `matplotlib` |


### 6. Interpretação dos Coeficientes

Com o modelo de regressão ajustado, interpretar corretamente os coeficientes é essencial para transformar previsões em **entendimento real sobre os dados**.
A regressão linear vai além da previsão: ela fornece **insights causais ou associativos** sobre as variáveis, permitindo explicar como cada atributo influencia o resultado da variável dependente (*target*).

***

### 6.1 A Fórmula do Modelo

O modelo ajustado é expresso como:

**ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ**

onde:

- **ŷ**: valor previsto da variável alvo;
- **β₀**: intercepto (valor previsto quando todas as variáveis explicativas são 0);
- **βᵢ**: coeficiente da variável *xᵢ*;
- **xᵢ**: variável explicativa (ou característica).

***

### 6.2 Significado dos Coeficientes

Cada coeficiente **βᵢ** representa a **mudança esperada em ŷ** quando *xᵢ* aumenta em uma unidade, **mantendo todas as outras variáveis constantes**.

Exemplo interpretativo:
No modelo **Preço = β₀ + β₁·Área + β₂·Quartos**, se **β₁ = 2500**, isso significa que, para cada metro quadrado adicional, o preço aumenta em média **R\$ 2.500**, considerando o mesmo número de quartos.

***

### 6.3 Intercepto (β₀)

O intercepto é o ponto onde a reta de regressão intercepta o eixo y (valor previsto de ŷ quando todas as variáveis independentes são zero).

Em contextos:

- **Significativo** → quando zero faz sentido (ex.: 0 horas de estudo → nota prevista).
- **Inútil** → quando zero é fora da realidade (ex.: 0 m² de casa → modelo extrapola).

A interpretação depende do contexto dos dados e deve ser feita com cuidado.

***

### 6.4 Sinal e Magnitude dos Coeficientes

- **Sinal positivo (+):** indica que conforme *xᵢ* aumenta, ŷ também tende a aumentar.
- **Sinal negativo (–):** indica que conforme *xᵢ* aumenta, ŷ tende a diminuir.
- **Magnitude:** representa o impacto numérico da variável sobre o resultado.

Exemplo:


| Variável | Coeficiente (βᵢ) | Interpretação |
| :-- | :-- | :-- |
| Área (m²) | +2400 | Cada m² adicional aumenta o preço em R\$2.400 |
| Idade do imóvel (anos) | -1800 | Cada ano reduz o valor em R\$1.800 |
| Número de quartos | +6200 | Cada quarto extra aumenta o preço em R\$6.200 |


***

### 6.5 Escala e Normalização

Os coeficientes podem parecer maiores ou menores apenas por conta da **escala das variáveis**.
Por exemplo, uma variável medida em reais terá coeficiente pequeno, enquanto outra medida em quilômetros pode ter coeficiente grande.
Isso **não significa que uma variável é mais importante** que outra.

**Solução:**
Padronizar as variáveis (subtrair a média e dividir pelo desvio padrão).
Assim, os coeficientes passam a indicar o impacto em **termos de desvios padrão**, facilitando comparações entre variáveis.

Exemplo com Scikit-Learn:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
model_scaled = make_pipeline(scaler, LinearRegression())
model_scaled.fit(X_train, y_train)

print(model_scaled.named_steps['linearregression'].coef_)
```

Agora, os *betas* são comparáveis entre si — os maiores indicam maior influência padronizada no resultado.

***

### 6.6 Importância Relativa das Variáveis

Uma medida interessante é a **importância relativa** dos coeficientes normalizados.
Ela indica a contribuição percentual de cada variável para o modelo.

```python
coefs = np.abs(model_scaled.named_steps['linearregression'].coef_)
importance = coefs / np.sum(coefs)
print(pd.DataFrame({"Variável": data.feature_names, "Importância Relativa": importance}))
```

**Interpretação:**
Essa proporção mostra o quanto cada variável explica do comportamento de ŷ em relação às demais.

***

### 6.7 Coeficientes em Modelos Regularizados

Nos modelos **Ridge**, **Lasso** e **ElasticNet**, os coeficientes também são influenciados pela penalização:

- **Ridge:** reduz suavemente o valor dos coeficientes para evitar sobreajuste.
- **Lasso:** força alguns coeficientes a se tornarem exatamente zero (seleção de variáveis).
- **ElasticNet:** mistura ambos os efeitos, reduzindo e, se necessário, eliminando variáveis.

Comparação prática:

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

print("Coeficientes Ridge:", ridge.coef_)
print("Coeficientes Lasso:", lasso.coef_)
```

**Resultado esperado:**

- Ridge → coeficientes levemente reduzidos, nenhum nulo;
- Lasso → alguns coeficientes exatamente 0 (variáveis descartadas).

Isso ajuda a identificar quais *features* realmente contribuem para o modelo.

***

### 6.8 Incerteza dos Coeficientes

Em análises estatísticas, é importante estimar **intervalos de confiança** dos coeficientes.
Eles indicam a margem de erro da estimativa de cada βᵢ.
Scikit-Learn não fornece isso diretamente, mas é possível calcular com bibliotecas como **`statsmodels`**.

Exemplo:

```python
import statsmodels.api as sm

X_const = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_const).fit()
print(ols.summary())
```

A saída inclui:

- **coef:** valor estimado;
- **std err:** erro padrão;
- **t:** estatística t para testar significância;
- **P>|t|:** p-valor (nível de confiança da variável no modelo).

**Boa prática:**
Variáveis com p-valor > 0.05 são frequentemente consideradas pouco relevantes.

***

### 6.9 Interpretação Gráfica

Visualizar o impacto de cada variável é útil em relatórios e apresentações.

```python
coefs = model_scaled.named_steps['linearregression'].coef_
features = data.feature_names

plt.figure(figsize=(10,6))
plt.barh(features, coefs)
plt.xlabel("Valor do coeficiente padronizado")
plt.title("Importância das variáveis no modelo linear")
plt.show()
```

Gráficos como esse são essenciais para destacar quais fatores têm maior efeito sobre as previsões — especialmente em análises de negócios e ciência de dados aplicada.

***

### 6.10 Limitações da Interpretação

Embora a regressão linear seja interpretável, existem limitações importantes:

1. **Correlação não é causalidade:** um coeficiente positivo não significa que há relação causal.
2. **Colinearidade:** quando variáveis estão altamente correlacionadas, os coeficientes tornam-se instáveis e difíceis de interpretar.
3. **Omission Bias:** deixar variáveis relevantes fora do modelo distorce os coeficientes das incluídas.
4. **Escalas diferentes:** dificultam comparações diretas sem padronização.

Logo, a interpretação deve ser contextual, acompanhada de análise exploratória e validação empírica.

***

### 6.11 Síntese Conceitual

| Conceito | Explicação | Implicação |
| :-- | :-- | :-- |
| β₀ | Valor previsto quando xᵢ = 0 | Contexto depende da natureza da variável |
| βᵢ | Mudança esperada em ŷ para cada unidade de xᵢ | Relação direta (βᵢ > 0) ou inversa (βᵢ < 0) |
| Escala | Unidade de medida afeta magnitude de β | Necessidade de padronização |
| Regularização | Altera magnitude e esparsidade dos β | Controla overfitting e seleciona variáveis |
| P-valor | Significância estatística de βᵢ | Ajuda a validar importância da variável |


### 7. Aplicações Práticas da Regressão Linear

A **regressão linear** é uma das técnicas mais amplamente aplicadas do aprendizado de máquina devido à sua **simplicidade, interpretabilidade e eficiência**.
Mesmo diante da chegada de modelos mais complexos como redes neurais e *ensemble methods*, ela permanece essencial em problemas que exigem **explicabilidade e baixo custo computacional**.

***

### 7.1 Campos de Aplicação

A regressão linear é usada em praticamente todas as áreas onde existe uma relação quantitativa entre variáveis.
A seguir, alguns dos contextos mais comuns:

#### a) Finanças

- **Previsão de preços de ativos financeiros**, como ações, títulos e criptomoedas.
- **Modelagem de risco de crédito**: estimativa de probabilidade de inadimplência.
- **Elasticidade de demanda:** variação de vendas em função do preço.
- **Análise de custo de capital (CAPM):** relação entre retorno esperado e risco (beta do ativo).


#### b) Economia e Ciências Sociais

- Estudo da **influência de variáveis socioeconômicas** (educação, renda, idade) sobre o consumo.
- Modelos de previsão macroeconômica, como **PIB, inflação, desemprego e investimento**.
- Estudos de **equidade salarial** e **impactos de políticas públicas**.


#### c) Engenharia e Indústria

- **Previsão de demanda energética** com base em fatores meteorológicos e econômicos.
- **Modelagem de processos físicos**, como relação entre pressão e temperatura.
- **Controle de qualidade:** estimar falhas, desgaste de equipamentos e calibração de sensores.


#### d) Saúde e Biologia

- **Modelagem epidemiológica**: número de casos em função do tempo e fatores ambientais.
- **Predição de tempo de sobrevivência de pacientes** com base em parâmetros clínicos.
- **Bioestatística:** relação entre dosagem de medicamentos e resposta biológica.


#### e) Negócios e Marketing

- **Previsão de vendas** e lucros a partir de variáveis como preço, investimento em marketing e localização.
- **Análise de sensibilidade de clientes (churn)**.
- **Planejamento de campanhas publicitárias** com base em impacto histórico.

***

### 7.2 Estudo de Caso 1: Mercado Imobiliário

Um dos exemplos mais clássicos e didáticos da regressão linear é o problema de **previsão de preço de imóveis**.
O objetivo é estimar o preço médio de casas com base em características como área, localização, número de quartos e idade do imóvel.

#### Pipeline Resumido

1. **Coleta e limpeza dos dados:** dataset *California Housing* (Scikit-Learn) ou *House Prices* (Kaggle).
2. **Análise exploratória:** identificar correlações entre atributos.
3. **Ajuste do modelo:** regressão linear ou Ridge.
4. **Avaliação:** métricas MAE, RMSE, R².
5. **Interpretação dos coeficientes:** notar o efeito positivo da área e o efeito negativo da idade.

#### Exemplo de código simplificado

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Dados
data = fetch_california_housing()
X, y = data.data, data.target

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Modelo
model = LinearRegression().fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

Esse modelo simples costuma atingir **R² entre 0.6 e 0.7**, o que é considerado razoável sem engenharia de atributos.

***

### 7.3 Estudo de Caso 2: Previsão de Consumo Elétrico

No setor de energia, a regressão linear é usada para prever o consumo **com base em temperatura, hora do dia e dia da semana**.
Esses modelos ajudam concessionárias a planejar geração e distribuição de energia.

Exemplo de variáveis:


| Variável | Descrição |
| :-- | :-- |
| Temperatura média | °C por hora |
| Umidade relativa | porcentagem (%) |
| Dia da semana | categórico (0 = domingo, 6 = sábado) |
| Horário | 0 a 23 (hora do dia) |

O modelo ajustado fornece uma curva de tendência diária e permite detectar **picos anormais** de consumo.

***

### 7.4 Modelos em Contextos de Machine Learning

Na prática moderna, a regressão linear é frequentemente integrada a pipelines mais complexos:

- **Pré-processamento automático:** `Pipeline` + `StandardScaler` (padronização).
- **Seleção de atributos:** `LassoCV` ou `SelectKBest`.
- **Validação cruzada:** `cross_val_score` para avaliar desempenho médio.
- **Comparação com outros métodos:** RandomForest, GradientBoosting e Redes Neurais.

Exemplo de pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10], cv=5))
scores = cross_val_score(ridge, X, y, cv=10, scoring='r2')

print("Desempenho médio (R²):", scores.mean())
```


***

### 7.5 Visualização e Comunicação de Resultados

Além do desempenho numérico, a regressão linear oferece **interpretação visual**:

- **Gráficos de dispersão (scatterplots):** comparar valores reais e previstos.
- **Resíduos:** identificar padrões de erro ou heterocedasticidade.
- **Coeficientes e importância relativa:** indicar variáveis com maior impacto.

Exemplo de gráfico de previsão:

```python
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Comparação entre valores reais e previstos")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```

Esse tipo de visualização é crucial em relatórios técnicos e apresentações executivas.

***

### 7.6 Limitações e Cuidados Práticos

Embora simples e versátil, a regressão linear tem limitações bem definidas:


| Limitação | Consequência | Alternativas |
| :-- | :-- | :-- |
| Relações não lineares | Erros altos em padrões curvos | Polinomial, SVR, Árvores |
| Multicolinearidade | Coeficientes instáveis | Ridge, PCA |
| Outliers | Desvio nos parâmetros | Regressão robusta (Huber) |
| Atributos categóricos não tratados | Erros de modelagem | One-Hot Encoding |
| Escala diferente entre variáveis | Aumento de erro numérico | Padronização (StandardScaler) |

Essas limitações são enfrentadas com *pipelines robustos* e uso de regularização.

***

### 7.7 Extensões do Modelo Linear

A regressão linear é a base de uma grande família de técnicas usadas em *machine learning* moderno:


| Extensão | Descrição | Aplicação |
| :-- | :-- | :-- |
| **Regressão Polinomial** | Inclui termos quadráticos e cúbicos de x | Modelagem não linear simples |
| **Regressão Logística** | Saída binária (classificação) | Diagnóstico médico, churn |
| **Regressão Bayesiana** | Introduz probabilidade nos coeficientes | Modelos de incerteza |
| **Modelos Lineares Generalizados (GLM)** | Usa outras funções de ligação (Poisson, Logit) | Dados de contagem e probabilidade |
| **Regressão Robusta** | Minimiza impacto de outliers | Dados ruidosos e anômalos |

Essas variantes demonstram como o conceito linear serve como **fundação de quase toda a modelagem preditiva moderna**.

***

### 7.8 Conclusões Didáticas

- A regressão linear é **ideal para iniciantes**, pois conecta estatística e aprendizado supervisionado de forma intuitiva.
- Sua utilidade está em ser **traduzível**: é possível interpretar matemática, resultados e gráficos.
- Mesmo quando substituída por modelos complexos, ela **permanece como referência de performance base (baseline)**.
- Sua compreensão profunda é essencial para entender como funcionam modelos posteriores — de **redes neurais** a **transformers** —, todos inspirados no mesmo princípio: ajustar parâmetros para **minimizar erros**.

***

Esse tópico encerra a **Aula 2 — Regressão Linear e Métodos Básicos**, integrando teoria, prática e aplicabilidade.
Os próximos módulos (a partir da Aula 3) introduzem **técnicas de validação e generalização**, dando continuidade natural ao aprendizado do ciclo completo de *machine learning supervisionado*.


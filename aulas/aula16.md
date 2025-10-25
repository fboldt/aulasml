## 16. Séries Temporais e Forecasting

### 1. Introdução às Séries Temporais (versão expandida)

As séries temporais são uma das áreas mais relevantes da ciência de dados aplicada, especialmente em contextos onde o tempo é um fator determinante para a dinâmica dos dados. Esse tipo de dado é representado como uma sequência cronológica de observações, geralmente espaçadas em intervalos fixos, como segundos, minutos, horas, dias, meses ou anos.

#### 1.1 Conceito Fundamental

Uma série temporal é formalmente definida como uma sequência de valores \$ \{y_t\}_{t=1}^T \$, onde \$ y_t \$ representa a observação no instante \$ t \$. Ao contrário de datasets tradicionais — em que as amostras são independentes entre si — as séries temporais apresentam dependência temporal; ou seja, os valores passados influenciam diretamente os futuros.

#### 1.2 Exemplos Clássicos

Alguns exemplos típicos incluem:

- Temperaturas médias diárias em uma cidade ao longo de um ano.
- Vendas mensais de um produto.
- Índices econômicos, como PIB trimestral ou taxa de inflação.
- Níveis de tráfego em uma rede de computadores.
- Séries biomédicas, como eletroencefalogramas (EEG) ou batimentos cardíacos.

Esses exemplos mostram a amplitude de aplicações possíveis — abrangendo desde finanças até saúde, energia, transportes e internet das coisas (IoT).

#### 1.3 Importância e Aplicações

A análise de séries temporais desempenha papéis centrais em áreas como:

- Planejamento e previsão comercial (demand forecasting).
- Detecção de anomalias e fraudes (anomaly detection).
- Previsão do consumo energético e eficiência operacional.
- Modelagem de comportamento de mercado e gestão de risco.

Com a crescente disponibilidade de dados temporais em tempo real, a capacidade de entendimento e previsão dessas séries é um diferencial competitivo para organizações e pesquisadores.

#### 1.4 Tipos de Abordagem

Duas interrogações principais guiam a análise de séries temporais:

1. **Modelagem e compreensão:** identificar padrões estruturais (tendência, sazonalidade e ciclos).
2. **Previsão (forecasting):** estimar valores futuros com base na dinâmica histórica.

As abordagens podem ser de natureza:

- **Clássica estatística**, como ARIMA e Suavização Exponencial.
- **Aprendizado mecânico tradicional**, como regressões e árvores de decisão.
- **Aprendizado profundo**, com redes neurais recorrentes, LSTM e Transformers.


#### 1.5 Estacionaridade e Importância do Tempo

Um conceito central é o de *estacionaridade*, isto é, quando estatísticas como média e variância permanecem constantes ao longo do tempo. Muitos modelos (como ARIMA) assumem estacionaridade para gerar previsões adequadas.
O tempo, portanto, não é apenas uma variável adicional: ele define a estrutura e as correlações internas do conjunto de dados, influenciando diretamente o tipo de modelo e de validação aplicados.

#### 1.6 Principais Desafios

- **Autocorrelação:** forte dependência entre observações sequenciais.
- **Sazonalidades múltiplas:** várias periodicidades coexistindo (ex.: hora do dia, dia da semana, mês).
- **Eventos externos (covariáveis):** feriados, campanhas, choques econômicos.
- **Dados ausentes ou ruído:** comuns em medições temporais contínuas.

Compreender essas particularidades é essencial antes de aplicar qualquer técnica preditiva.

***


### 2. Componentes das Séries Temporais (versão expandida)

O entendimento dos componentes que compõem uma série temporal é essencial para sua análise adequada e escolha de modelos de previsão. Esses componentes descrevem como os dados evoluem ao longo do tempo e possibilitam a decomposição da série em partes interpretáveis.

***

#### 2.1 Conceito de Decomposição

A decomposição de uma série temporal consiste em representar os dados como a soma (ou produto) de diferentes componentes fundamentais. Essa estrutura ajuda a separar padrões sistemáticos (como tendência e sazonalidade) de flutuações aleatórias.

As formas mais comuns de decomposição são:

- **Aditiva:**
\$ y_t = T_t + S_t + C_t + R_t \$
usada quando as variações ao longo do tempo são aproximadamente constantes.
- **Multiplicativa:**
\$ y_t = T_t \times S_t \times C_t \times R_t \$
adequada quando as variações crescem proporcionalmente à magnitude dos dados.

***

#### 2.2 Tendência (Trend)

A **tendência** representa o movimento global de longo prazo da série, podendo indicar crescimento, declínio ou estabilidade.

- Pode ser linear (constante no tempo) ou não linear (exponencial, polinomial, logística).
- A presença de tendência indica que o fenômeno observado possui uma direção predominante.
- Métodos de estimação incluem médias móveis, suavização exponencial simples ou regressão polinomial.
- Exemplo: aumento contínuo da temperatura média global ao longo das décadas.

***

#### 2.3 Sazonalidade (Seasonality)

A **sazonalidade** diz respeito a padrões que se repetem em intervalos fixos de tempo, geralmente associados a períodos regulares.

- Ela pode ser anual, mensal, semanal, diária ou até horária.
- Em decomposições, observamos variações previsíveis que se repetem com o mesmo período.
- Exemplo: aumento nas vendas de roupas no inverno ou tráfego da internet mais alto durante o dia.
- Detecção prática: análise de autocorrelação e decomposição via `seasonal_decompose` do `statsmodels`.

***

#### 2.4 Ciclo (Cycle)

O **componente cíclico** reflete flutuações periódicas de longo prazo, mas sem uma periodicidade fixa.

- Está relacionado a fenômenos macroeconômicos, como ciclos de crescimento e recessão.
- Pode durar anos e é geralmente mais suave do que a sazonalidade.
- A análise de ciclos requer séries longas e pode ser feita por filtros como Hodrick-Prescott (HP filter).

***

#### 2.5 Ruído ou Aleatoriedade (Noise)

O **ruído** captura as variações imprevisíveis que não se explicam pelos componentes estruturais.

- Representa eventos aleatórios, erros de medição ou mudanças abruptas não regulares.
- Idealmente, após a modelagem de tendência e sazonalidade, o ruído deve se comportar como ruído branco — ou seja, sem padrão e média zero.
- Em termos práticos, o ruído residual indica a parcela da incerteza que o modelo não conseguiu capturar.

***

#### 2.6 Relação entre os Componentes

Os componentes estão interconectados e a correta identificação deles é essencial para a modelagem.
Um modelo de previsão robusto precisa distinguir:

- O que é tendência estrutural de longo prazo.
- O que é variação sazonal previsível.
- O que é mera flutuação aleatória.

***

#### 2.7 Métodos de Extração dos Componentes

As técnicas mais comuns incluem:

- **Médias móveis:** suavizam a série removendo oscilações curtas.
- **Suavização exponencial (ETS):** ajusta pesos maiores a observações mais recentes.
- **Decomposição clássica (Additive/Multiplicative):** separa explicitamente cada componente.
- **STL (Seasonal-Trend decomposition using Loess):** método robusto para decomposição não linear, amplamente usado em aplicações modernas.

***

#### 2.8 Interpretação Prática

A correta decomposição ajuda a:

1. Compreender a estrutura temporal dos dados.
2. Escolher modelos adequados (como ARIMA quando estacionário, Prophet quando sazonalidade é forte).
3. Analisar padrões sazonais para planejamento estratégico (ex.: promoções, estoque, escalas operacionais).

***


### 3. Análise Exploratória de Dados Temporais (versão expandida)

A análise exploratória de séries temporais (Time Series EDA) é um passo essencial para compreender os padrões, dependências e irregularidades antes da modelagem. Sua função é revelar estrutura, sazonalidade, tendência, autocorrelação e anomalias nos dados de forma visual e estatística.

***

#### 3.1 Objetivos da EDA Temporal

A análise exploratória busca:

1. **Entender a estrutura temporal da série:** identificar padrões recorrentes e variações sistemáticas.
2. **Detectar anomalias e outliers:** pontos fora do padrão podem afetar previsões.
3. **Avaliar estacionaridade:** requisito comum em modelos clássicos (como ARIMA).
4. **Gerar hipóteses para modelagem:** insights sobre sazonalidade, periodicidade e ruído.

Enquanto a EDA tradicional foca em distribuições e correlações entre variáveis independentes, a EDA temporal enfatiza dependências internas entre observações ao longo do tempo.

***

#### 3.2 Visualização de Séries Temporais

A primeira etapa de análise envolve visualizar a série completa e suas transformações.

Principais tipos de gráficos:

- **Gráfico de linha:** forma mais direta de observar padrões de tendência e sazonalidade.
- **Boxplots por período:** mostram a distribuição dos valores por mês, dia da semana ou hora, evidenciando sazonalidades.
- **Heatmaps temporais:** permitem visualizar padrões de intensidade ao longo do tempo (ex.: vendas por hora/dia).
- **Autocorrelograma (ACF):** mostra o grau de correlação entre valores separados por k lags.
- **Partial ACF (PACF):** mede correlações ajustadas, removendo efeitos intermediários, útil para identificar ordens em modelos AR e MA.

***

#### 3.3 Transformações de Séries

Para permitir a análise e modelagem, é comum aplicar transformações:

- **Diferença (Differencing):** remove tendência, tornando a série estacionária.
\$ y'_t = y_t - y_{t-1} \$
- **Log ou Box-Cox:** estabiliza a variância em séries com crescimento exponencial.
- **Padronização e Normalização:** ajusta escala para comparações entre múltiplas séries.
- **Suavização (Smoothing):** remove ruídos, destacando tendência e ciclo via médias móveis ou filtro exponencial.

***

#### 3.4 Feature Engineering Temporal

A engenharia de atributos é vital para algoritmos de aprendizado supervisionado aplicados a séries temporais. Exemplos comuns incluem:

- **Lags:** valores passados usados como preditores (ex.: \$ y_{t-1}, y_{t-2}, ··· \$).
- **Rolling Statistics:** médias e desvios móveis que capturam padrões locais.
- **Decomposição:** inclusão explícita de componentes de tendência e sazonalidade.
- **Variáveis de calendário:** dia da semana, hora, mês, feriados, sazonalidades múltiplas.
- **Mudanças percentuais:** \$ \frac{y_t - y_{t-1}}{y_{t-1}} \$, úteis em séries financeiras.

Essas features ajudam modelos como **regressão linear**, **árvores de decisão** e **redes neurais** a capturar dependências temporais indiretamente.

***

#### 3.5 Análise de Estacionaridade

Uma série estacionária possui média e variância constantes ao longo do tempo.

- **Testes estatísticos:**
    - Teste de Dickey-Fuller Aumentado (ADF).
    - Teste KPSS (Kwiatkowski–Phillips–Schmidt–Shin).
- **Diagnóstico visual:** observar médias móveis e variância ao longo da série.
- Quando não estacionária, a série deve ser diferenciada ou transformada antes da aplicação de modelos clássicos.

***

#### 3.6 Deteção de Outliers e Anomalias

Os outliers podem indicar erros de medição ou eventos importantes (anomalias reais).
Técnicas principais:

- Inspeção visual e boxplots.
- Desvios-padrão sobre médias móveis.
- Modelos estatísticos e de machine learning (Isolation Forest, STL + IQR).
Detectar e tratar outliers adequadamente evita distorções em forecasts e métricas de avaliação.

***

#### 3.7 Correlação Temporal e Sazonalidade

A correlação entre valores passados e presentes é o núcleo da modelagem temporal.

- Funções de autocorrelação revelam dependências:
\$ \rho_k = \frac{Cov(y_t, y_{t-k})}{\sigma^2} \$.
- A periodicidade dos picos de autocorrelação indica a força e o período da sazonalidade.

***

#### 3.8 Ferramentas e Práticas em Python

Bibliotecas e funções úteis para análise:

- **pandas.plot**, **plotly.express** e **seaborn**: visualização preliminar.
- **statsmodels.tsa.seasonal_decompose**: decomposição clássica de séries.
- **pandas.Series.rolling**: cálculo de janelas móveis.
- **scipy.stats.boxcox**: transformações estabilizadoras de variância.
- **pmdarima.utils**: teste de estacionaridade automático e sugestão de ordens ARIMA.

***



### 4. Validação Temporal (versão expandida)

A validação temporal é uma etapa crítica no desenvolvimento de modelos preditivos para séries temporais, pois garante que a avaliação reflita de fato a capacidade do modelo em prever o futuro a partir do passado. Diferente de outros tipos de dados, onde a ordem das amostras pode ser aleatória, em séries temporais **a ordem temporal deve sempre ser preservada**.

***

#### 4.1 Importância da Validação Temporal

A principal meta da validação é medir o desempenho preditivo em condições realistas. Em forecasting, isso significa **simular o fluxo do tempo**: os dados do passado são usados para treinar o modelo, enquanto os dados do futuro são usados para teste.
Se o modelo utilizar informações futuras no treinamento (leakage temporal), ele apresentará um desempenho artificialmente alto e irrealista.

Essa etapa é essencial para:

- Identificar overfitting a variações passadas.
- Avaliar estabilidade do modelo em diferentes períodos.
- Escolher hiperparâmetros de forma confiável.
- Comparar diferentes abordagens sob mesmas condições temporais.

***

#### 4.2 Problemas da Validação Aleatória em Séries Temporais

Em dados independentes e identicamente distribuídos (i.i.d.), o particionamento aleatório dos dados (Holdout ou K-Fold tradicional) funciona bem.
Já em séries temporais, isso **quebra a dependência temporal** e causa **vazamento de informação**.

Exemplo problemático:
Um modelo pode aprender padrões do futuro (validação) e aplicar indevidamente ao passado (treinamento). Esse erro leva a previsões irreais, já que, na prática, o futuro jamais é conhecido no momento da predição.

Portanto, **a divisão temporal deve respeitar a cronologia dos dados**: sempre treinar com observações anteriores e testar com subsequentes.

***

#### 4.3 Técnicas de Validação Temporal

##### a) Holdout Temporal (Treino/Teste Simples)

Consiste em separar os primeiros 70–80% da série para treinamento e os últimos 20–30% para validação.
Simples de implementar, mas avalia o modelo apenas em uma janela temporal, o que pode gerar resultados enviesados se a série for não estacionária.

##### b) TimeSeriesSplit (Rolling Origin / Rolling Forecast)

Método mais robusto, implementado em `scikit-learn`.
Divide os dados em múltiplas partições progressivas, simulando o avanço do tempo com diferentes pontos de corte.

Exemplo com 5 divisões:


| Split | Treino | Validação |
| :-- | :-- | :-- |
| 1 | t₁ – t₃ | t₄ |
| 2 | t₁ – t₄ | t₅ |
| 3 | t₁ – t₅ | t₆ |
| … | … | … |

Dessa forma, o modelo é testado várias vezes em períodos subsequentes, proporcionando uma estimativa mais estável do erro.

##### c) Expanding Window vs. Sliding Window

- **Expanding Window:** o conjunto de treino cresce a cada iteração, incorporando dados novos sem descartar os antigos.
    - Vantagem: aproveita o histórico completo.
    - Desvantagem: custo computacional maior.
- **Sliding Window:** mantém um tamanho fixo de janela (ex.: últimos 12 meses) e move-a ao longo do tempo.
    - Vantagem: adequada para séries com mudanças de regime (não estacionárias).
    - Desvantagem: descarta informações antigas.

***

#### 4.4 Janelas de Validação e Horizonte de Previsão

No contexto de **forecasting multistep**, é necessário definir:

- **Janela de treinamento (input window):** quanto do passado é usado para prever o futuro.
- **Horizonte de previsão (forecast horizon):** o quanto no futuro se deseja prever (ex.: 1, 7 ou 30 passos).

Escolhas adequadas dependem do domínio do problema:

- Previsão de temperatura → horizonte curto (1-3 dias).
- Previsão de demanda industrial → horizonte intermediário (7-30 dias).
- Planejamento de produção → horizonte longo (meses a anos).

***

#### 4.5 Métricas de Avaliação

A validação temporal deve acompanhar métricas adequadas para séries temporais:

- **MAE (Mean Absolute Error):** erro médio absoluto.
- **RMSE (Root Mean Squared Error):** enfatiza grandes erros.
- **MAPE (Mean Absolute Percentage Error):** erro percentual médio (útil para interpretação).
- **sMAPE (Symmetric MAPE):** corrige assimetria presente no MAPE tradicional.
- **MASE (Mean Absolute Scaled Error):** compara a performance ao modelo de referência “naïve” (usando último valor como previsão).

Essas métricas permitem medir tanto a acurácia média quanto a estabilidade do modelo ao longo do tempo.

***

#### 4.6 Boas Práticas de Validação Temporal

1. Nunca embaralhar os dados temporais.
2. Sempre dividir respeitando a cronologia.
3. Avaliar o modelo em múltiplas janelas temporais (rolling validation).
4. Ajustar hiperparâmetros com base em médias de desempenho de diferentes janelas.
5. Em dados de alta frequência (ex.: financeiro), usar validação aninhada para evitar leakage por alinhamento errado.

***

#### 4.7 Ferramentas em Python

- **scikit-learn:** `TimeSeriesSplit`, `cross_val_score`.
- **tscv (time-series cross-validation)**: biblioteca especializada com recursos para múltiplos horizontes.
- **pmdarima.model_selection:** versão otimizada para ARIMA.
- **Prophet:** fornece mecanismo interno de ajuste com train/test automático em períodos definidos.

***


### 5. Modelos Clássicos de Forecasting (versão expandida)

Os modelos clássicos de previsão de séries temporais são baseados em estatística e constituem a base teórica fundamental para o entendimento das abordagens modernas. Eles capturam dependências temporais através de relações lineares entre valores passados e atuais, decompondo os padrões de tendência, sazonalidade e ruído.

***

#### 5.1 Introdução aos Modelos Clássicos

Antes do advento das redes neurais e métodos híbridos, o forecasting era dominado por modelos como ARIMA e suas variações sazonais.
Esses modelos seguem o princípio de **autoregressão linear**, onde o valor atual depende de seus valores passados e dos erros anteriores.
Mesmo hoje, os modelos clássicos continuam amplamente usados por sua interpretabilidade, facilidade de ajuste e solidez em séries de baixa complexidade.

***

#### 5.2 Modelo AR (AutoRegressive)

O modelo **AR(p)** expressa o valor atual da série como uma combinação linear de p valores passados:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t
$$

onde:

- \$ c \$: constante.
- \$ \phi_i \$: coeficientes de autoregressão.
- \$ \epsilon_t \$: ruído branco, com média zero e variância constante.

Para identificar a ordem p, utiliza-se o **gráfico de autocorrelação parcial (PACF)**. Picos significativos no PACF até a defasagem p indicam que o valor atual depende desses atrasos.

***

#### 5.3 Modelo MA (Moving Average)

O modelo **MA(q)** relaciona o valor atual com os erros passados:

$$
y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
$$

onde \$ \theta_i \$ são os pesos atribuídos aos erros anteriores.
Aqui, o gráfico de **autocorrelação (ACF)** é utilizado para selecionar q, já que ele mostra o impacto de choques temporais (ruído) sobre os valores observados futuros.

***

#### 5.4 Modelo ARMA (AutoRegressive Moving Average)

O **ARMA(p, q)** combina as ideias dos modelos AR e MA, sendo apropriado para séries **estacionárias** (sem tendência ou variação sazonal significativa):

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t
$$

Esse modelo é ideal para capturar dependências de curto prazo e padrões de ruído estruturado.

***

#### 5.5 Modelo ARIMA (AutoRegressive Integrated Moving Average)

O **ARIMA(p, d, q)** amplia o ARMA ao incluir a diferenciação para tratar séries **não estacionárias**.
O termo “Integrated” refere-se ao processo de diferenciação aplicado d vezes:

$$
y'_t = (1 - B)^d y_t
$$

onde B é o operador de defasagem (lag operator).
A equação resultante é:

$$
(1 - \sum_{i=1}^{p} \phi_i B^i)(1 - B)^d y_t = (1 + \sum_{j=1}^{q} \theta_j B^j) \epsilon_t
$$

A escolha dos parâmetros (p, d, q) é feita com base em:

- Inspeção dos gráficos **ACF** e **PACF**.
- Critérios de informação: **AIC** (Akaike Information Criterion) e **BIC** (Bayesian Information Criterion).
- Validação cruzada temporal.

O método **auto.arima** da biblioteca `pmdarima` automatiza essa seleção.

***

#### 5.6 Modelo SARIMA (Seasonal ARIMA)

Para séries com **sazonalidade**, utiliza-se o **SARIMA(p, d, q)(P, D, Q, s)**, onde s é o período sazonal.
Inclui componentes adicionais que capturam dependências sazonais:

$$
\Phi_P(B^s)(1 - B^s)^D y_t = \Theta_Q(B^s) \epsilon_t
$$

Esse modelo é eficaz em dados com padrões previsíveis de repetição (ex.: vendas mensais, temperatura anual).

***

#### 5.7 Modelos de Suavização Exponencial

Outra classe importante de modelos clássicos são os métodos de **suavização exponencial**, que atribuem pesos decrescentes às observações passadas:

- **Suavização Exponencial Simples:** adequada para séries sem tendência nem sazonalidade.
- **Método de Holt:** estende o modelo para incluir tendência linear.
- **Método Holt-Winters:** adiciona o componente sazonal (pode ser aditivo ou multiplicativo).

O modelo Holt-Winters, por exemplo, é amplamente usado para previsão de curto prazo em negócios por sua adaptabilidade e baixo custo computacional.

***

#### 5.8 Diagnóstico dos Modelos

Uma etapa essencial após o ajuste é verificar se os resíduos são ruído branco (sem autocorrelação).

- Gráficos de resíduos e testes de Ljung-Box avaliam essa condição.
- Se resíduos apresentarem autocorrelação, é necessário ajustar novamente os parâmetros ou introduzir termo sazonal ou de regressão.

***

#### 5.9 Implementação Prática em Python

Bibliotecas comuns:

- **statsmodels:** `ARIMA`, `SARIMAX`, `ExponentialSmoothing`.
- **pmdarima:** `auto_arima` para seleção automática de parâmetros.
- **forecasting frameworks (tsfresh, darts, sktime):** oferecem integração de ARIMA e métodos híbridos com machine learning.

Fluxo típico:

1. Verificar estacionaridade com ADF.
2. Aplicar diferenciação se necessário.
3. Escolher p e q via ACF/PACF.
4. Ajustar o modelo (fit).
5. Avaliar resíduos e métricas (MAE, RMSE).
6. Gerar previsões out-of-sample.

***

#### 5.10 Considerações Finais

Os modelos ARIMA e suas variantes continuam sendo pilares fundamentais do forecasting:

- Fornecem **interpretação estatística transparente** sobre tendência e autocorrelação.
- São base de comparação para modelos complexos (benchmarks).
- Em séries simples e estacionárias, frequentemente superam modelos de deep learning.

Apesar de suas limitações em séries altamente não lineares, a compreensão profunda desses modelos é indispensável para qualquer especialista em aprendizado de máquina aplicado a séries temporais.



### 6. Modelos Baseados em Decomposição e Regressão (versão expandida)

Os modelos baseados em **decomposição** e **regressão** são abordagens clássicas de previsão que buscam separar os comportamentos estruturais da série temporal (tendência, sazonalidade e ruído) e modelá-los explicitamente de forma analítica ou estatística. Eles são especialmente úteis quando há clareza sobre os fatores que influenciam os dados e quando se deseja interpretar os componentes do processo de geração da série.

***

#### 6.1 Conceito de Decomposição

A decomposição de séries temporais consiste em dividir a série observada \$ y_t \$ em partes interpretáveis:

$$
y_t = T_t + S_t + R_t
$$

ou

$$
y_t = T_t \times S_t \times R_t
$$

onde:

- \$ T_t \$: tendência (movimento de longo prazo).
- \$ S_t \$: sazonalidade (padrões periódicos de curto ou médio prazo).
- \$ R_t \$: componente aleatório ou ruído.

A escolha entre decomposição aditiva ou multiplicativa depende da natureza dos dados:

- **Aditiva:** adequada quando as flutuações sazonais e variações ao longo do tempo têm magnitude constante.
- **Multiplicativa:** ideal quando a amplitude da variação cresce com o nível da série (ex.: aumento proporcional em vendas).

***

#### 6.2 Métodos Clássicos de Decomposição

Três abordagens principais são amplamente utilizadas:

1. **Decomposição Clássica (Moving Average Decomposition):**
    - A tendência é estimada por médias móveis centradas.
    - A sazonalidade é identificada subtraindo a tendência da série original e observando padrões recorrentes.
    - O ruído é obtido como o resíduo entre a observação e a soma dos componentes determinísticos.
    - Aplicada em séries curtas e facilmente interpretáveis.
2. **Decomposição STL (Seasonal-Trend decomposition using Loess):**
    - Método robusto e flexível baseado em suavização local (Loess).
    - Suporta séries não estacionárias e múltiplas sazonalidades.
    - Ideal em contextos com comportamento sazonal mutável (ex.: tráfego web, demanda energética).
    - Implementação prática: `statsmodels.tsa.seasonal.STL`.
3. **X-11 e X-13ARIMA-SEATS:**
    - Desenvolvidos para séries econômicas oficiais.
    - Combinam decomposição com técnicas ARIMA para modelagem dos componentes.
    - Muito usados em macroeconomia (PIB, emprego, inflação).

***

#### 6.3 Modelos de Regressão Temporal

A abordagem baseada em **modelos de regressão** busca explicar a variável temporal com base em outras variáveis relevantes, internas (lags da própria série) ou externas (variáveis exógenas).

Modelo geral:

$$
y_t = \beta_0 + \beta_1 x_{1,t} + \beta_2 x_{2,t} + \dots + \beta_p x_{p,t} + \epsilon_t
$$

onde:

- \$ x_{i,t} \$ podem incluir lags (\$ y_{t-1}, y_{t-2}, ··· \$) e variáveis exógenas.
- \$ \epsilon_t \$ representa o erro aleatório.

Esse tipo de modelo permite incorporar fatores externos (como preços de insumos, temperatura ou indicadores econômicos) e avaliar sua influência na série.

***

#### 6.4 Modelos de Regressão com Decomposição

Combinar decomposição e regressão cria abordagens híbridas e interpretáveis:

- Extraem-se os componentes determinísticos (tendência e sazonalidade).
- Modela-se o componente residual via regressão ou aprendizado de máquina.

Exemplo prático:

1. Decompor série com STL.
2. Usar regressão linear para prever tendência.
3. Usar regressão polinomial ou spline para capturar ciclos não lineares.
4. Prever o resíduo com regressão ou ARIMA.

Esse processo melhora a robustez preditiva e a interpretabilidade.

***

#### 6.5 Modelos de Suavização Exponencial

Os métodos de **suavização exponencial** são uma ponte entre as técnicas de decomposição e os modelos dinâmicos. Eles assumem que:

- As observações mais recentes têm maior relevância.
- O peso das observações passadas decresce exponencialmente ao longo do tempo.

Tipos principais:

1. **Suavização Simples:**

$$
\hat{y}_{t+1} = \alpha y_t + (1 - \alpha)\hat{y}_t
$$

usada para séries estacionárias.
2. **Método de Holt:**
adiciona componente de tendência linear.
3. **Método de Holt-Winters:**
incorpora tendência e sazonalidade (aditiva ou multiplicativa).
Altamente empregado em aplicações empresariais e logísticas.

***

#### 6.6 Regressão Aditiva Geral (GAM) e Modelos com Funções de Base

Em casos onde os efeitos temporais são não lineares, usam-se modelos de regressão com funções flexíveis:

- **GAMs (Generalized Additive Models):**

$$
y_t = \beta_0 + f_1(t) + f_2(\text{sazonalidade}) + \epsilon_t
$$

onde \$ f_i(\cdot) \$ são funções suaves (splines), ajustadas automaticamente.
- Oferecem excelente compromisso entre flexibilidade e interpretabilidade.
- Implementações: `pyGAM`, `statsmodels.gam`, `mgcv` (em R).

***

#### 6.7 Análise de Resíduos e Diagnóstico

Após a modelagem, é necessário avaliar:

- Se os resíduos são ruído branco (sem autocorrelação).
- Se não restam padrões estruturais nas previsões.
- O desempenho do modelo por meio de métricas (MAE, RMSE, MAPE, MASE).

Um modelo bem ajustado deve capturar tendência e sazonalidade, deixando o resíduo puramente aleatório.

***

#### 6.8 Aplicações Típicas

- **Economia:** decomposição de indicadores macroeconômicos em ciclo e tendência.
- **Vendas e Marketing:** previsão de demanda e impacto de campanhas.
- **Energia e Clima:** modelagem sazonal de consumo e temperatura.
- **Engenharia e IoT:** previsão de séries industriais e medições de sensores.

***

#### 6.9 Vantagens e Limitações

**Vantagens:**

- Altamente interpretáveis.
- Simples de implementar.
- Baixa exigência computacional.

**Limitações:**

- Desempenho reduzido em séries não lineares ou com alta variabilidade.
- Dificuldade em capturar efeitos complexos de interação entre fatores.

***

Em conjunto, os modelos baseados em decomposição e regressão formam o pilar da análise clássica de séries temporais. Eles não apenas servem como ferramentas descritivas eficazes, mas também como base sólida para a construção de modelos híbridos que combinam estatística e aprendizado de máquina.



### 7. Modelos Modernos de Previsão — Prophet (versão expandida)

O **Prophet** é um modelo moderno de previsão de séries temporais desenvolvido pelo Facebook (atual Meta). Sua proposta é oferecer um método **robusto, interpretável e automatizado** para lidar com séries complexas, especialmente aquelas com **forte sazonalidade, feriados e mudanças de tendência**. Ele combina princípios estatísticos clássicos com técnicas flexíveis de ajuste e é amplamente utilizado em aplicações empresariais de forecast.

***

#### 7.1 Conceito e Motivação

O Prophet foi criado para preencher a lacuna entre dois mundos:

- Modelos estatísticos clássicos, como ARIMA e Holt-Winters, que exigem parametrização e análise cuidadosa.
- Modelos de machine learning, como regressões complexas e redes neurais, que demandam muito tempo de ajuste e pouca interpretabilidade.

Prophet se destaca por:

- Oferecer **resultados comparáveis a ARIMA e LSTM**, mas com parametrização mínima.
- Permitir fácil interpretação dos componentes (tendência, sazonalidade, feriados).
- Ser escalável para aplicação automatizada em múltiplas séries dentro de pipelines de dados.

***

#### 7.2 Estrutura do Modelo

O Prophet é um modelo **aditivo**, no qual a série temporal \$ y(t) \$ é representada como:

$$
y(t) = g(t) + s(t) + h(t) + \epsilon_t
$$

onde:

- \$ g(t) \$: componente de **tendência**, que captura o crescimento ou decaimento.
- \$ s(t) \$: componente de **sazonalidade** (diária, semanal, anual).
- \$ h(t) \$: componente de **feriados e eventos especiais**.
- \$ \epsilon_t \$: ruído aleatório (não explicado pelos demais termos).

Essa estrutura facilita a interpretação, pois cada componente pode ser visualizado e analisado separadamente.

***

#### 7.3 Modelagem da Tendência (\$ g(t) \$)

Prophet usa duas formas principais de modelagem de tendência:

1. **Crescimento Linear:**

$$
g(t) = (k + a(t)\delta)t + (m + a(t)\gamma)
$$

onde \$ k \$ é a taxa de crescimento e \$ m \$ é o intercepto.
Esse modelo é indicado quando há crescimento constante ao longo do tempo.
2. **Crescimento Logístico (com capacidade):**

$$
g(t) = \frac{C}{1 + \exp(-k(t - m))}
$$

onde \$ C \$ é a capacidade máxima (limite superior).
Ideal para representar crescimento que tende à saturação (ex.: adoção de um produto).

O modelo detecta automaticamente **changepoints**, pontos de inflexão na tendência, permitindo capturar mudanças abruptas, como crises econômicas ou picos de mercado.

***

#### 7.4 Modelagem da Sazonalidade (\$ s(t) \$)

A sazonalidade é descrita como uma **expansão de série de Fourier**:

$$
s(t) = \sum_{n=1}^{N} [a_n \cos(2\pi n t / P) + b_n \sin(2\pi n t / P)]
$$

- \$ P \$: período da sazonalidade (por exemplo, 365 dias para sazonalidade anual).
- A complexidade \$ N \$ controla a suavidade da sazonalidade.

O Prophet inclui sazonalidades **anuais, semanais e diárias** por padrão, mas também permite adicionar sazonalidades personalizadas, como efeitos mensais ou horários.

***

#### 7.5 Feriados e Eventos Especiais (\$ h(t) \$)

Um dos diferenciais do Prophet é o tratamento de **feriados e eventos temporais**.

- Permite incluir feriados específicos por país (ex.: Natal, Carnaval, Black Friday).
- Também admite eventos personalizados, como lançamentos de produtos ou campanhas.
- Os efeitos são modelados como variáveis indicadoras binárias, ajustadas por regressão linear dentro do modelo principal.

Essa abordagem é crucial em aplicações de negócios, nas quais feriados e ciclos promocionais têm impacto previsível.

***

#### 7.6 Ajuste do Modelo e Hiperparâmetros

Os principais hiperparâmetros de controle são:

- **changepoint_prior_scale:** regula a flexibilidade da tendência (menor valor → curva mais suave).
- **seasonality_prior_scale:** controla a força da sazonalidade.
- **holidays_prior_scale:** define o peso dado aos feriados.
- **interval_width:** largura do intervalo de incerteza das previsões (padrão: 0.8).

A calibração desses parâmetros é feita empiricamente ou via validação temporal com `cross_validation` da API Prophet.

***

#### 7.7 Avaliação e Diagnóstico

Após o ajuste, Prophet fornece diagnósticos e visualizações detalhadas:

- **Plot de componentes:** visualização separada de tendência, sazonalidade e efeitos de feriado.
- **Validação cruzada temporal:** implementada internamente (`performance_metrics`) para cálculos de RMSE, MAE e MAPE.
- **Resíduos:** análise do erro ao longo do tempo para detectar subajustes ou overfitting.
- **Previsões com intervalos de confiança:** através da amostragem de incerteza Bayesiana.

***

#### 7.8 Implementação em Python

Um exemplo prático de uso:

```python
from prophet import Prophet
import pandas as pd

# Preparação do dataset
df = pd.read_csv('dados.csv')
df.rename(columns={'data': 'ds', 'valor': 'y'}, inplace=True)

# Criação e ajuste do modelo
modelo = Prophet(
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
modelo.add_country_holidays(country_name='BR')
modelo.fit(df)

# Previsão futura
futuro = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(futuro)
```

Esse pipeline gera previsões interpretáveis e visualizações acessíveis com `modelo.plot(forecast)`.

***

#### 7.9 Aplicações Reais e Casos de Uso

Prophet é amplamente usado em:

- **Comércio eletrônico:** previsão de vendas e estoque.
- **Finanças:** previsão de receita e volume de transações.
- **Energia:** modelagem de consumo diário com padrões de sazonalidade.
- **Transporte e logística:** previsão de demanda ou tráfego em rotas.

Seu desempenho é comparável ao ARIMA e lesma, superando-os em cenários com sazonalidades múltiplas e feriados específicos.

***

#### 7.10 Vantagens e Limitações

**Vantagens:**

- Rápido ajuste e alta interpretabilidade.
- Capacidade automática de detecção de pontos de mudança.
- Inclusão fácil de feriados e sazonalidades adicionais.
- Funciona bem com dados faltantes e outliers leves.

**Limitações:**

- Supõe aditividade e linearidade local.
- Menor desempenho em dados altamente não lineares ou com interações complexas.
- Não capta autocorrelações de alta ordem tão bem quanto modelos baseados em dependência temporal (como LSTM).

***

Em resumo, o Prophet é um modelo híbrido moderno que combina **simplicidade estatística, grande interpretabilidade e automatização de tuning**, tornando-se uma ferramenta prática e poderosa para analistas e cientistas de dados que precisam de forecasts confiáveis e facilmente explicáveis.



### 8. Avaliação e Métricas (versão expandida)

A etapa de **avaliação de modelos de séries temporais** é responsável por determinar o quão bem uma técnica de previsão se ajusta aos dados e, principalmente, o quanto ela é capaz de **generalizar para o futuro**. Essa fase é essencial para validar se o modelo captura adequadamente as tendências, sazonalidades e padrões do processo gerador, sem superajustar ruídos ou flutuações aleatórias.

***

#### 8.1 Importância da Avaliação em Séries Temporais

Diferentemente de outras tarefas de aprendizado supervisionado, a avaliação em séries temporais deve **respeitar a ordem temporal** dos dados. Isso impede que informações do futuro influenciem o treinamento.
As principais metas da avaliação são:

- **Medir a precisão da previsão.**
- **Comparar diferentes modelos.**
- **Selecionar hiperparâmetros e janelas ótimas de treino.**
- **Verificar estabilidade e robustez ao longo do tempo.**

Um bom modelo de previsão deve apresentar **baixa taxa de erro e consistência temporal** — ou seja, sua performance não deve variar drasticamente em diferentes períodos de teste.

***

#### 8.2 Estratégias de Validação Temporal

1. **Treino/Teste com Corte Temporal Único:**
    - O conjunto de treinamento contém os dados até um instante \$ t \$, e o conjunto de teste contém o período subsequente.
    - Simples, porém limitado para avaliar estabilidade em diferentes fases.
2. **Rolling Window (TimeSeriesSplit):**
    - Utiliza janelas deslizantes de treinamento e teste, mantendo a sequência temporal:


| Divisão | Treino | Validação |
| :-- | :-- | :-- |
| 1 | t₁...t₅ | t₆ |
| 2 | t₂...t₆ | t₇ |
| 3 | t₃...t₇ | t₈ |

    - Permite calcular métricas médias e desvios, oferecendo uma visão mais robusta do desempenho.
3. **Expanding Window:**
    - A janela de treino cresce progressivamente a cada iteração.
    - Ideal para séries com comportamento evolutivo estável.
4. **Walk-Forward Validation:**
    - Variante do método anterior, usada em aplicações online.
    - O modelo é atualizado continuamente conforme novos dados são disponibilizados.

Essas abordagens são fundamentais para previsões contínuas, comuns em contextos de negócios, energia e finanças.

***

#### 8.3 Métricas de Erro Mais Utilizadas

Avaliar o erro de previsões requer métricas específicas, pois os dados podem ter diferentes escalas, unidades e níveis de variância.

##### a) Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|
$$

- Mede o erro médio absoluto entre previsão e valor real.
- Fácil de interpretar, mas não penaliza tanto grandes erros.


##### b) Mean Squared Error (MSE)

$$
MSE = \frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2
$$

- Penaliza erros grandes de forma quadrática.
- Útil quando erros grandes são mais críticos.

A sua raiz quadrada, **RMSE (Root Mean Squared Error)**, traz a métrica para a mesma unidade da variável original.

##### c) Mean Absolute Percentage Error (MAPE)

$$
MAPE = \frac{100}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|
$$

- Expressa o erro como porcentagem, facilitando comparação entre séries de diferentes magnitudes.
- Sensível a valores próximos de zero.


##### d) Symmetric MAPE (sMAPE)

$$
sMAPE = \frac{100}{n} \sum_{t=1}^{n} \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|)/2}
$$

- Corrige o problema de assimetria do MAPE clássico.
- Mais estável quando há valores baixos.


##### e) Mean Absolute Scaled Error (MASE)

$$
MASE = \frac{MAE}{MAE_{\text{naïve}}}
$$

- Normaliza o erro pela média absoluta de um modelo “naïve” (que prevê o último valor).
- Um valor menor que 1 indica desempenho melhor que o modelo trivial.

***

#### 8.4 Outras Métricas e Critérios de Informação

Além das métricas diretas de erro, podem ser empregados critérios estatísticos que consideram a **complexidade do modelo**:

- **AIC (Akaike Information Criterion):**

$$
AIC = 2k - 2\ln(L)
$$

onde \$ k \$ é o número de parâmetros e \$ L \$ é a verossimilhança.
- **BIC (Bayesian Information Criterion):**
Penaliza mais severamente modelos complexos que o AIC.
Ambos são usados na seleção de ordem de modelos ARIMA e SARIMA.

Esses critérios equilibram ajuste e simplicidade, evitando overfitting.

***

#### 8.5 Visualizações para Avaliação

A análise visual é parte essencial da avaliação preditiva:

- **Gráfico de previsão vs. valores reais:** revela atrasos e amplitude de erro.
- **Resíduos ao longo do tempo:** devem se comportar como ruído branco.
- **ACF/PACF dos resíduos:** ausência de autocorrelação confirma bom ajuste.
- **Gráficos de erro percentual ou rolling error:** mostram onde o modelo perde precisão.

Ferramentas como `matplotlib`, `seaborn` e `plotly` facilitam essas inspeções com dashboards dinâmicos.

***

#### 8.6 Avaliação em Diferentes Horizontes Temporais

Modelos podem apresentar desempenhos distintos dependendo do horizonte de previsão:

- **Curto prazo (1–5 passos):** avaliados por precisão imediata.
- **Médio prazo (6–30 passos):** avaliam estabilidade preditiva.
- **Longo prazo (>30 passos):** foco em tendência geral.

Uma prática recomendada é calcular as métricas em múltiplos horizontes para entender a degradação do desempenho com o tempo.

***

#### 8.7 Métricas Específicas de Negócio

Em situações aplicadas, erros absolutos nem sempre refletem impacto real.
Portanto, empresas podem adotar métricas customizadas, como:

- **Erro de previsão de demanda média por loja** (varejo).
- **Diferença percentual acumulada de energia gerada** (energia).
- **Taxa de acerto na previsão de picos de tráfego** (redes).

Essas métricas conectam o desempenho técnico às metas operacionais.

***

#### 8.8 Avaliação Comparativa

Para comparar modelos (ARIMA, Prophet, LSTM etc.), recomenda-se:

1. Aplicar **a mesma janela de validação temporal**.
2. Usar **as mesmas métricas padronizadas**.
3. Avaliar **acurácia e estabilidade conjunta** — um modelo menos preciso, porém mais estável, pode ser mais útil.

Tabelas de comparação e boxplots de desempenho são ferramentas ideais para análise comparativa robusta.

***

#### 8.9 Interpretação e Comunicação dos Resultados

A etapa final da avaliação inclui comunicar os resultados de forma acessível:

- Indicar **precisão média e variação (erro-padrão)**.
- Visualizar **intervalos de confiança preditivos**.
- Destacar **instabilidades e períodos com maior erro**.
- Traduzir as métricas em implicações práticas (ex.: erro médio de 5% equivale a 100 unidades vendidas a menos por dia).

***

A análise cuidadosa de métricas e validações temporais é o que transforma previsões em insights úteis e confiáveis. Sem uma avaliação adequada, até modelos sofisticados — como LSTM, SARIMA ou Prophet — correm o risco de parecer precisos, mas falharem diante de novos dados.


### 9. Aplicações Práticas (versão expandida)

O estudo de séries temporais e forecasting transcende o domínio teórico, apresentando inúmeras **aplicações práticas** em setores industriais, financeiros, científicos e tecnológicos. Essa aplicabilidade é resultado direto da crescente coleta de dados temporais em larga escala — de sensores IoT a plataformas de e-commerce —, permitindo a tomada de decisões baseada em previsões precisas e contextualizadas.

***

#### 9.1 Importância das Aplicações Práticas

Modelos de previsão de séries temporais têm papel estratégico por:

- Antecipar demandas e otimizar recursos.
- Reduzir custos operacionais por meio de planejamentos mais assertivos.
- Permitir respostas rápidas a eventos inesperados.
- Auxiliar na análise de riscos e planejamento de longo prazo.

Além disso, as previsões podem ser integradas a sistemas automatizados (como redes inteligentes ou estoques autogeridos) e aprimoradas continuamente por aprendizado incremental.

***

#### 9.2 Principais Domínios de Aplicação

##### a) Finanças e Economia

- **Previsão de preços de ativos:** séries financeiras (ações, câmbio, commodities) são modeladas com técnicas de ARIMA, GARCH e LSTM.
    - Exemplo: modelar a volatilidade de uma ação e calcular Value at Risk (VaR).
- **Indicadores econômicos:** PIB, inflação e desemprego são frequentemente analisados com modelos sazonais (SARIMA, Prophet).
- **Séries multivariadas:** previsão simultânea de variáveis correlacionadas (ex.: taxas de juros e inflação).
- Ferramentas complementares: modelos VAR (Vector AutoRegressive) e cointegration tests.


##### b) Varejo e Marketing

- **Previsão de vendas:** ajuste de modelos para prever volume de vendas por produto, loja ou região.
- **Planejamento de estoque:** evitar excesso ou falta de itens em períodos sazonais (ex.: Black Friday, Natal).
- **Campanhas promocionais:** mensuração do impacto temporal de campanhas sobre as vendas.
- **Prophet + Machine Learning:** predições híbridas com variáveis exógenas como feriados, clima ou redes sociais.


##### c) Energia e Sustentabilidade

- **Demanda energética:** previsão horária ou diária de consumo elétrico, essencial para planejamento da rede.
- **Geração renovável:** estimativas de vento e radiação solar com base em históricos climáticos.
- **Otimização de consumo:** uso de forecasts para ajustar preços dinâmicos de energia (tarifação inteligente).
- **Modelos comuns:** ARIMA sazonal, Prophet, LSTM e CNN-1D aplicados a séries contínuas.


##### d) Saúde e Epidemiologia

- **Previsão de casos e surtos:** modelagem temporal de contaminações ou internações hospitalares.
- **Monitoramento de sinais vitais:** séries de EEG, ECG ou glicose possuem dependência temporal complexa.
- **Análise de séries multivariadas:** identificação de padrões em sensores fisiológicos combinados.
- Exemplo: uso de redes recorrentes (LSTM/GRU) para prever episódios cardíacos em tempo real.


##### e) Transporte e Logística

- **Previsão de tráfego:** estimar congestionamentos e tempos médios de deslocamento.
- **Gestão de frotas e rotas:** ajustar frequência de entregas com base em demanda prevista.
- **Previsão de atrasos e volumes:** comum em aeroportos, portos e transporte público.
- **Modelos adequados:** RNNs, Prophet com sazonalidade diária e ARIMA com covariáveis.


##### f) Manufatura e IoT Industrial

- **Manutenção preditiva:** previsão de falhas em máquinas com base em séries de sensores.
- **Controle de qualidade:** detecção de anomalias e desvios em processos produtivos.
- **Previsão de consumo de insumos e energia:** otimização de produção conforme padrões sazonais.
- **Técnicas avançadas:** ARIMA-LSTM híbrido e modelagem multivariada com variáveis de processo.


##### g) Meteorologia e Climatologia

- **Previsão de temperatura, precipitação e vento:** aplicação clássica de modelos temporais.
- **Análise de anomalias climáticas:** séries longas permitem detecção de mudanças sistêmicas (mudança climática).
- **Modelos:** regressão linear com features temporais, LSTM, ou modelos baseados em atenção (Transformers).

***

#### 9.3 Séries Temporais em Ciência e Pesquisa

Nos contextos científicos, séries temporais são empregadas para detectar padrões e inferir causalidade temporal em experimentos.
Exemplos incluem:

- Observação de sinais astronômicos ou sísmicos.
- Análise de atividade neural ao longo do tempo.
- Estudos de ecossistemas e evolução de populações.
Esses casos requerem alta resolução temporal e filtragem avançada de ruído.

***

#### 9.4 Aplicações em Inteligência Artificial e Machine Learning

As séries temporais constituem uma das formas mais ricas de dados para IA aplicada.
Alguns exemplos incluem:

- **Previsão de séries financeiras com redes neurais profundas (DNN, LSTM, Transformer).**
- **Reconhecimento de padrões em séries auditivas (speech-to-text) e biomédicas.**
- **Previsão de tráfego em redes (network forecasting).**
- **Uso de embeddings sequenciais em modelos transformers como BERT-Time.**

Essas abordagens permitem capturar dependências de longo alcance e não linearidades complexas, superando modelos puramente estatísticos em muitas situações.

***

#### 9.5 Integração com Ferramentas e Ambientes Reais

A aplicabilidade prática depende também da integração com sistemas de produção:

- **APIs e dashboards interativos:** exibição das previsões em tempo real (Plotly Dash, Power BI, Streamlit).
- **Infraestrutura em nuvem:** pipelines automatizados com AWS Forecast, Azure ML, GCP Vertex AI.
- **Monitoramento contínuo:** verificação de desvios e reparametrização automática de modelos via *backtesting*.
- **Aprendizado contínuo (online learning):** reentrenamento automático à medida que novos dados chegam.

***

#### 9.6 Boas Práticas e Considerações Éticas

Em aplicações de previsão, a confiabilidade e o contexto são fundamentais:

1. **Transparência:** divulgar incertezas e limitações do modelo.
2. **Atualização contínua:** prever com dados desatualizados leva a erros exponenciais.
3. **Impacto socioeconômico:** previsões em saúde e logística afetam vidas; devem ser auditáveis.
4. **Diversidade de fontes:** combinar variáveis internas e externas melhora a robustez.

***

#### 9.7 Conclusão

As aplicações práticas de séries temporais demonstram o poder da análise preditiva na transformação de dados em ação. Do mercado financeiro a sistemas inteligentes de energia, os princípios de **forecasting** fornecem um elo decisivo entre observação e decisão.
Dominar essas técnicas é essencial tanto para cientistas de dados quanto para engenheiros e pesquisadores em qualquer domínio.


### 10. Exercício Prático (versão expandida)

Este tópico tem como objetivo consolidar o aprendizado teórico sobre **séries temporais e forecasting** por meio de um exercício prático completo, permitindo que o estudante passe por todas as etapas do processo — desde a exploração inicial dos dados até a avaliação final das previsões. O propósito não é apenas obter resultados numéricos, mas **compreender o fluxo de raciocínio envolvido na modelagem temporal** e desenvolver senso crítico sobre as escolhas metodológicas.

***

#### 10.1 Objetivos do Exercício

1. Compreender o fluxo completo de análise de séries temporais.
2. Aplicar conceitos de tendência, sazonalidade e decomposição.
3. Implementar modelos clássicos e modernos (ARIMA e Prophet).
4. Comparar o desempenho de diferentes abordagens.
5. Interpretar resultados e elaborar conclusões baseadas em evidências.

***

#### 10.2 Dataset Sugerido

Escolha um dataset de fácil acesso, que contenha observações temporais contínuas. Exemplos adequados:

- **Vendas diárias de um e-commerce.**
- **Consumo de energia elétrica por hora.**
- **Temperatura média diária em uma cidade.**
- **Número de acessos diários a um website.**

Caso o dataset não contenha feriados ou eventos, é possível enriquecer os dados com variáveis exógenas (por exemplo, dummies para fins de semana ou feriados nacionais).

***

#### 10.3 Etapa 1 — Exploração e Visualização

1. **Carregar os dados** e garantir que a coluna temporal esteja em formato datetime.
2. **Visualizar a série bruta**:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('series.csv', parse_dates=['data'])
plt.plot(df['data'], df['valor'])
plt.title('Série Temporal Original')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.show()
```

3. **Identificar padrões visuais:** tendência, sazonalidade e outliers.
4. **Aplicar decomposição:**

```python
from statsmodels.tsa.seasonal import seasonal_decompose
resultado = seasonal_decompose(df['valor'], model='additive', period=30)
resultado.plot()
plt.show()
```

Essa visualização evidenciará como cada componente contribui para a variação total da série.

***

#### 10.4 Etapa 2 — Preparação e Estacionaridade

Verifique se a série é estacionária:

- **Teste de Dickey-Fuller Aumentado (ADF):**

```python
from statsmodels.tsa.stattools import adfuller
resultado = adfuller(df['valor'])
print('p-valor:', resultado[^1])
```

Se o p-valor > 0.05, a série não é estacionária.
- Caso necessário, aplique **diferenciação:**

```python
df['diferenciada'] = df['valor'].diff().dropna()
```


***

#### 10.5 Etapa 3 — Modelagem com ARIMA

1. **Determinar parâmetros iniciais (p, d, q)** com auxílio de ACF e PACF.
2. Ajustar o modelo:

```python
from pmdarima import auto_arima
modelo_arima = auto_arima(df['valor'], seasonal=False, trace=True)
modelo_arima.summary()
```

3. **Gerar previsões:**

```python
previsoes = modelo_arima.predict(n_periods=30)
plt.plot(previsoes, label='Previsão')
plt.legend()
plt.show()
```

4. Avaliar os resíduos — eles devem se comportar como ruído branco.

***

#### 10.6 Etapa 4 — Modelagem com Prophet

1. **Preparar o dataset:**

```python
from prophet import Prophet

df_prophet = df.rename(columns={'data': 'ds', 'valor': 'y'})
modelo = Prophet(yearly_seasonality=True, weekly_seasonality=True)
modelo.fit(df_prophet)
```

2. **Gerar previsões futuras:**

```python
futuro = modelo.make_future_dataframe(periods=30)
forecast = modelo.predict(futuro)
modelo.plot(forecast)
modelo.plot_components(forecast)
```

3. **Interpretar resultados:** observar tendência, sazonalidade e intervalos de incerteza.

***

#### 10.7 Etapa 5 — Avaliação dos Modelos

Calcule métricas de desempenho e compare os resultados:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")
```

Analise qual modelo apresenta menor erro médio e maior estabilidade temporal.

***

#### 10.8 Etapa 6 — Discussão e Interpretação

Responda às seguintes questões:

- Os padrões observados refletem comportamento esperado da variável?
- O modelo conseguiu capturar corretamente a sazonalidade?
- Quais períodos apresentaram maior erro e por quê?
- Há indicadores de sobreajuste (overfitting)?
- Como o modelo poderia ser aprimorado (inclusão de variáveis, ajuste de hiperparâmetros, métodos híbridos)?

***

#### 10.9 Etapa 7 — Extensão Opcional

Para alunos avançados, recomenda-se aplicar:

- **Modelos híbridos ARIMA-LSTM:** combinar previsão linear (ARIMA) e não linear (LSTM).
- **Transformações de frequência:** converter dados diários em semanais ou mensais.
- **Previsão de séries multivariadas:** incorporar variáveis externas, como clima ou feriados.
- **Validação cruzada temporal (walk-forward):** testar robustez do modelo em múltiplos blocos temporais.

***

#### 10.10 Entregáveis Esperados

- Código-fonte limpo e documentado.
- Gráficos de decomposição, previsão e resíduos.
- Tabela comparativa com métricas de avaliação.
- Breve relatório explicando:

1. Análise dos componentes temporais.
2. Comparação entre modelos ARIMA e Prophet.
3. Interpretação dos resultados e possíveis melhorias.

***


### 11. Extensões e Conceitos Avançados (versão expandida)

O estudo de séries temporais avança rapidamente, impulsionado pela integração entre **modelos clássicos**, **aprendizado de máquina** e **deep learning**. Este tópico aprofunda técnicas modernas que superam as limitações dos métodos tradicionais e oferecem maior capacidade de modelar dependências complexas, múltiplas variáveis e comportamentos não lineares.

***

#### 11.1 Modelos Híbridos (ARIMA + Redes Neurais)

Os **modelos híbridos** combinam a capacidade explicativa dos modelos estatísticos com o poder de modelagem não linear das redes neurais.
O raciocínio por trás desses modelos é decompor a série em duas partes:

1. **Componente linear:** modelada por ARIMA ou SARIMA.
2. **Componente não linear (residual):** modelada por redes neurais (geralmente LSTM ou MLP).

**Etapas típicas:**

1. Ajustar um modelo ARIMA e obter resíduos.
2. Treinar uma rede neural para prever os resíduos.
3. Somar ambas as previsões para o resultado final.

Esse método captura tanto estruturas determinísticas quanto dinâmicas não lineares, sendo eficaz em séries industriais e financeiras.

***

#### 11.2 Modelos de Deep Learning para Séries Temporais

O aprendizado profundo revolucionou o campo de previsão temporal, oferecendo modelos capazes de compreender dependências de longo prazo e múltiplas variáveis correlacionadas.

##### a) Redes Recorrentes (RNN)

As **RNNs (Recurrent Neural Networks)** processam sequências temporalmente ordenadas, mantendo um estado interno que representa a memória da série.
Contudo, RNNs tradicionais sofrem com o problema do **gradiente explosivo ou dissipativo** em séries longas.

##### b) Long Short-Term Memory (LSTM)

As **LSTM (Long Short-Term Memory)** resolvem esse problema através de portas de controle — *input*, *output* e *forget* — permitindo aprender padrões de longo alcance.
São amplamente usadas em previsão de tráfego, finanças e séries meteorológicas.
Equação simplificada para a célula de memória:

$$
c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t
$$

$$
h_t = o_t \cdot \tanh(c_t)
$$

##### c) Gated Recurrent Unit (GRU)

Uma simplificação da LSTM que utiliza menos parâmetros, mantendo desempenho similar. Ideal para séries curtas ou bases pequenas.

##### d) Convolutional Neural Networks (CNN-1D)

Embora comumente associadas à visão computacional, as **CNNs unidimensionais** aplicam convoluções temporais que capturam padrões locais de curto prazo.
Muitas arquiteturas modernas combinam CNNs com LSTMs para obter o melhor dos dois mundos.

##### e) Transformers Temporais

Arquiteturas baseadas em **atenção** (originalmente criadas para NLP) estão sendo aplicadas com sucesso em séries temporais.
Modelos como **Temporal Fusion Transformer (TFT)** e **Informer** aprendem relações temporais complexas sem depender de estados recorrentes, permitindo paralelização e escalabilidade.
Essas redes utilizam mecanismos de atenção que atribuem pesos diferentes às observações passadas, conforme sua relevância para a previsão atual.

***

#### 11.3 Séries Temporais Multivariadas

Em muitos casos, a variável de interesse depende de múltiplos fatores exógenos.
Modelos multivariados permitem prever uma série \$ y_t \$ com base em variáveis adicionais \$ x_{1,t}, x_{2,t}, ..., x_{n,t} \$.
Exemplo: prever o consumo energético com base na temperatura, hora do dia e feriados.

**Modelos adequados:**

- **VAR (Vector AutoRegressive):** extensão do AR para múltiplas séries correlacionadas.
- **LSTM Multivariado:** inclui múltiplas entradas em paralelo.
- **Transformers multivariados:** aprendem dependências cruzadas automatizadas entre variáveis.

***

#### 11.4 Aprendizado por Transferência e Pré-Treinamento

A aplicação de **transfer learning** em forecasting permite aproveitar conhecimento obtido em séries semelhantes.
Exemplo: um modelo pré-treinado em dados de consumo de energia pode ser ajustado para uma nova região com poucos dados disponíveis.
Técnicas recentes empregam embeddings temporais e ajustes finos (*fine-tuning*) de Transformers, acelerando o treinamento e melhorando a generalização.

***

#### 11.5 Modelagem Probabilística e Incerteza

Modelos modernos incorporam **estimativas de incerteza** junto às previsões pontuais, fornecendo intervalos de confiança adaptativos.
Essas abordagens são cruciais para **tomada de decisão sob risco**, especialmente em negócios e operações críticas.
Métodos comuns incluem:

- Prophet e ARIMA com intervalos baseados em erro padrão.
- Redes bayesianas e ensembles (*Bagging*, *Dropout* como aproximação Bayesiana).
- Modelos **Quantile Regression** para diferentes percentis (ex.: prever P50, P90, P95).

***

#### 11.6 Aprendizado Online e Forecasting em Tempo Real

Em sistemas dinâmicos, novos dados chegam continuamente, tornando necessário o **aprendizado incremental**.
Técnicas de **online learning** ajustam os parâmetros sem reprocessar toda a base histórica.
Frameworks como **River**, **sktime-streams** e **MLOps pipelines** com reentrenamento contínuo garantem previsões sempre atualizadas.

***

#### 11.7 Benchmarking e AutoML em Séries Temporais

Ferramentas de **AutoML temporal** ajudam a encontrar automaticamente a melhor combinação de modelos, parâmetros e métricas:

- **AutoTS**, **Kats**, **Darts**, **sktime**, **nixtla (StatsForecast)**.
Essas soluções implementam comparação entre ARIMA, Prophet, LSTM, Transformers e suas variações, automatizando validação temporal e seleção de hiperparâmetros.

***

#### 11.8 Considerações Computacionais

Modelos de deep learning exigem:

- **Escalonamento temporal consistente** (normalização por janela).
- **Grande volume de dados** para evitar overfitting.
- **Hardware especializado** (GPU/TPU) quando o horizonte de previsão é extenso.

Para aplicações práticas, recomenda-se começar com métodos clássicos, evoluindo para redes neurais somente quando o ganho justifica o custo computacional.

***

#### 11.9 Tendências Atuais e Futuras

Entre as principais frentes de pesquisa e desenvolvimento estão:

- **Forecasting multimodal:** combinação de texto, imagem e séries temporais (por exemplo, sensores + relatórios).
- **Explainable Forecasting Models:** modelos de previsão interpretáveis com atenção visualizável.
- **Energia de modelos eficientes:** uso de Transformers leves (Lite/Informer).
- **Forecasting causal:** identificação de relações de causa e efeito, não apenas correlações.

Essas tendências mostram a convergência entre análise temporal, aprendizado profundo e inteligência artificial explicável.

***

#### 11.10 Conclusão

Os conceitos avançados ampliam o alcance das técnicas de previsão, permitindo lidar com **grandes volumes de dados, múltiplas variáveis e dinâmicas altamente não lineares**. Combinando a **robustez estatística dos métodos clássicos** com o **poder de generalização do deep learning**, os pesquisadores e praticantes têm hoje um conjunto de ferramentas capaz de enfrentar desafios reais em previsão de séries temporais — da energia renovável à manutenção preditiva.

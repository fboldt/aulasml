

## 1. Introdução aos Sistemas de Recomendação

### 1. Introdução aos Sistemas de Recomendação 

#### 1.1 Conceito e Motivação

- Sistemas de recomendação são **algoritmos de filtragem de informação** projetados para sugerir itens relevantes a usuários, com base em padrões de comportamento e preferências observadas.
- Surgiram como resposta à sobrecarga informacional no ambiente digital — a impossibilidade prática de um usuário explorar manualmente todas as opções disponíveis.
- Seu objetivo central é **aumentar a personalização e engajamento**, reduzindo o esforço cognitivo do usuário e maximizando o valor entregue por uma plataforma (ex.: retenção, conversão ou tempo de uso).


#### 1.2 Contexto Histórico

- A primeira geração de recomendadores (década de 1990) surgiu com métodos **baseados em regras** e **filtragem colaborativa manual** (GroupLens foi um dos primeiros projetos experimentais).
- A partir dos anos 2000, grandes plataformas de e-commerce e mídia (Amazon, YouTube, Netflix) começaram a usar modelos estatísticos e fatoração de matrizes.
- A era moderna inclui uso de **deep learning e modelos baseados em grafos**, permitindo recomendações contextuais, multimodais e personalizadas em tempo real.


#### 1.3 Aplicações Típicas

- **Música e vídeo:** Spotify, Netflix, YouTube utilizam recomendações para sequenciar conteúdos de forma personalizada.
- **E-commerce:** Amazon e Shopee sugerem produtos com base em perfis de compra e comportamento de navegação.
- **Redes sociais:** TikTok e Instagram ajustam o feed conforme interações e padrões de atenção.
- **Educação:** plataformas adaptativas (Coursera, Duolingo) personalizam trilhas de aprendizado.


#### 1.4 Fontes de Dados e Requisitos

- **Explícitas:** avaliações numéricas, curtidas, favoritos.
- **Implícitas:** tempo de visualização, cliques, histórico de navegação, abandonos.
- **Conteúdo:** metadados dos itens, descrições textuais, características multimodais (imagens, áudio).
- A qualidade das recomendações depende do **volume, variedade e veracidade** dos dados coletados.


#### 1.5 Benefícios

- Melhora a **experiência do usuário** e a **eficiência de busca**.
- Aumenta métricas de negócio, como **retenção**, **engajamento** e **venda cruzada**.
- Promove **descoberta de novos itens** e **diversificação de consumo**.


#### 1.6 Desafios e Aspectos Éticos

- **Cold start:** dificuldade de recomendar para novos usuários/itens.
- **Viés e bolhas de filtro:** o sistema pode reforçar comportamentos e limitar a diversidade de conteúdo.
- **Privacidade:** recomendação personalizada exige dados sensíveis, exigindo governança e conformidade (ex.: LGPD, GDPR).
- **Explicabilidade:** cresce a demanda por sistemas que justifiquem suas recomendações, especialmente em áreas sensíveis (educação, saúde, finanças).


#### 1.7 Estrutura Geral de um Sistema de Recomendação

- **Coleta e pré-processamento:** normalização de ratings e logs de interação.
- **Modelagem:** escolha entre filtering colaborativo, baseado em conteúdo ou modelos híbridos.
- **Geração de candidatos:** seleção inicial de itens relevantes com base em similaridade ou modelos preditivos.
- **Reranking e avaliação:** ordenação final considerando contexto, diversidade e métricas específicas.
- **Servir e monitorar:** integração do modelo no ambiente de produção, com monitoramento de desempenho e deriva de dados.


#### 1.8 Exemplos em Python

- Biblioteca **Surprise**: ideal para experimentação rápida com filtragem colaborativa.
- **LightFM**: combina fatores latentes e conteúdo (modelo híbrido).
- **TensorRec** e **implicit**: versões flexíveis com embeddings e aprendizado profundo.

***

### 2. Filtering Colaborativo 

#### 2.1 Conceito Fundamental

- O **collaborative filtering (CF)** é uma abordagem para recomendação baseada no princípio de que **usuários semelhantes têm gostos semelhantes**.
- A suposição central é de **correlação de preferências**: se dois usuários avaliaram itens de forma parecida no passado, eles tenderão a fazê-lo novamente no futuro.
- O método ignora as características intrínsecas dos itens — dependendo apenas da **interação usuário–item** (ex.: notas, cliques, tempo de uso).


#### 2.2 Representação Matemática

- Uma base de recomendação pode ser representada por uma matriz \$ R \in \mathbb{R}^{m \times n} \$, onde:
    - \$ m \$: número de usuários.
    - \$ n \$: número de itens.
    - \$ R_{ui} \$: rating do usuário \$ u \$ para o item \$ i \$.
- O objetivo é **preencher os valores ausentes** de \$ R \$, ou seja, prever quais itens o usuário pode gostar.
- A similaridade entre vetores de avaliações pode ser medida por:

$$
\text{sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
$$

(similaridade do cosseno), ou por correlações de Pearson.


#### 2.3 Abordagens Principais

1. **User-based Filtering:**
    - Identifica um conjunto de *vizinhos* similares ao usuário-alvo e usa suas avaliações para estimar notas não observadas.
    - Exemplo: um usuário que gosta dos mesmos filmes que outro usuário provavelmente gostará de novos filmes bem avaliados por ele.
2. **Item-based Filtering:**
    - Calcula similaridade entre itens e recomenda os mais parecidos com os já consumidos.
    - Mais eficiente quando o número de itens é menor que o de usuários, pois as relações entre itens mudam menos com o tempo.

#### 2.4 Cálculo de Predição em Filtering Colaborativo

- A predição da nota de um usuário \$ u \$ para um item \$ i \$ pode seguir:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u, v) (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u, v)|}
$$

onde:
    - \$ \bar{r}_u \$ é a média das notas do usuário \$ u \$.
    - \$ N(u) \$ é o conjunto de vizinhos semelhantes a \$ u \$.


#### 2.5 Desafios Computacionais

- **Escalabilidade:** a matriz \$ R \$ é geralmente muito esparsa (menos de 1% de preenchimento), o que torna o cálculo de similaridades computacionalmente caro.
- **Sparsity problem:** há poucos dados em comum para comparar usuários ou itens novos.
- **Cold start:** o sistema não consegue gerar boas recomendações para novos usuários ou itens sem histórico.
- **Atualização dinâmica:** em sistemas com milhões de interações por segundo (como Netflix ou Amazon), é difícil recalcular similaridades de modo eficiente.


#### 2.6 Melhorias e Extensões

- **Neighborhood smoothing:** introdução de termos de regularização para balancear a contribuição de vizinhos com poucos dados.
- **Trust-based CF:** ponderação baseada em relações de confiança explícitas entre usuários (útil em redes sociais).
- **Matrix factorization:** alternativa para lidar com sparsity e aprender *fatores latentes* (tratado no Tópico 5).
- **Híbridos com conteúdo:** complementam falhas do CF puro em domínios de item novo.


#### 2.7 Experimentos e Implementação

- **Bibliotecas comuns:**
    - *Surprise*: oferece implementações diretas de KNNBasic, KNNWithMeans, KNNBaseline.
    - *LightFM*: mistura CF e embeddings com gradiente descendente.
- **Pipeline típico:**

1. Preparação dos dados (ratings ou implícitos).
2. Definição da métrica de similaridade (cosine, Pearson, etc.).
3. Cálculo de vizinhos e geração de predições.
4. Avaliação com *cross-validation* e métricas como RMSE, Precision@K e Recall@K.


#### 2.8 Exemplo Numérico

Considere três usuários (A, B, C) avaliando três filmes:


| Usuário | Matrix (Filmes 1–3) | Média |
| :-- | :-- | :-- |
| A | [5, 4, ?] | 4.5 |
| B |  | 4.7 |
| C | [1, 2, ?] | 1.5 |

A similaridade de A e B (pelo cosseno) é alta, enquanto A e C têm correlação negativa. Logo, a predição para o Filme 3 de A tenderá a aproximar-se da média ponderada de \$ B \$’s avaliações ≈ 5.

#### 2.9 Impacto e Aplicações

- Base de recomendação do **Netflix Prize (2006)** foi estruturada em CF, levando à adoção global do modelo de vizinhança e fatoração de matrizes.
- CF ainda é um dos pilares de recomendação moderna, servindo como base para extensões em **deep learning**, **graph-based recommenders** e **context-aware models**.

***



### 3. Filtering Baseado em Conteúdo 

#### 3.1 Conceito e Fundamentos

- O **Content-Based Filtering (CBF)** recomenda itens com **características semelhantes** àqueles que o usuário já demonstrou interesse.
- Parte do princípio de que **as preferências do usuário são consistentes**, ou seja, se um usuário gosta de um item com certas propriedades, provavelmente gostará de outros com atributos semelhantes.
- Diferente do filtering colaborativo, o CBF **não depende de outros usuários**, mas apenas dos atributos dos itens e do histórico individual do usuário.


#### 3.2 Estrutura Geral do Sistema

Um sistema baseado em conteúdo geralmente possui três componentes principais:

1. **Perfil do usuário:** vetor representando suas preferências inferidas (por exemplo, gêneros de filmes, palavras-chave em textos, diretores, etc.).
2. **Descrição dos itens:** representação vetorial de cada item, geralmente extraída de texto ou metadados.
3. **Mecanismo de matching:** calcula similaridade entre o perfil do usuário e os vetores de itens, ranqueando-os conforme relevância.

Matematicamente, dada uma matriz de características \$ X \$ dos itens e um vetor de preferência \$ p_u \$ do usuário, o escore de similaridade é:

$$
\text{score}(i, u) = \cos(p_u, x_i) = \frac{p_u \cdot x_i}{\|p_u\| \|x_i\|}
$$

#### 3.3 Construção do Perfil do Usuário

O perfil pode ser construído:

- **De forma explícita:** o usuário informa manualmente preferências (por exemplo, marcando gêneros de filmes).
- **De forma implícita:** o sistema aprende a partir das interações passadas (ex.: avaliações ou tempo gasto em cada item).

Uma abordagem comum é calcular o perfil como média ponderada das features dos itens avaliados positivamente pelo usuário:

$$
p_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui} \cdot x_i
$$

onde \$ I_u \$ é o conjunto de itens avaliados e \$ r_{ui} \$ é a nota dada.

#### 3.4 Representação de Itens

Os itens podem ser descritos por diferentes tipos de características:

- **Textuais:** títulos, descrições e resenhas, representadas via TF-IDF, Bag-of-Words ou embeddings (Word2Vec, BERT).
- **Categóricas:** gênero, autor, ano, idioma, codificados por *one-hot encoding*.
- **Multimodais:** imagens, áudio, vídeo — extraindo embeddings gerados por redes neurais convolucionais (CNNs) ou transformers.

Essas representações geralmente são concatenadas ou incorporadas em **vetores densos de alta dimensionalidade** para cálculos de similaridade.

#### 3.5 Similaridade e Ranqueamento

Os métodos mais utilizados para comparar o perfil do usuário e os itens são:

- **Similaridade do cosseno**: mede o ângulo entre vetores (mais comum).
- **Distância Euclidiana inversa**: mede diferença geométrica direta.
- **Correlação de Pearson**: útil quando há normalização por médias.

O sistema retorna um ranking de itens ordenados pelo escore de similaridade.

#### 3.6 Vantagens

- Funciona bem com **poucos usuários** (não depende de dados coletivos).
- Evita o **problema de cold start** do tipo “novo usuário”, pois pode começar com um perfil básico.
- Produz **recomendações explicáveis**: é possível justificar a recomendação com as características do item (ex.: “recomendado porque você assistiu a outros filmes de drama com o mesmo diretor”).


#### 3.7 Limitações

- **Cold start de itens:** novos itens não podem ser recomendados se suas características não forem conhecidas.
- **Falta de diversidade:** tende a repetir o mesmo tipo de item, gerando pouca exploração (“efeito bolha”).
- **Representação limitada:** depende da qualidade e completude das features dos itens.
- **Dificuldade de capturar preferências latentes:** o modelo não infere semelhanças que não estejam expressas explicitamente nos dados.


#### 3.8 Extensões e Modelos Avançados

- **Aprendizado de Representação (Representation Learning):** uso de *autoencoders* ou *transformers* para gerar embeddings semânticos de usuários e itens.
- **Deep Content-Based Models:** aplicam redes neurais profundas para combinar múltiplas modalidades de dados (texto, imagem, áudio).
- **Context-Aware Recommender Systems:** incluem variáveis contextuais como local, horário ou dispositivo.
- **Exploração com Reinforcement Learning:** sistemas que balanceiam exploração e exploração (exploring new items vs. exploiting known preferences).


#### 3.9 Exemplo Prático (Python)

Usando *scikit-learn*:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descrições dos itens
descricoes = [
    "filme de ação com cenas futuristas",
    "drama psicológico ambientado em uma prisão",
    "ação e ficção científica com robôs"
]

# Vetorização TF-IDF
vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(descricoes)

# Similaridade entre itens
similaridade = cosine_similarity(matriz_tfidf)
```

Essa matriz de similaridade pode ser usada diretamente para gerar recomendações de itens semelhantes ao que o usuário visualizou.

#### 3.10 Aplicações Reais

- **Netflix:** usa CBF para citar filmes semelhantes ao assistido (“porque você viu…”).
- **Spotify:** recomenda faixas com base em embeddings de áudio e metadados.
- **Amazon:** combina descrição textual e categoria para sugerir produtos relacionados.
- **Goodreads:** recomenda livros similares por gênero, autor e palavras-chave das sinopses.

***




### 4. Modelos Híbridos 

#### 4.1 Motivação e Conceito Geral

- Os **modelos híbridos** surgem como resposta às **limitações dos métodos puramente colaborativos ou baseados em conteúdo**.
- O objetivo é combinar múltiplas fontes de informação — ratings, conteúdo, e dados contextuais — para gerar recomendações **mais precisas, diversificadas e robustas**.
- Em essência, o híbrido busca capturar tanto a **semelhança entre usuários** (colaboração) quanto a **semelhança entre itens e perfis individuais** (conteúdo).


#### 4.2 Tipos de Combinação

Segundo Burke (2002), há seis formas principais de combinação híbrida:

1. **Weighted Hybrid**
    - Combina os resultados de múltiplos algoritmos atribuindo pesos às suas saídas:

$$
\text{score}(i, u) = \alpha \cdot \text{CF}(i, u) + \beta \cdot \text{CBF}(i, u)
$$
        - \$ \alpha \$ e \$ \beta \$ são ponderações ajustadas empiricamente ou por validação.
        - Vantagem: simplicidade e controle sobre o impacto de cada modelo.
2. **Switching Hybrid**
    - O sistema escolhe dinamicamente qual modelo usar com base em critérios predefinidos.
    - Exemplo: usar filtering colaborativo quando há dados suficientes de interações e baseado em conteúdo em caso de cold start.
3. **Mixed Hybrid**
    - Itens de ambos os sistemas (CBF e CF) são apresentados conjuntamente no ranking final, promovendo diversidade.
    - Comum em plataformas como Netflix e Amazon, que misturam recomendações “semelhantes ao seu histórico” com “populares entre usuários similares”.
4. **Feature Augmentation**
    - Um modelo usa a saída ou representações internas de outro como entrada.
    - Exemplo: embeddings aprendidos pela fatoração de matrizes podem ser usados como features no modelo baseado em conteúdo.
5. **Cascade Hybrid**
    - Um algoritmo gera candidatos iniciais e outro refina o ranking final.
    - Exemplo: CBF filtra itens por similaridade textual e CF reordena conforme padrões coletivos de preferência.
    - Muito usado em sistemas de larga escala e pipelines de deep learning para recomendação (ex.: YouTube Recommender).
6. **Meta-level Hybrid**
    - Um modelo aprende a partir da estrutura interna de outro, não apenas de sua saída.
    - Exemplo: usar representações vetoriais aprendidas por uma rede neural de CF para aprimorar a fase de filtragem baseada em conteúdo.

#### 4.3 Modelagem Matemática Simplificada

Um modelo híbrido generalizado pode ser expresso como:

$$
\hat{r}_{ui} = \sum_{k=1}^{K} \lambda_k \cdot f_k(u, i)
$$

onde:

- \$ f_k(u, i) \$: escore fornecido pelo k-ésimo modelo (CF, CBF, etc.);
- \$ \lambda_k \$: peso ou coeficiente de mistura otimizado via aprendizado supervisionado ou heurístico;
- \$ K \$: número de modelos combinados.

A otimização pode ocorrer via ajuste direto dos \$ \lambda_k \$ por **gradient descent** ou via algoritmos de ensemble (como *stacking* e *boosting*).

#### 4.4 Vantagens dos Modelos Híbridos

- **Mitigam cold start e sparsity**: usam características de itens e padrões de grupo simultaneamente.
- **Melhoram precisão e cobertura**: recomendam itens relevantes mesmo para usuários novos ou nichos de conteúdo.
- **Aumentam diversidade**: combinam preferências pessoais e popularidade coletiva.
- **Permitem reciclar modelos legados**: integram sistemas pré-existentes (CBF e CF) sem reescrever toda a arquitetura.


#### 4.5 Desafios e Considerações

- **Complexidade de implementação**: combinam diferentes fontes e formatos de dados.
- **Custo de treinamento**: podem envolver múltiplos modelos e camadas de integração.
- **Interpretação de resultados**: a transparência da recomendação é reduzida.
- **Tuning de pesos e hiperparâmetros**: requer validação cuidadosa ou aprendizado adaptativo.
- **Problema de redundância**: se os modelos combinados forem muito correlacionados, a diversidade de resultados pode não aumentar.


#### 4.6 Aplicações Reais

- **Netflix:** combina modelos CF baseados em embeddings, regressões lineares, features de conteúdo (gênero, elenco, produção) e ajustes contextuais (ex.: horário, dispositivo).
- **Spotify:** integra CBF (para identificar faixas similares), CF (para playlists colaborativas) e redes neurais de recomendação seqüencial.
- **Amazon:** utiliza score híbrido ponderado entre personalização e tendências globais para manter diversidade e novidade.


#### 4.7 Modelos Híbridos com Aprendizado Profundo

O avanço das redes neurais levou ao surgimento dos **Deep Hybrid Models**, que unificam embeddings aprendidos de usuários e itens em arquiteturas neurais:

- **Wide \& Deep Networks (Google, 2016):** combinam features explícitas (wide) com embeddings aprendidos (deep).
- **Neural Collaborative Filtering (NCF):** integra representações latentes não-lineares de CF e conteúdo.
- **DeepFM e xDeepFM:** combinam fatoração de interações com aprendizado profundo para modelar relações de alto nível entre features.

Matematicamente, o modelo híbrido profundo pode ser formalizado como:

$$
\hat{r}_{ui} = \sigma(W_2 \cdot \text{ReLU}(W_1[x_u \oplus x_i]) + b_2)
$$

onde \$ x_u \$ e \$ x_i \$ são embeddings aprendidos de usuários e itens, e \$ \oplus \$ denota concatenação vetorial.

#### 4.8 Exemplo de Implementação Simplificada (Python)

```python
# Exemplo híbrido simples: combinação ponderada de CF e CBF
import numpy as np

def hybrid_score(cf_score, cbf_score, alpha=0.7):
    return alpha * cf_score + (1 - alpha) * cbf_score

# Exemplo: usuário com pontuação CF e CBF pré-calculadas
cf_scores = np.array([0.92, 0.85, 0.60])
cbf_scores = np.array([0.80, 0.75, 0.90])

final_scores = hybrid_score(cf_scores, cbf_scores, alpha=0.65)
print(final_scores)
```

Esse código demonstra um modelo híbrido **ponderado**, ajustável via parâmetro \$ \alpha \$, controlando a influência dos dois sistemas.

#### 4.9 Direções de Pesquisa Contemporâneas

- **Graph-based Hybrid Models:** representação de usuários e itens em grafos heterogêneos, com aprendizado via *Graph Neural Networks (GNNs)*.
- **Explainable AI (XAI)** aplicada à recomendação híbrida, promovendo transparência e interpretabilidade.
- **Fairness e diversidade algorítmica:** balanceamento entre precisão individual e equidade populacional.
- **Continual Learning:** atualização incremental de embeddings e pesos híbridos em ambientes de dados dinâmicos.

***



### 5. Fatoração de Matrizes 

#### 5.1 Motivação e Visão Geral

- A **fatoração de matrizes** é uma técnica fundamental em filtering colaborativo baseado em modelos.
- Seu objetivo é **descobrir padrões latentes** em uma matriz de interações usuário-item extremamente esparsa.
- Em vez de depender de similaridades diretas (como no CF tradicional), o modelo busca representar usuários e itens em **espaços vetoriais de menor dimensão**, onde a proximidade geométrica indica semelhança de preferência.

***

#### 5.2 Representação Matemática

Seja \$ R \in \mathbb{R}^{m \times n} \$ uma matriz de classificações (ratings) onde:

- \$ m \$: número de usuários
- \$ n \$: número de itens
- \$ R_{ui} \$: nota atribuída pelo usuário \$ u \$ ao item \$ i \$

O método busca decompor \$ R \$ em dois vetores latentes:

$$
R \approx P Q^T
$$

onde:

- \$ P \in \mathbb{R}^{m \times k} \$: matriz que representa cada **usuário** em um espaço de características latentes;
- \$ Q \in \mathbb{R}^{n \times k} \$: matriz que representa cada **item** no mesmo espaço;
- \$ k \ll \min(m, n) \$: número de dimensões latentes.

Cada elemento previsto \$ \hat{r}_{ui} \$ é dado por:

$$
\hat{r}_{ui} = p_u^T q_i
$$

ou, em forma expandida:

$$
\hat{r}_{ui} = \sum_{f=1}^k p_{uf} q_{if}
$$

onde \$ p_{uf} \$ e \$ q_{if} \$ representam a interação entre o fator \$ f \$ do usuário e do item.

***

#### 5.3 Função de Custo e Regularização

Para ajustar os vetores latentes, minimiza-se o erro quadrático entre as notas previstas e observadas:

$$
\min_{P, Q} \sum_{(u, i) \in K} (R_{ui} - p_u^T q_i)^2 + \lambda ( \|p_u\|^2 + \|q_i\|^2 )
$$

onde:

- \$ K \$: conjunto de pares usuário–item observados;
- \$ \lambda \$: parâmetro de regularização que previne overfitting.

A otimização pode ser feita com:

- **Stochastic Gradient Descent (SGD)**;
- **Alternating Least Squares (ALS)**;
- **Coordinate Descent** em casos esparsos.

***

#### 5.4 Interpretação dos Fatores Latentes

- Os **fatores latentes** não são observáveis diretamente, mas capturam dimensões implícitas de gosto, como gênero, estilo, faixa etária-alvo etc.
- Exemplo: em um sistema de filmes, um fator pode representar a preferência por *ação* vs. *romance*, outro por *drama pesado* vs. *comédia leve*.
- Isso torna a fatoração de matrizes uma forma de **aprendizado de representação não supervisionado**.

***

#### 5.5 Fatoração com Viés (Bias-aware Factorization)

Modelos mais sofisticados incorporam **tendências globais e individuais**:

$$
\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i
$$

onde:

- \$ \mu \$: média global das notas;
- \$ b_u \$: viés do usuário (tende a dar notas mais altas ou baixas);
- \$ b_i \$: viés do item (alguns itens são geralmente mais bem avaliados).

Esse modelo foi amplamente utilizado no **Netflix Prize**, com ganhos significativos em RMSE em relação às abordagens puramente colaborativas.

***

#### 5.6 Variantes de Fatoração

1. **SVD (Singular Value Decomposition)**
    - Versão clássica em álgebra linear, aplicada quando \$ R \$ é densa.
    - Não diretamente aplicável a dados esparsos, mas serviu de base para SVD regularizada.
2. **SVD++**
    - Extende o SVD incorporando informações de interações implícitas (ex.: cliques, visualizações parciais).
    - A predição torna-se:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T \left( p_u + |N(u)|^{-\frac{1}{2}} \sum_{j \in N(u)} y_j \right)
$$

onde \$ y_j \$ são fatores adicionais derivados dos itens que o usuário interagiu.
3. **NMF (Non-negative Matrix Factorization)**
    - Impõe restrição \$ P, Q \ge 0 \$, tornando os fatores mais interpretáveis.
    - Popular em aplicações onde interpretabilidade é crucial (ex.: recomendação de notícias).
4. **Bayesian PMF (Probabilistic Matrix Factorization)**
    - Modela incerteza dos parâmetros via inferência bayesiana, útil em contextos ruidosos e dados esparsos.

***

#### 5.7 Relação com Deep Learning

A fatoração de matrizes inspirou diversos métodos de recomendação neural, como:

- **Neural Matrix Factorization (NeuMF):** substitui a multiplicação \$ p_u^T q_i \$ por uma rede neural profunda, permitindo capturar relações não lineares.
- **Autoencoders Colaborativos (CAE):** utilizam aprendizado não supervisionado para reconstruir perfis de usuários e inferir itens faltantes.
- **Graph Neural Networks (GNNs):** generalizam a fatoração considerando estruturas de grafo em interações usuário–item.

***

#### 5.8 Exemplo de Implementação (Python)

Um exemplo simples usando o *Surprise*:

```python
from surprise import SVD, Dataset, accuracy
from surprise.model_selection import cross_validate

# Carregar base de dados MovieLens
data = Dataset.load_builtin('ml-100k')

# Modelo de fatoração SVD
model = SVD(n_factors=50, reg_all=0.02, lr_all=0.005)

# Avaliação via cross-validation
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

Esse modelo aplica fatoração de matrizes regularizada para prever notas em um dataset de filmes, permitindo avaliar precisão (RMSE e MAE).

***

#### 5.9 Vantagens e Desvantagens

**Vantagens**

- Captura padrões complexos e latentes.
- Escalável para milhões de usuários e itens com otimizações paralelas.
- Generaliza melhor que métodos memória-baseados (vizinhanca).

**Desvantagens**

- Falha em casos de cold start (usuários/itens novos).
- Pouca interpretabilidade intuitiva das dimensões latentes.
- Necessidade de ajuste fino de parâmetros e número de fatores \$ k \$.

***

#### 5.10 Aplicações Reais e Extensões

- **Netflix:** fatoração de matrizes foi núcleo do modelo vencedor do Netflix Prize.
- **Amazon e Spotify:** usam fatoração em embeddings combinada com redes neurais.
- **YouTube:** usa técnicas híbridas de fatoração e deep learning (cascateadas) para filtragem de candidatos.

***




### 6. Problemas de Cold Start 

#### 6.1 Conceito e Impacto

- O **problema de Cold Start** ocorre quando um sistema de recomendação precisa **gerar previsões sem dados históricos suficientes**.
- Esse desafio afeta especialmente abordagens **colaborativas**, que dependem de interações passadas (avaliações, cliques, tempo de uso).
- É um dos obstáculos mais críticos em ambientes dinâmicos, como **plataformas de streaming, e-commerce e redes sociais**, onde novos usuários e itens surgem constantemente.

Existem **três tipos principais de cold start**:

1. **Usuário frio (user cold start)** — um novo usuário entra no sistema sem histórico de preferências.
2. **Item frio (item cold start)** — novos produtos, filmes ou músicas são adicionados sem avaliações prévias.
3. **Sistema frio (system cold start)** — o sistema é recém-implantado e ainda não possui volume suficiente de dados para gerar recomendações significativas.

***

#### 6.2 Causas e Natureza do Problema

O cold start deriva de **esparsidade extrema dos dados**:

- Em uma matriz de interações usuário-item \$ R \$, a maioria dos valores é desconhecida, o que reduz a capacidade preditiva.
- Quando o conjunto \$ R_{ui} \$ de interações conhecidas tende a zero, as abordagens baseadas em vizinhança ou fatoração de matrizes **não conseguem gerar vetores latentes confiáveis**.
- O problema também é agravado por **mudanças de contexto** — novos produtos, modas, ou comportamentos sazonais alteram os padrões subjacentes de consumo.

***

#### 6.3 Estratégias para Mitigação

Diversas abordagens têm sido propostas para contornar o cold start, geralmente categorizadas em **três grupos complementares**:

##### a) Estratégias Baseadas em Conteúdo

- Em caso de **item cold start**, usa-se os **atributos dos itens** (metadados, descrições, gênero, elenco, etc.) para realizar recomendações iniciais.
- Exemplo: recomendar um novo filme de ficção científica para usuários que já assistiram e avaliaram positivamente filmes desse gênero.
- Ferramentas como **TF-IDF**, **Word2Vec** e **BERT embeddings** podem vetorializar descrições e permitir similaridade sem histórico.


##### b) Estratégias Baseadas em Perfil

- O sistema pede ao **usuário novo** informações explícitas durante o cadastro:
    - questionários ou seleção de categorias de interesse;
    - integração com redes sociais (dados demográficos ou de preferências públicas).
- Essas informações iniciais compõem um **perfil frio inicial**, usado até que interações reais comecem a ser coletadas.


##### c) Estratégias Híbridas e Modelos Neurais

- Combinam dados de conteúdo com padrões de interação (quando disponíveis).
- Modelos *deep-hybrids* aprendem simultaneamente embeddings de itens e usuários:

$$
\hat{r}_{ui} = f_\theta(x_u, x_i)
$$

onde \$ x_u \$ e \$ x_i \$ são vetores de características de usuário e item, e \$ f_\theta \$ é uma rede neural parametrizada.
- Esses modelos capturam relações não lineares, generalizando melhor para usuários e produtos novos.

***

#### 6.4 Cold Start em Diferentes Contextos

| Tipo de Cold Start | Origem do Problema | Soluções Frequentes |
| :-- | :-- | :-- |
| Usuário novo | Falta de interações | Questionários, dados demográficos, login social |
| Item novo | Falta de avaliações | Conteúdo textual, imagens, atributos do item |
| Sistema novo | Baixo volume de dados | Dados sintéticos, popularidade inicial, transferência de modelos |


***

#### 6.5 Técnicas Avançadas

1. **Modelos de Transferência e Aprendizado Multitarefa**
    - Reutilizam embeddings ou parâmetros de domínios semelhantes (ex.: usar modelo treinado de filmes para iniciar um recomendador de séries).
    - Técnicas como *pretraining* e *fine-tuning* reduzem a necessidade de grandes volumes de dados novos.
2. **Contextualização e Feedback Implícito**
    - Considera sinais indiretos de interesse, como tempo de permanência, cliques, ou scrolls.
    - Algoritmos context-aware ajustam a recomendação inicial com base em tempo, localização, e dispositivo.
3. **Exploração Ativa (Active Learning)**
    - O sistema pergunta estrategicamente ao usuário por avaliações em itens com **alto potencial informativo**.
    - O modelo aprende mais rápido com menos interações, equilibrando exploração e conforto do usuário.
4. **GNNs e Modelos de Propagação de Informação**
    - Redes neurais baseadas em grafos (e.g., GraphSAGE, GAT) tratam o problema de cold start explorando conexões semânticas entre nós.
    - Mesmo sem avaliações diretas, esses métodos podem inferir preferências a partir de similaridades estruturais no grafo de usuários e itens.

***

#### 6.6 Exemplo Prático de Solução Híbrida

```python
# Exemplo: abordagem simples de cold start com embeddings de conteúdo
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Embeddings de dois filmes novos (extraídos de descrições)
item_embeddings = np.array([[0.8, 0.1, 0.4], [0.7, 0.2, 0.5]])

# Perfil do usuário (aprendido de histórico anterior)
user_profile = np.array([0.75, 0.15, 0.45])

# Similaridade entre perfil e itens novos
scores = cosine_similarity([user_profile], item_embeddings)
print(scores)
```

Este trecho mostra uma forma simples de contornar cold start usando vetores de conteúdo para medir afinidade entre usuário e itens sem histórico de avaliações.

***

#### 6.7 Boas Práticas para Ambientes Reais

- **Coletar dados implícitos** desde o primeiro contato (scroll, hover, tempo de visualização).
- **Integrar múltiplas fontes de dados** (conteúdo, rede social, contexto temporal).
- **Aplicar modelos adaptativos** que aprendem continuamente (aprendizado online).
- **Analisar métricas específicas** para cold start, como *warm-up time* e *initial engagement rate*.

***

#### 6.8 Pesquisas Recentes e Tendências

- **Meta-learning para recomendação fria:** modelos que aprendem a aprender preferências rapidamente (inspirado em few-shot learning).
- **Language-driven cold start:** uso de *Large Language Models (LLMs)* para interpretar descrições e gerar embeddings ricos em semântica.
- **MLOps para recomendação dinâmica:** pipelines automatizados de atualização e reengenharia de features quando novos itens entram no catálogo.
- **Ética e privacidade:** estratégias de mitigação devem seguir marcos regulatórios (LGPD, GDPR) e evitar oversharing de dados sensíveis.

***

#### 6.9 Conclusão

O problema de **cold start** representa o **ponto de fragilidade mais comum** em sistemas de recomendação contemporâneos.
Sua superação exige integração profunda entre técnicas **estatísticas, de aprendizado profundo e de engenharia de dados**, com foco em:

- aprendizado incremental,
- exploração ativa,
- e personalização orientada por contexto e conteúdo.

Essa compreensão é essencial para projetar **sistemas de recomendação escaláveis, justos e responsivos**, capazes de proporcionar boas sugestões desde o primeiro contato do usuário com a plataforma.


***

### 7. Avaliação de Sistemas de Recomendação 

#### 7.1 Importância da Avaliação

- A avaliação é um **componente essencial** no ciclo de desenvolvimento de sistemas de recomendação, pois permite medir objetivamente a **eficácia, precisão e utilidade** das recomendações.
- Um sistema pode gerar sugestões plausíveis, mas **avaliar se ele realmente melhora a experiência do usuário e atende aos objetivos de negócio** exige métricas quantitativas e qualitativas bem definidas.
- A ausência de uma metodologia rigorosa pode levar a **overfitting, viés de recomendação e interpretações incorretas de desempenho**.

***

#### 7.2 Tipos de Avaliação

Há **três modalidades principais** de avaliação em sistemas de recomendação:

1. **Avaliação Offline**
    - Baseada em dados históricos e métricas computadas sobre *datasets* de treino, validação e teste.
    - Permite comparações rápidas entre modelos, antes da implantação real.
    - Exemplo: uso do *MovieLens dataset* para testar algoritmos sob controle total dos dados.
2. **Avaliação Online (A/B Testing)**
    - Requer ambiente de produção.
    - Mede o impacto real das recomendações em métricas de negócio (taxa de clique, retenção, engajamento).
    - Divide o tráfego em grupos (controle e teste) e compara resultados de forma estatisticamente robusta.
3. **Avaliação por Usuários (User Studies)**
    - Analisa percepções qualitativas de usuários (relevância percebida, satisfação, diversidade).
    - Importante em fases de design e em recomendações de domínios sensíveis (educação, saúde, finanças).

***

#### 7.3 Estrutura Experimental Offline

1. **Coleta e particionamento de dados**
    - Divisão do histórico em conjuntos de treino, validação e teste.
    - Estratégias:
        - *Random split:* embaralha e divide aleatoriamente (usado em datasets densos).
        - *Temporal split:* separa por tempo, simulando cenários de recomendação real.
        - *Leave-One-Out:* retém uma interação por usuário para teste (comum em aprendizado de ranking).
2. **Construção do pipeline de recomendação**
    - Treinamento do modelo sobre o conjunto de treino.
    - Predição de interações futuras sobre o conjunto de teste.
    - Cálculo das métricas quantitativas.

***

#### 7.4 Métricas Quantitativas

##### a) Erros de Predição (para sistemas baseados em notas)

- **MAE (Mean Absolute Error):**

$$
MAE = \frac{1}{|\Omega|} \sum_{(u,i)\in\Omega} |R_{ui} - \hat{R}_{ui}|
$$
- **RMSE (Root Mean Squared Error):**

$$
RMSE = \sqrt{\frac{1}{|\Omega|} \sum_{(u,i)\in\Omega} (R_{ui} - \hat{R}_{ui})^2}
$$
    - Penaliza mais fortemente erros grandes.
    - Foi a métrica de referência no **Netflix Prize**.


##### b) Métricas de Ranking (para sistemas baseados em ordenação)

- **Precision@K:** proporção de itens relevantes entre os top-K recomendados.

$$
Precision@K = \frac{|Rel_u \cap Rec_u|}{K}
$$
- **Recall@K:** proporção de itens relevantes recuperados entre todos os relevantes.

$$
Recall@K = \frac{|Rel_u \cap Rec_u|}{|Rel_u|}
$$
- **F1@K:** harmônico entre precisão e revocação.
- **nDCG@K (Normalized Discounted Cumulative Gain):** mede a relevância ponderada pela posição no ranking.

$$
nDCG@K = \frac{1}{IDCG@K} \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$


##### c) Outras métricas relevantes

- **Coverage:** proporção de itens do catálogo que foram recomendados ao menos uma vez.
- **Diversity:** mede variedade entre itens recomendados (ex.: distância média de similaridade).
- **Novelty e Serendipity:** avaliam o grau de ineditismo e surpresa útil da recomendação.
- **Mean Reciprocal Rank (MRR):** avalia a posição do primeiro item relevante em uma lista ranqueada.

***

#### 7.5 Avaliação Online e Controle Experimental

- **A/B Testing:**
    - O método divide usuários em grupos que recebem diferentes versões do sistema.
    - Exemplo: versão A (algoritmo base) × versão B (novo modelo híbrido).
    - Métricas de comparação típicas:
        - Taxa de cliques (CTR)
        - Tempo médio de sessão
        - Conversão (vendas, downloads, cadastros)
    - A significância estatística é avaliada com testes como **t-test** ou **bootstrap**.
- **Interleaving:** técnica que mescla resultados de dois modelos nos rankings exibidos, reduzindo variância e tempo de teste online.

***

#### 7.6 Métricas Qualitativas e Exploratórias

Além das métricas numéricas, recomenda-se incluir **avaliações subjetivas e contextuais**, especialmente em domínios de interação humana:

- **Satisfação percebida:** questionários pós-uso (ex.: escalas Likert).
- **Utilidade percebida:** relevância e timeliness (adequação temporal da recomendação).
- **Diversidade semântica:** avaliação da variedade temática por especialistas.
- **Aderência ética e não enviesada:** análise do equilíbrio entre grupos populacionais e categorias.

***

#### 7.7 Ferramentas e Bibliotecas de Avaliação

- **Surprise** — fornece métricas MAE, RMSE, FCP e métodos de *cross-validation*.
- **RecBole** — framework para avaliação de recomendações heterogêneas (CF, CBF, GNNs).
- **Implicit e LightFM** — permitem avaliação baseada em métricas de ranking.
- **Elliot** — infraestrutura recente para experimentação reprodutível em larga escala.

Exemplo básico em Python com *Surprise*:

```python
from surprise import Dataset, SVD
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')
model = SVD()
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5)
```


***

#### 7.8 Desafios e Armadilhas Comuns

- **Overfitting experimental:** ajustes excessivos às métricas offline sem considerar generalização real.
- **Viés de posição:** itens mais visíveis são mais clicados, mesmo que menos relevantes.
- **Efeito de retroalimentação:** recomendações influenciam os próprios dados que serão usados para nova avaliação.
- **Viés de popularidade:** métricas podem favorecer itens amplamente consumidos.
- **Métricas desalinhadas com o objetivo:** por exemplo, otimizar RMSE não necessariamente melhora engajamento.

***

#### 7.9 Boas Práticas

- Combinar **métricas de precisão** (acurácia) e **métricas de experiência** (diversidade, novidade).
- Utilizar **validação temporal** em lugar de divisões aleatórias em domínios dinâmicos.
- Avaliar múltiplos aspectos do sistema: utilidade, justiça, privacidade e transparência.
- Empregar **dashboards de monitoramento contínuo**, integrados a pipelines de MLOps, para acompanhar degradação de desempenho.
- Documentar e reproduzir experimentos com frameworks padronizados.

***

#### 7.10 Tendências de Pesquisa em Avaliação

- **Learning-to-evaluate:** uso de modelos de ML para aprender métricas correlacionadas com engajamento real.
- **Conversational Evaluation:** avaliação dinâmica em sistemas interativos (chatbots e recomendadores com diálogo).
- **Fairness e Explainability Metrics:** novas medidas que equilibram precisão com justiça e transparência.
- **Offline-to-Online Correlation:** pesquisas em mapeamento entre desempenho offline e ganhos reais em produção.

***

#### 7.11 Conclusão

A avaliação de sistemas de recomendação vai além da medição de precisão: ela **determina a confiabilidade científica e prática** do modelo.
O uso combinado de **métricas quantitativas, qualitativas e de negócio** — aliado a experimentação controlada e monitoramento contínuo — é a base para sistemas **robustos, éticos e centrados no usuário**, capazes de evoluir com segurança em ambientes reais.


***

### 8. Práticas Avançadas e Tendências em Sistemas de Recomendação 

#### 8.1 Evolução e Contexto Geral

- Os **sistemas de recomendação modernos** evoluíram de modelos estatísticos simples para arquiteturas **neuro-simbólicas, multimodais e contextuais**.
- Atualmente, frameworks de deep learning, aprendizado online e computação distribuída tornaram as recomendações **dinâmicas, personalizadas e escaláveis em tempo real**.
- O foco deixou de ser apenas **predizer notas** e passou a envolver **entendimento profundo do comportamento do usuário**, **explicabilidade** e **resiliência ética**.

***

#### 8.2 Modelos Neurais Profundos para Recomendação

O uso de redes neurais trouxe novas possibilidades de modelar interações complexas usuário–item:

1. **Neural Collaborative Filtering (NCF)**
    - Substitui a multiplicação escalar da fatoração de matrizes por uma rede neural:

$$
\hat{r}_{ui} = f_{\theta}(p_u, q_i)
$$

onde \$ f_{\theta} \$ é uma MLP (Multilayer Perceptron).
    - Captura **relações não lineares** e interações de ordem superior entre usuários e itens.
2. **Autoencoders Colaborativos (CF-AE e VAE)**
    - Reconstruem vetores de preferências com base em entradas parciais;
    - O *Variational Autoencoder (VAE)* introduz regularização probabilística, útil para lidar com ruído e sparsity extremos.
3. **Redes Recorrentes (RNNs) e Transformers**
    - Modelam **sequências de comportamento** (ex.: histórico de cliques e sessões).
    - Ideais para recomendação sequencial e predição de próxima interação.
    - Exemplo: *GRU4Rec* e *BERT4Rec* — capturam dependências temporais com excelente desempenho em fluxo contínuo de dados.
4. **Graph Neural Networks (GNNs)**
    - Representam usuários e itens como nós de um grafo bipartido;
    - Permitem **propagação de informação semântica** pelas conexões (ex.: homofilia e relacionamento implícito);
    - Frameworks populares: *PinSage (Pinterest)*, *LightGCN*, *GraphSAGE*.

***

#### 8.3 Recomendação Contextual (*Context-Aware Recommendation*)

- Esses modelos incorporam **variáveis contextuais** que influenciam as preferências do usuário:
    - Localização, hora do dia, dispositivo, clima, evento social etc.
- Exemplo: um restaurante pode ser recomendado apenas durante o horário de almoço no aplicativo de delivery.
- Abordagens típicas:
    - **Tensor factorization:** estende \$ R_{ui} \$ para \$ R_{uic} \$, introduzindo o contexto \$ c \$ como terceira dimensão.
    - **Contextual Bandits:** técnicas de aprendizado por reforço que ajustam recomendações baseadas no estado atual do usuário.

***

#### 8.4 Sistemas de Recomendação com Aprendizado por Reforço

- Utilizam **Reinforcement Learning (RL)** para maximizar recompensas cumulativas ao longo do tempo.
- Diferem dos modelos supervisionados, pois consideram efeitos de longo prazo — ex.: manter o usuário engajado durante várias sessões.
- Estrutura típica:
    - **Agente:** o recomendador.
    - **Ambiente:** usuários e catálogo.
    - **Ação:** item recomendado.
    - **Recompensa:** feedback (clique, compra, tempo de permanência).
- Algoritmos populares:
    - *Deep Q-Networks (DQN)*
    - *Policy Gradient*
    - *Actor–Critic Methods*
- Aplicações: personalização de playlists, feed dinâmico de notícias, anúncios adaptativos.

***

#### 8.5 Recomendação Multimodal

- Com o crescimento de dados ricos (imagens, áudio, texto, vídeos), a recomendação multimodal combina diferentes representações:
    - Exemplo: considerar simultaneamente *thumbnail*, *descrição textual* e *tags* de um vídeo no YouTube.
- Combinação pode ocorrer via:

1. **Early Fusion:** concatenação direta de embeddings.
2. **Late Fusion:** combinação das saídas de redes separadas.
3. **Cross-modal Attention:** aprendizado de pesos entre modalidades (inspirado em arquiteturas Transformer).
- Ferramentas modernas: *CLIP*, *Multimodal Transformers (MMT)*, *DeepCross*.

***

#### 8.6 Explicabilidade e Transparência (XAI em Recomendação)

- Explicar recomendações fortalece **confiança e engajamento do usuário**.
- Abordagens:
    - **Modelo pós-hoc:** análise de similaridades, visualização de embeddings e exemplos de justificativas (“Recomendado porque você ouviu...”).
    - **Modelo intrinsecamente explicável:** utiliza estruturas lógicas ou atenção interpretável.
- Técnicas comuns:
    - *LIME*, *SHAP* e *Integrated Gradients* para visualizar contribuição das features.
- Desafios: equilibrar interpretabilidade e complexidade sem comprometer desempenho.

***

#### 8.7 Fairness, Ética e Viés Algorítmico

- Sistemas de recomendação podem amplificar desigualdades, concentrando visibilidade em poucos itens ou grupos.
- Fontes típicas de viés:
    - Dados históricos enviesados;
    - Popularidade desbalanceada;
    - Falta de diversidade de conteúdo.
- Estratégias de mitigação:
    - **Fairness-aware learning:** introduz restrições de equidade no treinamento;
    - **Debiasing de dados:** reamostragem e reponderação;
    - **Modelos adversariais:** neutralizam informações sensíveis.
- Exemplo: garantir que artistas independentes recebam exposição equivalente à de grandes gravadoras.

***

#### 8.8 Integração com MLOps e Produção

- Grandes plataformas aplicam **MLOps para recomendação contínua**, integrando:
    - coleta de dados em tempo real,
    - retraining automatizado,
    - monitoramento de deriva de conceito (*concept drift*).
- Estrutura típica do pipeline:

1. Feature Store → treinamento incremental → avaliação contínua;
2. Deploy de modelo → observabilidade → ajuste automático de parâmetros.
- Ferramentas: *TensorFlow Recommenders*, *Kubeflow*, *Feast*, *Airflow*.

***

#### 8.9 Tendências Emergentes

1. **Large Language Models (LLMs) como Recomendadores**
    - Modelos como GPT e PaLM-2 são usados para gerar recomendações contextuais baseadas em prompts de linguagem natural.
    - Capacidade de integrar dados estruturados e não estruturados (“explique-me por que este produto combina comigo”).
2. **Graph-based Embeddings Unificados**
    - Aprendizado de uma rede de relações heterogêneas (usuário–item–contexto) usando *graph contrastive learning*.
3. **Recomendação Auto-supervisionada**
    - Uso de pretext tasks (ex.: predição de próxima interação mascarada) sem necessidade de rótulos manuais.
4. **Federated Recommendation**
    - Treinamento descentralizado, garantindo privacidade de usuários (sem compartilhar dados brutos).
    - Aplicável em domínios sensíveis (saúde, finanças, educação).
5. **Edge AI e Recomendação em Dispositivos Locais**
    - Execução de inferência no dispositivo (smartphone, smartwatch) para recomendações rápidas e privadas.

***

#### 8.10 Conclusão

Os sistemas de recomendação avançaram de pipelines estáticos para **ecossistemas inteligentes**, capazes de compreender contexto, multimodalidade e ética.
As tendências convergem para três eixos:

- **Profundidade** (aprendizado neural profundo e multimodal),
- **Adaptação** (recomendação dinâmica e contínua),
- **Responsabilidade** (transparência, equidade e privacidade).

Essas práticas formam a base das próximas gerações de recomendadores — **autônomos, interpretáveis e centrados no ser humano** — refletindo o papel da inteligência artificial como mediadora de decisões personalizadas.


***

### 9. Referências e Leituras Recomendadas 

#### 9.1 Livros Fundamentais

1. **Ricci, F., Rokach, L., Shapira, B. — *Recommender Systems Handbook (3ª Edição, Springer, 2021)*.**
    - Considerado o guia mais completo da área, cobre desde métodos simples de filtragem colaborativa até técnicas avançadas de deep learning, fairness e explicabilidade.
    - Ideal para estudantes de pós-graduação e pesquisadores, pois combina teoria, algoritmos e casos de uso industriais.
2. **Aggarwal, C. C. — *Recommender Systems: The Textbook (Springer, 2016)*.**
    - Texto clássico com ênfase em modelos matemáticos e avaliação experimental.
    - Contém formulações detalhadas de métodos fatorados, híbridos e contexto-dependentes.
    - Inclui tópicos de escalabilidade e recomendação em redes sociais.
3. **Zhang, S., Yao, L., Sun, A., \& Tay, Y. — *Deep Learning based Recommender System: A Survey and New Perspectives (ACM Computing Surveys, 2019).*
    - Revisão abrangente sobre modelos neurais para recomendação, mapeando arquiteturas, datasets e métricas.
    - Propõe taxonomia moderna para redes convolucionais, recorrentes e de atenção aplicadas ao problema de recomendação.
4. **Jannach, D., Lerche, L., Kamehkhosh, I., \& Jugovac, M. — *Recommender Systems: An Introduction (Cambridge, 2010).*
    - Apresenta conceitos práticos, incluindo design de interface e interação humano-sistema.
    - Excelente referência para entender o ciclo completo de projeto de recomendadores, do algoritmo ao impacto no usuário.
5. **Davidson, J., et al. — *The YouTube Video Recommendation System (RecSys Conference, 2010).*
    - Artigo seminal que documenta o pipeline industrial de recomendação mais escalável do mundo, baseado em regressão logística e filtragem colaborativa.
    - Mostra os desafios de latência, diversidade e feedback-loop no ambiente de recomendação em tempo real.

***

#### 9.2 Leituras Avançadas e Pesquisas Recentes

1. **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., \& Chua, T.-S. — *Neural Collaborative Filtering (WWW, 2017).*
    - Propõe o modelo NCF, marco inicial dos sistemas de recomendação neurais, substituindo a multiplicação de fatorações por MLPs.
2. **Wang, X., He, X., et al. — *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR, 2020).*
    - Introduz uma forma eficiente de redes de grafos para recomendação, removendo redundâncias das arquiteturas GNN.
3. **Zhou, G., et al. — *Deep Interest Network for Click-Through Rate Prediction (AAAI, 2019).*
    - Modelo de referência para recomendação baseada em comportamento de cliques, muito usado em publicidade personalizada.
4. **Sun, F., et al. — *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations (CIKM, 2019).*
    - Aplica Transformers com codificação bidirecional ao histórico sequencial do usuário, superando RNNs em precisão preditiva.
5. **Hric, M., Darst, R. K., \& Fortunato, S. — *Community Detection in Networks: Structural vs. Attribute Information (Scientific Reports, 2016).*
    - Fundamenta o uso de redes complexas e grafos heterogêneos na modelagem social para recomendação baseada em comunidade.

***

#### 9.3 Artigos Clássicos (Históricos e Seminais)

- **Resnick, P., \& Varian, H. (1997).** *Recommender Systems*. *Communications of the ACM, 40(3)* — primeiro artigo a definir formalmente o paradigma de recomendação na web.
- **Sarwar, B. et al. (2001).** *Item-based Collaborative Filtering Recommendation Algorithms*. *WWW Conference* — introduz o modelo item-based amplamente adotado até hoje.
- **Koren, Y., Bell, R., \& Volinsky, C. (2009).** *Matrix Factorization Techniques for Recommender Systems*. *IEEE Computer* — artigo fundamental do Netflix Prize.
- **Burke, R. (2002).** *Hybrid Recommender Systems: Survey and Experiments*. *User Modeling and User-Adapted Interaction* — classificação seminal das arquiteturas híbridas.
- **Adomavicius, G., \& Tuzhilin, A. (2005).** *Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions.* *IEEE Transactions on Knowledge and Data Engineering.*

***

#### 9.4 Ferramentas e Frameworks de Implementação

1. **Surprise**
    - Biblioteca Python voltada a experimentos acadêmicos de CF e avaliação (RMSE, Precision@K, etc.).
    - Ideal para prototipagem rápida e validação offline.
2. **LightFM**
    - Framework híbrido que combina embeddings de conteúdo e interação implícita.
    - Suporte a treinamentos usando *warp loss* e *logistic loss*, com integração ao NumPy/SciPy.
3. **TensorFlow Recommenders (TFRS)**
    - Desenvolvido pelo Google, facilita pipelines de deep recommendation e integração com TensorFlow Extended (TFX).
    - Permite uso de GNNs, embeddings e aprendizado sequencial em larga escala.
4. **RecBole**
    - Framework de pesquisa modular desenvolvido em PyTorch.
    - Inclui +90 algoritmos clássicos e neurais, padronizando processos de carga, treino, avaliação e comparação.
5. **Implicit e PySpark ALS**
    - Voltadas a grande escala e recomendação implícita.
    - Implementam fatoração de matrizes com *Alternating Least Squares* distribuído.

***

#### 9.5 Datasets Públicos de Referência

| Dataset | Descrição | Nº de Usuários/Itens | Tipo |
| :-- | :-- | :-- | :-- |
| **MovieLens (100k, 1M, 20M)** | Clássico dataset de ratings de filmes | 6k usuários / 3k filmes | Explícito |
| **Amazon Product Data** | Dados reais de compras e reviews | milhões de pares | Implícito |
| **Last.fm e Million Song Dataset** | Escuta musical e metadados de artistas | 1M usuários / 400k faixas | Implícito |
| **Goodbooks-10K** | Avaliações e categorias de livros | 53k usuários / 10k livros | Explícito |
| **Pinterest / Steam / Yelp** | Conjuntos multimodais incluindo texto e imagem | variável | Híbrido |

Esses datasets são amplamente usados em benchmarks de artigos científicos e cursos de pós-graduação.

***

#### 9.6 Recursos Complementares

- **Cursos Online**
    - *Coursera – Recommender Systems Specialization* (University of Minnesota).
    - *Deep Learning for Recommender Systems* (Udemy / fast.ai).
- **Conferências Prioritárias**
    - *RecSys* (ACM Recommender Systems Conference).
    - *KDD, WWW, SIGIR, AAAI, NeurIPS*: principais fóruns de pesquisa sobre IA aplicada a recomendação.
- **Repositórios GitHub**
    - *Awesome Recommender Systems* — coleção curada de algoritmos, papers e datasets.
    - *RecBole-library* — código aberto para experimentação e reprodução de resultados de artigos.

***

#### 9.7 Sugestão de Trilhas de Leitura

**Trilha teórica (introdução e conceitos):**
Resnick \& Varian (1997) → Ricci et al. (2021) → Aggarwal (2016).

**Trilha prática (implementação e avaliação):**
Jannach (2010) → Koren (2009) → LightFM ou TFRS frameworks.

**Trilha avançada (aprendizado profundo):**
Zhang et al. (2019) → He et al. (2017) → Wang et al. (2020) → Zhou et al. (2019).

**Trilha ética e explicável:**
Burke (2002) → Zhang et al. (2023, XAI in Recsys) → Ricci et al. cap. 33.

***

#### 9.8 Conclusão

O estudo das referências listadas fornece uma **base sólida de três camadas**:

- **Clássica:** fundamentos teóricos e matemáticos (Koren, Burke, Ricci).
- **Neural e Moderna:** redes profundas, GNNs e multimodais (He, Zhang, Sun).
- **Crítica e Ética:** transparência, diversidade e impacto social (Adomavicius, Ricci, Shapira).

Essas leituras consolidam o domínio em **sistemas de recomendação inteligentes, escaláveis e responsáveis**, integrando ciência de dados, engenharia de software e princípios de inteligência artificial aplicada.


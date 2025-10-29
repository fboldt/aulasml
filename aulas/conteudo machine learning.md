# Disciplina de Machine Learning - Conteúdo Detalhado das 24 Aulas

---

## 1 — Introdução ao Machine Learning

- Conceitos e aplicações de ML, diferença para programação tradicional
- Paradigmas: supervisionado, não-supervisionado, semi-supervisionado, reforço
- Workflow de projetos em ML; etapas de um pipeline
- Apresentação do ambiente prático (Jupyter, scikit-learn) e datasets clássicos
- Reflexão sobre limitações, ética, viés e desafios sociais

---

## 2 — Regressão Linear e Métodos Básicos

- Regressão linear simples e múltipla: matemática e interpretação de coeficientes
- Ajuste por mínimos quadrados, regularização (Ridge, Lasso, ElasticNet)
- Métricas de avaliação: MAE, MSE, RMSE, R²
- Demonstração prática em scikit-learn com datasets reais

---

## 3 — Técnicas de Validação e Cross-Validation

- Importância da validação e riscos do overfitting
- Técnicas: Holdout, K-Fold, Stratified K-Fold, LOOCV, Nested CV, Time Series CV
- Implementação em Python, integração em pipelines, métricas de avaliação
- Exercícios: comparar validação simples, estratificada e temporal

---

## 4 — Hyperparameter Tuning e AutoML

- Diferença entre parâmetro e hiperparâmetro
- GridSearch, RandomizedSearch, Bayesian Optimization, Optuna
- Fundamentos de AutoML (TPOT, Auto-sklearn)
- Boas práticas de tuning com validação cruzada e análise crítica de resultados

---

## 5 — Classificação Binária e Multiclasse

- Regressão logística e fundamentos da classificação
- Estratégias OvR, OvO, Softmax para multiclasse
- Métricas: acurácia, precision, recall, F1, matriz de confusão, ROC-AUC
- Pipeline prático para binário e multiclasse

---

## 6 — K-Nearest Neighbors e Otimização

- Algoritmo, escolha de K, métricas de distância (Euclidiana, Manhattan, Minkowski)
- Uso de cross-validation para tuning
- Aplicações em classificação e regressão; comparação com modelos lineares

---

## 7 — Support Vector Machines (SVM)

- Separação máxima, hiperplano, conceito de margem e support vectors
- Kernel trick: kernels lineares, polinomial, RBF
- Tuning de C, gamma, degree; validação cruzada
- SVM para classificação, regressão e detecção de anomalias

---

## 8 — Modelos Probabilísticos

- Teorema de Bayes e paradigmas probabilísticos
- Naive Bayes (gaussiano, multinomial, binário), aplicações e limitações
- Gaussian Processes: intuição, regressão, classificação e incerteza

---

## 9 — Árvores de Decisão

- Algoritmos: ID3, C4.5, CART
- Critérios: entropia, ganho de informação, índice Gini, MSE
- Poda (pruning), overfitting, feature importance
- Visualização de árvores e caminhos de decisão

---

## 10 — Engenharia e Seleção de Features
- Criação, transformação e limpeza de variáveis
- Encoding, imputação, discretização
- Técnicas de seleção: filter, wrapper, embedded (Lasso, Árvores)
- Feature importance, permutation importance, integração com pipelines

---

## 11 — Dados Desbalanceados
- Problemas e consequências do desbalanceamento
- Técnicas: Oversampling, Undersampling, SMOTE, cost-sensitive learning
- Métricas apropriadas: F1, ROC-AUC, PR-AUC, G-mean, matriz de confusão especial
- Exercícios com detecção de fraude, churn, saúde

---

## 12 — Testes de Significância Estatística
- Conceito de hipótese nula (H₀) e hipótese alternativa (H₁)
- Erros tipo I e tipo II, níveis de significância (α)
- Testes paramétricos: t-test (uma e duas amostras), ANOVA
- Testes não-paramétricos: Mann-Whitney, Kruskal-Wallis, Wilcoxon
- Interpretação de valores-p, confiabilidade e poder do teste
- Aplicações práticas em Machine Learning: comparação de modelos por métricas
- Implementação em Python (scipy.stats) e visualização de distribuições amostrais
---

## 13 — Clustering: K-Means e Variações

- Algoritmo, inicialização, passo-a-passo
- Elbow method, Silhouette Score para escolha de k
- Clustering como feature engineering, KMeans++
- Aplicações e limitações práticas

---

## 14 — Outros Métodos de Clustering

- Hierarchical (agglomerative/divisive), linkage, dendrogramas
- DBSCAN: densidade e outliers
- GMM: modelagem probabilística, soft clustering
- Critérios para escolha; benchmark em dados reais

---

## 15 — Redução de Dimensionalidade

- PCA: teoria, variância explicada, aplicações em compressão e visualização
- Kernel PCA: relações não-lineares
- t-SNE, UMAP: visualização de alta dimensionalidade, clusters, estruturas complexas
- Exercícios de reconstrução e análise crítica

---

## 16 — Séries Temporais e Forecasting

- EDA temporal: tendência, sazonalidade, lags, rolling features
- Validação temporal (TimeSeriesSplit)
- Modelos ARIMA, Prophet, tuning de p/d/q, inferência de tendência e previsões
- Exercício prático em datasets reais: energia, vendas, finanças

---

## 17 — Detecção de Anomalias

- Isolation Forest, LOF, One-Class SVM
- Outlier detection em dados tabulares e séries temporais
- Métricas adequadas, tuning de thresholds, comparação de métodos
- Estudo de caso: fraude, falhas industriais, segurança

---

## 18 — Semi-supervisionado e Active Learning

- Self-training, co-training, label propagation
- Fundamentos, aplicações em rotulagem eficiente, NLP, imagens, diagnóstico médico
- Estratégias de active learning: uncertainty sampling, oracle queries
- Simulação de pipelines reais com dados rotulados e não rotulados

---

## 19 — Sistemas de Recomendação

- Filtering colaborativo (usuário-usuário, item-item)
- Filtering baseado em conteúdo, engenharia de features
- Modelos híbridos, matrix factorization (SVD, NMF); desafios de cold start
- Avaliação: RMSE, Recall@k, Precision@k, A/B, abordagens em MovieLens

---

## 20 — NLP Clássico

- Pré-processamento: tokenização, stopwords, lematização, limpeza
- Bag-of-Words, TF-IDF, n-grams, features especiais de textos
- Classificação de texto com ML tradicional (NB, SVM, logística)
- Pipeline completo e análise de erros/textos extraídos

---

## 21 — Algoritmos de Otimização (Genéticos, Swarm, SA)

- Genetic Algorithms: ciclo, seleção, crossover, mutação, fitness
- Particle Swarm Optimization: atualização, social/cognitivo, aplicações em tuning/contínuo
- Simulated Annealing: resfriamento, soluções piores, escape de mínimos locais
- Uso prático em ajuste de hiperparâmetros, roteamento, scheduling

---

## 22 — Interpretabilidade e Explicabilidade

- Feature importance clássica e permutation importance
- LIME: explicação local e visual
- SHAP: baseada em teoria dos jogos, importância global/local
- Comunicação de decisões, explicabilidade em contextos críticos, limitações e pitfalls

---

## 23 — MLOps, Serialização e Deploy

- Serialização (pickle, joblib, ONNX)
- Deploy via APIs RESTful com FastAPI/Flask; construção de endpoints /predict
- Containerização com Docker, pipeline CI/CD
- Monitoramento e versionamento, logging, checagem de drift, práticas de MLOps básico

---

## 24 — Projeto Final, Revisão e Perspectivas

- Apresentação de projetos: problema, pipeline, justificativas, resultados e desafios
- Feedback e debate coletivo, revisão dos principais temas do curso
- Perspectivas futuras: trilhas avançadas em ML, ética, open science, tendências em IA
- Espaço para Q&A, avaliação do curso e encaminhamentos individuais

---

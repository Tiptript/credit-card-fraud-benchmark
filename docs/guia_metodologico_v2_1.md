# Guia de Estudos — As melhorias da v2.1 e por que elas eram necessárias

Este documento cobre cada conceito que estava ausente (ou errado) no TCC original e que a v2.1 implementa. Para cada tema: o que é, por que o seu trabalho especificamente precisava dele, como está implementado no código, as armadilhas clássicas, e o que ler. A ordem dos temas é a ordem sugerida de estudo — os primeiros são fundamentos que invalidavam o trabalho; os últimos são extensões de estado da arte.

---

## 1. Disciplina train / validation / test e vazamento de dados

**O que é.** A regra mais importante de todo o aprendizado de máquina aplicado: qualquer decisão tomada com base nos dados — hiperparâmetro, threshold, escolha de features, escolha do próprio modelo, até a normalização — consome informação. Se essa decisão usa o conjunto de teste, a estimativa de desempenho fica otimista, porque você ajustou o sistema ao gabarito. O teste existe para ser tocado *uma única vez*, no fim, com tudo congelado. Toda seleção acontece na validação (ou em validação cruzada dentro do treino).

**Por que o seu TCC precisava.** O resultado-manchete (F1 = 0,8247 da Regressão Logística) veio de um threshold escolhido maximizando F1 *no conjunto de teste* — a forma mais direta de vazamento de seleção. E o ranking final fazia "duplo mergulho": para cada modelo, escolhia o melhor entre quatro pipelines *também olhando o teste*. A conclusão central do trabalho ("o ajuste de threshold foi a etapa mais determinante") era tautológica: otimizar uma métrica no gabarito sempre melhora a métrica no gabarito.

**Vazamento tem várias formas, e você precisa reconhecer todas:**

*Vazamento de seleção* — escolher threshold/hiperparâmetro/modelo no teste (o seu caso principal). *Vazamento de pré-processamento* — ajustar `StandardScaler`, PCA ou SMOTE nos dados completos antes do split; a média e o desvio do teste "contaminam" o treino. O texto da sua metodologia dizia que o balanceamento foi aplicado "antes da divisão", o que seria exatamente isso (o código fazia diferente do texto — outro problema). *Vazamento temporal* — treinar com dados do futuro para prever o passado (tema 2). *Vazamento de feature* — usar uma variável que só existe por causa do alvo (ex.: flag de chargeback para prever fraude).

**Como está na v2.1.** Três partições temporais (70/15/15). Thresholds (`threshold_max_f1`, `threshold_min_cost`) recebem apenas `y_val`/`p_val`. O tuning do XGBoost avalia candidatos na validação. Scaler e SMOTE vivem dentro de `Pipeline`, então são ajustados apenas quando `.fit()` roda no treino. O teste aparece uma única vez, no fim, com tudo congelado.

**Armadilha clássica.** "Mas eu só olhei o teste uma vez para escolher o threshold" — não importa a quantidade de olhadas; importa que *qualquer* decisão condicionada ao teste vicia a estimativa. Outra: rodar o experimento inteiro, ver o resultado, "melhorar" o código e rodar de novo no mesmo teste — cada iteração dessas é um grau de liberdade do pesquisador gasto contra o mesmo gabarito (o problema do *garden of forking paths*).

**Ler:** Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, cap. 7 (Model Assessment and Selection) — a referência definitiva; Cawley & Talbot (2010), "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", *JMLR* — o paper que formaliza exatamente o erro que você cometeu.

---

## 2. Validação out-of-time e concept drift

**O que é.** Quando os dados têm ordem temporal e o uso real do modelo é prever o futuro, o protocolo de avaliação deve imitar isso: treinar no passado, testar no futuro. Split aleatório embaralha o tempo e deixa o modelo "ver o futuro" — transações de amanhã no treino ajudam a classificar transações de ontem no teste. *Concept drift* é o fenômeno de a distribuição mudar com o tempo (fraudadores adaptam táticas), que é justamente o que torna a validação temporal obrigatória e a mais severa: modelos avaliados out-of-time quase sempre pontuam abaixo do split aleatório, e essa diferença é a estimativa honesta.

**Por que o seu TCC precisava.** Fraude é o caso canônico de drift — o seu próprio texto cita o conceito na Discussão e o descarta alegando "ausência de dados temporais", o que é falso: a coluna `Time` existe no dataset e é plotada no seu notebook. Pior: com split aleatório e `Time` como feature, o modelo pode memorizar a *posição temporal* das fraudes vizinhas que vazaram do teste para o treino. Dal Pozzolo — o criador do dataset que você usou, citado nas suas referências — construiu a tese dele inteira em torno de tratar esses dados como um *stream* temporal.

**Como está na v2.1.** `temporal_split` ordena por `Time` e corta em 70/15/15; o teste é o período mais recente. `Time` bruto sai da matriz de features e vira `hour` (hora do dia), um padrão cíclico legítimo — fraude noturna é um sinal real e generalizável, enquanto o índice monotônico bruto não extrapola.

**Armadilha.** Estratificar um split temporal (forçar a mesma taxa de fraude em cada partição) destrói a temporalidade — aceite que as partições terão prevalências diferentes; isso é realismo, não defeito. Outra: fazer CV k-fold comum em dados temporais; o correto é `TimeSeriesSplit` (janelas expansivas ou deslizantes).

**Ler:** Dal Pozzolo (2015), *Adaptive Machine Learning for Credit Card Fraud Detection* (tese, ULB) — leitura obrigatória, é o manual do seu próprio dataset; Gama et al. (2014), "A Survey on Concept Drift Adaptation", *ACM Computing Surveys*.

---

## 3. PR-AUC vs ROC-AUC em desbalanceamento extremo

**O que é.** A curva ROC plota TPR contra FPR; a curva Precision–Recall plota precisão contra recall. Com prevalência de 0,17%, o FPR é dividido por ~284 mil negativos: um modelo que gera 3.000 falsos positivos (inutilizável na prática — cada FP é uma revisão manual) tem FPR de ~1% e parece excelente na ROC. A precisão, ao contrário, é dividida pelos *alertas*, então sente cada falso positivo. Por isso a PR-AUC (ou *Average Precision*) é a métrica de ranqueamento correta em desbalanceamento extremo, e a ROC-AUC fica sistematicamente inflada.

**Por que o seu TCC precisava.** Suas tabelas reportavam ROC-AUC como métrica de qualidade, e o próprio resultado gritava o problema: o Naive Bayes tinha a *melhor* ROC-AUC do baseline (0,9342) e terminou com o *pior* F1 (0,3261) — uma contradição que o texto nunca discute e que a PR-AUC teria exposto de imediato. O seu notebook até calculava curvas PR num bloco, mas a AP nunca entrou nas tabelas.

**Como está na v2.1.** `average_precision_score` é a métrica primária de ranqueamento (colunas `ap_val`, `ap_test_mean`); ROC-AUC é reportada como secundária; o baseline aleatório da PR (= prevalência) aparece nas curvas para dar escala.

**Armadilha.** Interpolação linear entre pontos da curva PR superestima a área — use `average_precision_score`, não `auc(recall, precision)` do trapézio. E lembre: AP mede *ranqueamento* (o modelo ordena bem?); F1/precisão/recall medem a *decisão em um threshold* — são perguntas diferentes (tema 4).

**Ler:** Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", *PLOS ONE*; Davis & Goadrich (2006), "The Relationship Between Precision-Recall and ROC Curves", *ICML*.

---

## 4. Threshold como decisão econômica: aprendizado sensível a custo

**O que é.** Um classificador probabilístico produz um score; a *classificação* é uma decisão separada, que depende de custos. A teoria da decisão bayesiana dá a resposta fechada: com probabilidade calibrada p e custos C_FP e C_FN, bloqueie quando `p > C_FP / (C_FP + C_FN)` (Elkan, 2001). Em fraude, o custo é *dependente do exemplo*: um FN custa o valor da transação (chargeback), então uma fraude de R$ 5.000 e uma de R$ 20 não deveriam ter o mesmo threshold implícito. Isso é o campo do *example-dependent cost-sensitive learning*.

**Por que o seu TCC precisava — e este é o ponto mais importante para você como economista.** O F1 pesa FP e FN igualmente, uma escolha economicamente arbitrária que o trabalho nunca justifica. Ao mesmo tempo, sua fundamentação teórica inteira (Coase, Williamson, custos de transação) *é* uma teoria de custos — e o dataset tem a coluna `Amount`. A ponte estava pronta e não foi feita: escolher o threshold minimizando perda esperada em R$ transforma a teoria de ornamento em método, e transforma o resultado de "F1 = 0,82" (que ninguém numa diretoria entende) em "este modelo economiza R$ X por quinzena versus aprovar tudo" (que qualquer banca de Economia entende). Bahnsen et al. aplicaram exatamente essa formulação a exatamente este dataset.

**Como está na v2.1.** `expected_cost` soma o `Amount` das fraudes perdidas (FN) mais `cost_fp` por transação legítima bloqueada. `threshold_min_cost` varre thresholds na validação minimizando esse custo. A tabela final ordena por `cost_test` e mostra as referências triviais ("aprovar tudo", "bloquear tudo") — se o seu modelo não bate "aprovar tudo", ele destrói valor. A análise de sensibilidade (tema 10) varre `cost_fp` em {2, 10, 50}.

**Armadilha.** Custo esperado exige probabilidade *calibrada* (tema 5) — um score mal calibrado dá um threshold ótimo sem sentido. E o custo de FP não é só a revisão manual: inclui atrito com o cliente (transação legítima negada gera churn), que é heterogêneo. Modelar isso é a extensão natural v2.2.

**Ler:** Elkan (2001), "The Foundations of Cost-Sensitive Learning", *IJCAI* — curto e fundamental; Bahnsen, Aouada & Ottersten (2014), "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring / Fraud Detection"; Hand (2009), "Measuring classifier performance: a coherent alternative to the area under the ROC curve", *Machine Learning*.

---

## 5. Calibração de probabilidades

**O que é.** Um modelo é calibrado se, entre as transações às quais ele atribui p = 0,3, cerca de 30% são de fato fraude. Calibração é ortogonal a discriminação: um modelo pode ranquear perfeitamente (AP alta) e ter probabilidades absurdas. Métodos: *Platt scaling* (regressão logística sobre o score, boa com poucos positivos) e *regressão isotônica* (não paramétrica, precisa de mais dados). Diagnóstico: curva de calibração (reliability diagram) e *Brier score* (erro quadrático médio da probabilidade).

**Por que o seu TCC precisava.** Três razões. O texto elogia as "probabilidades calibradas" da regressão logística sem nunca verificar. O LinearSVC não produz probabilidade nenhuma — `decision_function` é uma distância ao hiperplano, e o seu código a usava como se fosse score comparável. E toda a lógica de threshold ótimo (tema 4) pressupõe probabilidades com significado; sem calibração, o "threshold ótimo de 0,0181" é um número sem interpretação. Bônus: SMOTE e class_weight *distorcem* a calibração deliberadamente (mudam a prevalência efetiva de treino), então modelos balanceados precisam de recalibração ou de correção analítica da probabilidade — o tema central do paper do Dal Pozzolo que dá nome ao dataset ("Calibrating Probability with Undersampling").

**Como está na v2.1.** LinearSVC envolvido em `CalibratedClassifierCV(method="sigmoid", cv=3)` — a calibração é ajustada por CV interna *no treino*. `brier_score_loss` na tabela; `CalibrationDisplay` gera o gráfico de confiabilidade no teste. Para os não supervisionados, o score é normalizado min–max (pelos limites da *validação*) e o Brier é omitido deliberadamente, porque não é probabilidade.

**Ler:** Platt (1999), "Probabilistic Outputs for Support Vector Machines"; Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with Supervised Learning", *ICML* — mostra quais famílias de modelos saem calibradas e quais não; Dal Pozzolo et al. (2015), "Calibrating Probability with Undersampling for Unbalanced Classification", *IEEE SSCI* — a citação correta do seu dataset.

---

## 6. Ablação: um fator por vez

**O que é.** Para atribuir efeito a uma técnica, você compara o sistema com e sem ela, mantendo todo o resto constante. Se duas técnicas que atacam o mesmo problema estão sempre ligadas juntas, o efeito de cada uma é *não identificável* — vocabulário de econometria que você já domina: é um problema de identificação, idêntico a regressar y sobre duas variáveis perfeitamente colineares.

**Por que o seu TCC precisava.** Todos os modelos tinham `class_weight='balanced'` *sempre*; sobre isso aplicava-se SMOTE; sobre isso, threshold tuning. Três mecanismos anti-desbalanceamento empilhados. O achado mais interessante do trabalho ("SMOTE não melhorou o F1") era inaproveitável: o baseline nunca esteve desbalanceado do ponto de vista da função de perda, porque os pesos já compensavam.

**Como está na v2.1.** Cada modelo roda sob três estratégias mutuamente exclusivas: `baseline` (nada), `class_weight` (reponderação no modelo; no XGBoost, `scale_pos_weight`; no NB, priors uniformes), `smote` (oversampling no pipeline, sem pesos). Agora "SMOTE ajuda?" tem resposta limpa — e a literatura recente sugere que, com modelos fortes e threshold bem escolhido, a resposta é frequentemente *não* (o que, se confirmado nos seus dados, é o seu resultado central, agora com desenho válido).

**Ler:** He & Garcia (2009), "Learning from Imbalanced Data" (você já cita — releia com olhos de ablação); van den Goorbergh et al. (2022), "The harm of class imbalance corrections for risk prediction models", *JAMIA* — argumento moderno de que correção de desbalanceamento frequentemente piora calibração sem melhorar decisão.

---

## 7. Três incertezas diferentes: múltiplas seeds, bootstrap e CV

**O que é.** "Qual a incerteza do meu resultado?" tem pelo menos três respostas distintas, e um trabalho sério reporta as relevantes. (1) *Variabilidade algorítmica*: modelos estocásticos (florestas, boosting com subsample, redes, e qualquer coisa após SMOTE, que sorteia vizinhos) dão resultados diferentes por semente; rodar N seeds e reportar média ± desvio mede a instabilidade do algoritmo. (2) *Variabilidade amostral do teste*: com ~90 fraudes no teste, a métrica tem erro amostral grande; o bootstrap estratificado (reamostrar o teste com reposição, preservando o nº de positivos) dá um IC 95%. (3) *Variabilidade de split*: validação cruzada mede quão sensível o resultado é à partição — em dados temporais, via `TimeSeriesSplit`.

**Por que o seu TCC precisava.** Um split, uma seed, zero intervalos: a afirmação "Regressão Logística (0,8247) venceu a Árvore (0,7800)" era ininterpretável — com ~98 fraudes de teste, essa diferença cabe folgadamente dentro do ruído. E a Árvore de Decisão, que o próprio texto descreve como instável, foi comparada com uma execução única.

**Como está na v2.1.** `n_seeds = 5` para toda configuração estocástica (o critério inclui SMOTE: `seeds_for` marca como estocástico qualquer estratégia smote, mesmo com modelo determinista). A tabela traz `ap_test_mean ± ap_test_std` (incerteza algorítmica) e `ap_test_ci` por bootstrap (incerteza amostral). As decisões de threshold usam a *média das probabilidades entre seeds* — um seed-ensemble, que é um preditor mais estável do que qualquer seed individual.

**Armadilha.** Não confunda as três: um IC de bootstrap apertado não significa que o algoritmo é estável entre seeds, e vice-versa. E não use o desvio entre seeds para construir um "IC" com t de Student — as execuções compartilham os mesmos dados, não são independentes.

**Ler:** Bouckaert & Frank (2004), "Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms"; Efron & Tibshirani, *An Introduction to the Bootstrap*, caps. 1–6.

---

## 8. Comparação estatística de classificadores: McNemar

**O que é.** Para dizer que o modelo A é melhor que o B *no mesmo conjunto de teste*, o teste correto explora o pareamento: cada transação foi classificada pelos dois. O teste de McNemar olha apenas as discordâncias — b = casos em que só A erra, c = casos em que só B erra — e testa H0: b e c vêm de uma binomial(b+c, 0,5). Na versão exata, o p-valor sai de `binomtest`. Se b + c é pequeno, não há poder para distinguir os modelos, e é honesto dizer isso.

**Por que o seu TCC precisava.** O ranking final ordenava quatro números pontuais sem qualquer teste. "Superior a todos os demais experimentos" é uma afirmação estatística, e o trabalho não tinha estatística para fazê-la.

**Como está na v2.1.** `mcnemar_exact` compara as duas melhores configurações pelo critério de custo, nas predições do threshold econômico. No smoke test, p = 0,14 — ou seja, "não distinguíveis com estes dados", que é exatamente o tipo de conclusão que o TCC original nunca poderia ter alcançado (e que uma banca respeita mais do que um ranking falso-preciso).

**Ler:** Dietterich (1998), "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", *Neural Computation* — o paper canônico; Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets", *JMLR* — para quando houver vários datasets.

---

## 9. SMOTE feito direito (e por que talvez você nem precise dele)

**O que é.** SMOTE (Chawla et al., 2002) cria exemplos sintéticos da classe minoritária por interpolação entre vizinhos. Dois pontos de execução são inegociáveis: (1) só no treino — sintetizar antes do split cria "gêmeos" de observações do teste dentro do treino, inflando tudo; (2) dentro de um `imblearn.Pipeline`, para que em validação cruzada o SMOTE seja reajustado em cada fold de treino, nunca tocando o fold de avaliação.

**Por que o seu TCC precisava.** Além do texto contradizer o código sobre *quando* o SMOTE era aplicado, faltava a citação do paper original (usar SMOTE sem citar Chawla é como usar MQO sem citar... bem, é grave). Conceitualmente, faltava entender o que o SMOTE realmente faz: ele muda a prevalência efetiva, o que equivale aproximadamente a mover o threshold e distorcer a calibração — por isso, com threshold otimizado explicitamente (tema 4), o ganho do SMOTE tende a evaporar. Seu achado empírico apontava para isso; o desenho não permitia afirmá-lo.

**Como está na v2.1.** `SMOTE` como primeiro passo de um `ImbPipeline`, com `random_state` variando junto com as seeds, e como uma estratégia *isolada* na ablação.

**Ler:** Chawla et al. (2002), "SMOTE: Synthetic Minority Over-sampling Technique", *JAIR*; Elor & Averbuch-Elor (2022), "To SMOTE, or not to SMOTE?" — revisão empírica moderna que chega à conclusão provocativa do título.

---

## 10. Análise de sensibilidade da razão de custos

**O que é.** O threshold ótimo e até o *ranking* de modelos dependem da razão C_FP/C_FN. Como esses custos são estimativas (quanto custa exatamente uma revisão manual? e o atrito?), o resultado precisa ser testado contra variações plausíveis — o análogo direto da análise de sensibilidade que você já conhece de avaliação de projetos e de calibração de modelos econômicos.

**Por que o seu TCC precisava.** Com F1, a razão de custos estava implícita e escondida (1:1). Explicitá-la e varrê-la transforma uma fragilidade ("de onde veio esse custo?") em uma seção de resultados ("o modelo A domina para custos de revisão baixos; o B, para altos — a escolha do classificador é uma decisão econômica").

**Como está na v2.1.** `cost_fp_grid = [2, 10, 50]`; para as três melhores configurações, o threshold é reescolhido na validação para cada custo e o custo de teste recalculado (`cost_sensitivity.csv`). No smoke test o ranking já inverteu entre custos — exatamente o fenômeno a discutir.

**Ler:** Saltelli et al., *Global Sensitivity Analysis: The Primer*, cap. 1 (a lógica geral); o próprio Elkan (2001) para a dependência threshold–custos.

---

## 11. Gradient boosting e XGBoost

**O que é.** Boosting constrói um ensemble *sequencial*: cada árvore nova é ajustada ao gradiente do erro do ensemble atual (por isso "gradient boosting" — é descida de gradiente no espaço de funções). O XGBoost (Chen & Guestrin, 2016) adiciona regularização explícita na função objetivo, tratamento nativo de esparsidade, e implementação por histogramas que o torna rápido. Hiperparâmetros centrais: `n_estimators` × `learning_rate` (trocam entre si — mais árvores com passo menor), `max_depth` (interações entre features), `subsample`/`colsample_bytree` (aleatorização que reduz variância), `scale_pos_weight` (reponderação da classe positiva — o mecanismo canônico do XGB para desbalanceamento). Desde ~2016, GBMs (XGBoost/LightGBM/CatBoost) são o baseline dominante em dados tabulares — um comparativo sem GBM é considerado incompleto em qualquer venue.

**Por que o seu TCC precisava.** O estudo comparava quatro algoritmos de 1958–1995 em 2025, e a própria Discussão citava um trabalho com Random Forest (Mishra & Ghorpade) sem implementar nem RF nem boosting. A pergunta "qual modelo detecta melhor?" não pode ser respondida omitindo a família que vence os benchmarks há uma década.

**Diferença conceitual que vale internalizar:** Random Forest é *bagging* — árvores profundas independentes em amostras bootstrap, média reduz variância; boosting é *sequencial* — árvores rasas corrigindo erros, reduz viés. Por isso RF é mais robusto "out of the box" e boosting rende mais com tuning.

**Como está na v2.1.** `XGBClassifier` com `eval_metric="aucpr"`, `tree_method="hist"`, e um tuning leve de 3 candidatos avaliados na validação (`tune_xgb`) — deliberadamente pequeno, para demonstrar o protocolo correto sem explodir o tempo; a v2.2 substituiria por busca com `TimeSeriesSplit` aninhada. `HistGradientBoostingClassifier` do sklearn fica como segunda implementação da mesma família.

**Ler:** Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System", *KDD*; Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine", *Annals of Statistics* — a origem; Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on tabular data?", *NeurIPS* — por que GBM ≥ redes em tabular.

---

## 12. Isolation Forest: detecção de anomalias sem rótulo

**O que é.** Em vez de modelar as classes, o Isolation Forest (Liu et al., 2008) explora uma propriedade dos pontos raros: eles são fáceis de *isolar*. Constroem-se árvores com cortes aleatórios; o número médio de cortes até isolar um ponto é o score — anômalos se isolam em poucos cortes. Custo quase linear, sem usar o rótulo.

**Por que o seu TCC precisava.** A conclusão recomendava "aprendizado não supervisionado" para trabalhos futuros sem rodar o método que é uma linha de sklearn. Mais importante: o não supervisionado responde uma pergunta de negócio diferente — *e quando o padrão de fraude é novo e não há rótulo dele no histórico?* (concept drift, de novo). Um sistema real combina os dois: supervisionado para padrões conhecidos, anomalia para o desconhecido.

**Como está na v2.1.** `IsoForestScorer` treina no treino inteiro sem rótulo; o score de anomalia é normalizado pelos limites da *validação* (nem a normalização pode olhar o teste); os rótulos da validação entram só na escolha do threshold — protocolo semi-supervisionado padrão para comparar com supervisionados. Espere AP bem menor que os supervisionados: o valor do método é qualitativo (robustez a padrões novos), e reportar essa diferença honestamente é parte do resultado.

**Ler:** Liu, Ting & Zhou (2008), "Isolation Forest", *ICDM*; Chandola, Banerjee & Kumar (2009), "Anomaly Detection: A Survey", *ACM Computing Surveys*.

---

## 13. Autoencoders para detecção de fraude

**O que é.** Um autoencoder é uma rede neural treinada para reconstruir a própria entrada passando por um gargalo (aqui 30 → 16 → 8 → 16 → 30). Treinado *apenas em transações legítimas*, ele aprende a variedade (manifold) do comportamento normal; uma fraude, vinda de outra distribuição, é mal reconstruída, e o erro de reconstrução vira o score de anomalia. É a versão "deep learning" da detecção de anomalias — e a resposta concreta ao seu texto, que invocava deep learning como promessa futura sem nunca especificar arquitetura ou mecanismo.

**Por que o seu TCC precisava.** Além de materializar a recomendação vaga da sua conclusão, o autoencoder ensina duas ideias transferíveis: (1) aprender a distribuição do normal em vez da fronteira entre classes; (2) o erro de reconstrução como estatística de teste — que é, no fundo, um teste de aderência (goodness-of-fit) aprendido, um conceito que conversa com a sua formação estatística.

**Como está na v2.1.** `AutoencoderScorer` usa `MLPRegressor` do sklearn (X → X) com `early_stopping`, treinado só nas legítimas do treino; score = MSE de reconstrução por linha. Implementação deliberadamente simples e sem dependência de PyTorch/TensorFlow; a versão "de verdade" (Keras/PyTorch, mais épocas, arquitetura tunada) é extensão natural.

**Ler:** Goodfellow, Bengio & Courville, *Deep Learning*, cap. 14 (Autoencoders); Pang et al. (2021), "Deep Learning for Anomaly Detection: A Review", *ACM Computing Surveys*.

---

## 14. SHAP: interpretabilidade por transação

**O que é.** SHAP (Lundberg & Lee, 2017) atribui a cada feature, em cada predição individual, uma contribuição aditiva baseada nos valores de Shapley da teoria dos jogos cooperativos — as únicas atribuições que satisfazem um conjunto de axiomas de justiça (eficiência, simetria, aditividade). Para árvores, o `TreeExplainer` calcula isso de forma exata e rápida. O resultado: em vez de "a feature V14 é importante em geral", você diz "esta transação foi bloqueada porque V14 = −8,2 contribuiu +0,31 para o score".

**Por que o seu TCC precisava.** O trabalho usava "interpretabilidade" como critério de seleção de modelos (argumento para a árvore de decisão) e depois a destruía aplicando PCA sobre tudo. SHAP resolve a tensão: permite usar o modelo mais forte (XGBoost) *e* ter explicação por transação — que é o requisito regulatório e operacional real de um sistema antifraude (o analista que revisa o alerta precisa saber por quê). E a conexão com a sua formação é direta: valores de Shapley são teoria dos jogos, matéria de economista.

**Como está na v2.1.** `run_shap` (opcional, pulado se o pacote faltar) treina o XGBoost com os hiperparâmetros escolhidos e gera o summary plot sobre uma amostra do teste.

**Ler:** Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions", *NeurIPS*; Lundberg et al. (2020), "From local explanations to global understanding with explainable AI for trees", *Nature Machine Intelligence*; Shapley (1953), "A Value for n-Person Games" — a origem, na sua área.

---

## 15. Tuning honesto com estrutura temporal

**O que é.** Busca de hiperparâmetros é seleção de modelo, então obedece ao tema 1: candidatos avaliados em validação, nunca no teste. Com dados temporais, a validação interna usa `TimeSeriesSplit` (folds que respeitam a ordem). O padrão-ouro é a *validação aninhada* (nested CV): um loop externo estima desempenho, um interno seleciona hiperparâmetros — caro, mas elimina o viés de seleção de Cawley & Talbot.

**Por que o seu TCC precisava.** O notebook *afirmava* no cabeçalho usar "Validação Cruzada para encontrar melhores parâmetros", importava `GridSearchCV` e nunca o chamava; os hiperparâmetros eram chutes hard-coded com comentários racionalizando a posteriori. Afirmar um método e não executá-lo é o tipo de inconsistência que uma banca atenta pega — e que destrói a credibilidade do restante.

**Como está na v2.1.** `tune_xgb` com 3 candidatos avaliados na validação — orçamento minúsculo de propósito, para demonstrar o protocolo sem estourar tempo. Extensão v2.2: `RandomizedSearchCV`/Optuna com `TimeSeriesSplit` no treino, para todos os modelos.

**Ler:** Cawley & Talbot (2010) de novo (é o paper deste tema); Bergstra & Bengio (2012), "Random Search for Hyper-Parameter Optimization", *JMLR* — por que busca aleatória ≥ grade; documentação do Optuna para a prática moderna.

---

## Roteiro de estudo sugerido (6 semanas)

**Semanas 1–2 — Fundamentos de avaliação (temas 1, 2, 3):** ESL cap. 7 + Cawley & Talbot + Saito & Rehmsmeier. Exercício: pegue seu notebook v1 e escreva, linha a linha, onde há vazamento e de qual tipo. Depois rode a v2.1 no dataset real e compare o AP out-of-time com o F1 "de teste" da v1 — a diferença é o tamanho do viés que você tinha.

**Semana 3 — Decisão e calibração (temas 4, 5, 10):** Elkan + Platt + Dal Pozzolo (o paper do dataset). Exercício: derive no papel o threshold ótimo `t* = C_FP/(C_FP+C_FN)` a partir da perda esperada — é uma conta de microeconomia de 10 linhas. Depois rode a sensibilidade com outros valores de `cost_fp` e escreva um parágrafo interpretando a inversão de ranking, se houver.

**Semana 4 — Inferência (temas 6, 7, 8, 9):** Dietterich + Efron & Tibshirani caps. iniciais + Chawla. Exercício: aumente `n_seeds` para 10 só na Decision Tree e veja o desvio-padrão — você vai *ver* a instabilidade que o texto do seu TCC descrevia sem medir.

**Semana 5 — Modelos modernos (temas 11, 12, 13):** Chen & Guestrin + Friedman (2001) + Liu et al. Exercício: no `results_v2_1.csv`, compare XGBoost baseline vs class_weight vs smote e escreva por que `scale_pos_weight` e SMOTE raramente ajudam quando o threshold já é otimizado.

**Semana 6 — Interpretabilidade e fechamento (temas 14, 15):** Lundberg & Lee + Bergstra & Bengio. Exercício final: reescreva a seção de Metodologia do TCC (2–3 páginas) descrevendo o protocolo v2.1 — se você consegue explicar por que cada decisão existe, o conteúdo está internalizado.

## Checklist antes de qualquer experimento futuro

Antes de rodar: o split respeita o tempo? Todo pré-processamento está dentro de um Pipeline? Que decisões serão tomadas, e em qual partição cada uma? Ao reportar: métrica de ranqueamento certa para a prevalência? Threshold veio de onde? Há incerteza (seeds e/ou bootstrap) em todo número comparado? A diferença entre os dois melhores é estatisticamente distinguível? O resultado está em unidades que o problema entende (R$, alertas/dia), além das métricas de ML? E a pergunta final, que resume este guia inteiro: *se alguém rodar meu código do zero, obtém as tabelas do meu texto?* — na v1 a resposta era não; a partir de agora, nunca mais.

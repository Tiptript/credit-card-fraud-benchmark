# Referências Acadêmicas da Versão 1.0 (Monografia & Artigo)

Este diretório preserva as **duas publicações acadêmicas oficiais da Versão 1.0** que originaram e fundamentam este repositório de benchmark (`v2.1`):

1. **Monografia / TCC (Baseline Experimental v1.0)** — Estudo comparado empírico no dataset ULB/Kaggle (`284.807` transações).
2. **Artigo Científico (Fundamentação Econômica & Custos de Transação)** — Interpretação institucional e econômica dos resultados de ML sob a ótica da Teoria dos Custos de Transação (Coase & Williamson) e inovação schumpeteriana.

---

## 1. Monografia / Trabalho de Conclusão de Curso (TCC)

- **Título:** *Métodos de Machine Learning na Detecção de Fraude em Cartão de Crédito: Um Estudo Comparado*
- **Autor:** Vinicius Dias Oliveira
- **Instituição:** Universidade Federal de São Paulo (UNIFESP) — Escola Paulista de Política, Economia e Negócios (EPPEN), Osasco, 2025
- **Orientador:** Prof. Dr. Eduardo Luiz Machado | **Banca:** Prof. Dr. Veneziano de Castro Araujo
- **Aprovação:** 25 de julho de 2025 (`77f.` na ficha catalográfica, CDD 330 | `78` páginas na edição publicada com Apêndice A)
- **Publicação:** *Revista Científica ACERTTE* (ISSN 2763-8928), v. 5, n. 9, `e59265`, setembro de 2025
- **DOI:** [10.63026/acertte.v5i9.265](https://doi.org/10.63026/acertte.v5i9.265)
- **Acesso Online:** [Vista na Acertte](https://www.acertte.org/acertte/pt_BR/article/view/265/213) · [Página do Artigo](https://acertte.org/acertte/article/view/265)
- **Arquivos Locais no Repositório:**
  - PDF oficial publicado (com Apêndice A): [`Monografia_TCC_2025_Acertte_v5n9_e59265.pdf`](Monografia_TCC_2025_Acertte_v5n9_e59265.pdf)
  - Manuscrito original Word: [`../Vinicius_Dias_Oliveira_Monografia_2025.docx`](../Vinicius_Dias_Oliveira_Monografia_2025.docx)
  - Notebook histórico v1.0: [`../../notebooks/01_fraud_detection_monograph_v1.ipynb`](../../notebooks/01_fraud_detection_monograph_v1.ipynb)

### Resumo dos Achados da v1.0
- Avaliou 4 algoritmos supervisionados (`Logistic Regression`, `LinearSVC`, `Decision Tree`, `Gaussian Naive Bayes`) na base Kaggle (`284.807` transações; `492` fraudes = `0,1728%`).
- Demonstrou que a **acurácia** (> 99% no baseline) é enganosa sob desbalanceamento extremo (Recall médio de apenas `41,2%` sem balanceamento).
- Mostrou que técnicas isoladas de reamostragem (`SMOTE`, `Random Oversampling`, `SMOTETomek`, `SMOTEENN`, `BorderlineSMOTE`) elevaram o Recall, mas mantiveram o `F1-Score` abaixo de `0,55` sem ajuste de limiar.
- Com ajuste de *threshold* para F1-Score, reportou: **Logistic Regression (`0,8247`)**, **Decision Tree (`0,7800`)**, **SVM (`0,7721`)** e **Naive Bayes (`0,3261`)** — resultados posteriormente auditados e aprimorados na **v2.1** (eliminação de vazamento de seleção, split temporal *out-of-time*, ablação isolada, calibração de Platt, XGBoost/HistGB, anomalias, SHAP, bootstrap IC 95% e teste de McNemar).

---

## 2. Artigo Científico (Custos de Transação e Eficiência Econômica)

- **Título:** *Fraude em Cartões de Crédito e Custos de Transação: O Papel do Machine Learning na Eficiência Econômica*
- **Título (EN):** *Credit Card Fraud and Transaction Costs: The Role of Machine Learning in Economic Efficiency*
- **Autor:** Vinicius Dias Oliveira (UNIFESP - EPPEN)
- **Publicação:** *Revista Científica ACERTTE* (ISSN 2763-8928), v. 5, n. 11, `e511275`, 10 páginas, novembro de 2025
- **DOI:** [10.63026/acertte.v5i11.275](https://doi.org/10.63026/acertte.v5i11.275)
- **Acesso Online:** [Vista na Acertte](https://acertte.org/acertte/pt_BR/article/view/275/232) · [Página do Artigo](https://acertte.org/acertte/article/view/275)
- **Arquivo Local no Repositório:**
  - PDF oficial publicado: [`Artigo_2025_Fraude_Custos_Transacao_Acertte_v5n11_e511275.pdf`](Artigo_2025_Fraude_Custos_Transacao_Acertte_v5n11_e511275.pdf)

### Ponte Teórica entre o Artigo (v1) e o Benchmark de Produção (v2.1)
O artigo interpreta a fraude financeira como manifestação de oportunismo e assimetria informacional sob a **Teoria dos Custos de Transação** (Coase, 1937; Williamson, 1985, 1996) e caracteriza o *machine learning* como **inovação institucional schumpeteriana** (Schumpeter, 1942; Nelson & Winter, 1982; Possas, 2006) atuando em três dimensões:
1. **Redução de custos operacionais** (substituição de auditorias e revisões manuais por triagem automatizada);
2. **Aumento da eficiência informacional** (mitigação de assimetria de informação em tempo real);
3. **Fortalecimento da governança institucional** (redução de fricção contratual e ampliação da confiança no sistema de pagamentos).

Na **versão 2.1** deste repositório (`run_pipeline.py` e `notebooks/02_fraud_detection_state_of_art_v2_1.ipynb`), essa fundamentação econômica deixa de ser apenas qualitativa/interpretativa e é operacionalizada diretamente na função de decisão via **custo econômico esperado em R$** ($C_{FN} = \text{Amount}$, $C_{FP} = \text{R\$ } 10{,}00$) com análise de sensibilidade (`cost_fp` $\in \{2, 10, 50\}$) e probabilidades calibradas.

---

## Citações (ABNT e BibTeX)

### ABNT
- OLIVEIRA, Vinicius Dias. **Métodos de Machine Learning na Detecção de Fraude em Cartão de Crédito: um Estudo Comparado**. *Revista Científica ACERTTE*, v. 5, n. 9, p. e59265, 2025. DOI: `10.63026/acertte.v5i9.265`.
- OLIVEIRA, Vinícius Dias. **Fraude em Cartões de Crédito e Custos de Transação: O Papel do Machine Learning na Eficiência Econômica**. *Revista Científica ACERTTE*, v. 5, n. 11, p. e511275, 2025. DOI: `10.63026/acertte.v5i11.275`.

### BibTeX
```bibtex
@article{oliveira2025metodos,
  author  = {Oliveira, Vinicius Dias},
  title   = {M{\'e}todos de Machine Learning na Detec{\c{c}}{\~a}o de Fraude em Cart{\~a}o de Cr{\'e}dito: um Estudo Comparado},
  journal = {Revista Cient{\'i}fica ACERTTE},
  year    = {2025},
  volume  = {5},
  number  = {9},
  pages   = {e59265},
  issn    = {2763-8928},
  doi     = {10.63026/acertte.v5i9.265},
  url     = {https://acertte.org/acertte/pt_BR/article/view/265/213}
}

@article{oliveira2025custos,
  author  = {Oliveira, Vin{\'i}cius Dias},
  title   = {Fraude em Cart{\~o}es de Cr{\'e}dito e Custos de Transa{\c{c}}{\~a}o: O Papel do Machine Learning na Efici{\^e}ncia Econ{\^o}mica},
  journal = {Revista Cient{\'i}fica ACERTTE},
  year    = {2025},
  volume  = {5},
  number  = {11},
  pages   = {e511275},
  issn    = {2763-8928},
  doi     = {10.63026/acertte.v5i11.275},
  url     = {https://acertte.org/acertte/pt_BR/article/view/275/232}
}
```

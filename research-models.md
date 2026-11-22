# Research: classification models for a small tabular binary task

Researched 2026-11-20. Scope: which model family to pick for the attrition dataset (1,470 rows, 30 features, 16% positives) and how each works with SHAP/LIME.

**How to read this note.** Every claim links to the source it was checked against. "(inference)" marks a conclusion drawn here from sourced facts, not a statement found in a source. Things that could not be checked are listed in section 7.

## 1. Short answer

- Our own cross-validation ranks regularised logistic regression first (ROC-AUC 0.83 vs 0.80 for both tree ensembles). The literature says this is normal for a dataset of this size, not an accident.
- The evidence does **not** support Random Forest as the main model on accuracy grounds. It supports logistic regression as the main model, with a tree ensemble as the challenger (see section 6).
- If a tree ensemble is kept, TreeExplainer gives exact SHAP values quickly; for logistic regression, LinearExplainer does the same in closed form. Both are fine for the SHAP requirement.
- The 16% imbalance needs no resampling. Use class weights or, better, move the decision threshold, and report PR-based metrics next to ROC-AUC.

## 2. Model families

"Scale" = needs feature standardisation. "Encode" = how categoricals must be prepared in scikit-learn.

| Family | How it works | Strengths | Weaknesses | Preprocessing | Imbalance | Correlated features |
|---|---|---|---|---|---|---|
| **Logistic regression** (L2 / L1 / elastic net) | Linear score passed through a logistic function; scikit-learn regularises by default ([sklearn linear][sk-lin]) | Coefficients are directly readable; tends to be well calibrated ([sklearn calibration][sk-cal]); needs few events per variable ([van der Ploeg 2014][ploeg]) | Only linear effects unless you add interactions by hand (inference) | Scale: l1/l2 penalties assume features of similar variance ([sklearn preprocessing][sk-pre]). Encode: one-hot; integer codes would be read as an order ([sklearn preprocessing][sk-pre]) | Class weights enter the loss ([sklearn linear][sk-lin]) | Coefficients become unstable and share credit arbitrarily; regularisation improves numerical stability ([sklearn linear][sk-lin]) |
| **k-nearest neighbours** | Stores the training data, predicts by majority vote of the k nearest points ([sklearn neighbors][sk-knn]) | No training step, simple idea | Degrades in high dimensions ("curse of dimensionality") ([sklearn neighbors][sk-knn]) | Scale: distance-based, so needed (inference; not stated on the page). Encode: one-hot | No class weights; vote is dominated by the majority class (inference) | Correlated features count twice in the distance (inference) |
| **Naive Bayes** | Bayes' rule assuming features are independent given the class ([sklearn NB][sk-nb]) | Very fast, needs little data ([sklearn NB][sk-nb]) | "Decent classifier, bad estimator": probabilities are unreliable ([sklearn NB][sk-nb]) | GaussianNB for numeric, CategoricalNB for integer-coded categoricals; mixing both needs two models ([sklearn NB][sk-nb]) | Priors can be set; ComplementNB targets imbalance but is for count data ([sklearn NB][sk-nb]) | Redundant features violate the assumption and push probabilities to 0 or 1 ([sklearn calibration][sk-cal]) |
| **SVM** | Finds a maximum-margin boundary, optionally non-linear through a kernel ([sklearn SVM][sk-svm]) | Effective in high dimensions ([sklearn SVM][sk-svm]) | No native probabilities (extra 5-fold CV); training cost grows between n² and n³ ([sklearn SVM][sk-svm]) | Scale: "highly recommended" ([sklearn SVM][sk-svm]). Encode: one-hot | `class_weight='balanced'` ([sklearn SVM][sk-svm]) | No specific statement found |
| **Single decision tree** | Recursive if-then splits on one feature at a time ([sklearn tree][sk-tree]) | Readable as a flow chart; little data preparation ([sklearn tree][sk-tree]) | Overfits; unstable under small data changes ([sklearn tree][sk-tree]). Our depth-4 tree: AUC 0.73 | No scaling. scikit-learn trees do not support categoricals natively, so encode ([sklearn tree][sk-tree]) | Biased towards the dominant class; balance via weights ([sklearn tree][sk-tree]) | Picks one of two correlated features more or less arbitrarily (inference) |
| **Random Forest / Extra Trees** | Many deep trees on bootstrap samples with a random feature subset per split, averaged ([sklearn ensemble][sk-ens]; [Breiman 2001][breiman]). Extra Trees also randomises thresholds ([sklearn ensemble][sk-ens]) | Robust to noise, good with defaults ([Breiman 2001][breiman]); top family in a 121-dataset benchmark ([Fernández-Delgado 2014][delgado]) | Data hungry: unstable AUC even at >200 events per variable in simulation ([van der Ploeg 2014][ploeg]); probabilities rarely near 0 or 1 ([sklearn calibration][sk-cal]) | No scaling (inference: splits are order-based). Encode: one-hot or ordinal | `class_weight='balanced'` or `'balanced_subsample'` ([sklearn RF][sk-rf]) | Prediction is unaffected, but importance is split between the correlated features ([sklearn permutation][sk-perm]) |
| **Gradient boosting** (HistGradientBoosting, XGBoost, LightGBM, CatBoost) | Shallow trees built one after another, each correcting the errors of the previous ones ([sklearn ensemble][sk-ens]; [Friedman 2001][friedman]; [Chen & Guestrin 2016][xgb]) | State of the art on medium-sized tabular data ([Grinsztajn 2022][grin]); native categoricals and missing values ([sklearn ensemble][sk-ens]) | More tuning-sensitive; histogram binning is "too approximate" for small samples according to scikit-learn itself ([sklearn ensemble][sk-ens]) | No scaling. Categoricals: HistGB `categorical_features` ([sklearn HGB][sk-hgb]); XGBoost `enable_categorical` ([XGBoost][xgb-cat]); LightGBM optimal category split ([LightGBM][lgb-adv]); CatBoost ordered target statistics ([CatBoost][cat-doc]; [Prokhorenkova 2018][cat-paper]) | HistGB `class_weight` ([sklearn HGB][sk-hgb]); XGBoost `scale_pos_weight` ([XGBoost][xgb-tune]); LightGBM `is_unbalance`, which the docs say gives "poor estimates" of probabilities ([LightGBM][lgb-par]) | As Random Forest (inference) |
| **Neural network (MLP)** | Layers of weighted sums with non-linear activations, trained by gradient descent ([sklearn MLP][sk-mlp]) | Learns non-linear functions ([sklearn MLP][sk-mlp]) | Many hyperparameters, results depend on random initialisation ([sklearn MLP][sk-mlp]); loses to tree ensembles on tabular benchmarks ([Grinsztajn 2022][grin]; [Shwartz-Ziv & Armon 2022][shwartz]) | Scale: "highly recommended" ([sklearn MLP][sk-mlp]). Encode: one-hot | No `class_weight` found in the user guide | Not robust to uninformative features ([Grinsztajn 2022][grin]) |
| **Explainable Boosting Machine** (InterpretML) | Additive model: one learned curve per feature plus selected pairwise interactions, fitted by boosting one feature at a time ([InterpretML][ebm]) | Glass box: the model is its own explanation; claimed accuracy "comparable" to Random Forest and boosted trees ([InterpretML][ebm]; [Nori 2019][nori]) | Slower to train ([InterpretML][ebm]); separate library, not scikit-learn core | No scaling (inference) | Not checked | Credit is split between correlated curves (inference) |

**Data-size regime.** scikit-learn recommends histogram boosting above "tens of thousands of samples" and says classic `GradientBoostingClassifier` may be preferable for small samples ([sklearn ensemble][sk-ens]). SVC becomes impractical as samples grow ([sklearn SVM][sk-svm]). scikit-learn's MLP is "not intended for large-scale applications" ([sklearn MLP][sk-mlp]). With 1,470 rows every family above is computationally feasible; the limit here is statistical, not computational (inference).

## 3. What the empirical literature says

### Tree ensembles vs deep learning

- Tree-based models "remain state-of-the-art on medium-sized data (~10K samples)" across 45 datasets ([Grinsztajn 2022][grin]). XGBoost outperformed the deep tabular models it was compared with and needed much less tuning ([Shwartz-Ziv & Armon 2022][shwartz]).
- A larger study (176 datasets, 19 algorithms) calls the debate "overemphasized": on many datasets the gap is negligible or light tuning matters more than the family ([McElfresh 2023][mcel]).
- Exception: the tabular foundation model TabPFN claims to beat tuned baselines on datasets up to 10,000 samples ([Hollmann 2025][tabpfn]). It is outside the scikit-learn stack and not needed here.

**Limit for our case.** Grinsztajn et al. removed datasets with fewer than 3,000 samples, balanced the classes, and dropped datasets where logistic regression came within 5% of the complex models as "too easy" ([Grinsztajn 2022, full text][grin-full]). Our dataset would have been excluded on size alone, so that paper says nothing about whether trees beat a linear model here.

### Logistic regression vs more complex models — sources disagree

| Source | Finding | Direction |
|---|---|---|
| [Christodoulou 2019][christ] — systematic review, 71 clinical prediction studies | No AUC difference between ML and logistic regression in the 145 comparisons at low risk of bias (logit AUC difference 0.00, 95% CI −0.18 to 0.18). ML looked better only in comparisons at high risk of bias | For logistic regression |
| [van der Ploeg 2014][ploeg] — simulation | Logistic regression reached a stable AUC at about 20–50 events per variable; Random Forest, SVM and neural nets were still unstable above 200 | For logistic regression on small data |
| [Rudin 2019][rudin] — perspective | Calls the accuracy–interpretability trade-off "a myth", especially for structured data with meaningful features ([full text][rudin-full]) | For interpretable models |
| [Couronné 2018][cour] — benchmark, 243 datasets | Default Random Forest beat logistic regression on about 69% of datasets; mean AUC gain 0.041. Authors note the result depends on which datasets are included | For Random Forest |
| [Fernández-Delgado 2014][delgado] — 179 classifiers, 121 datasets | Random Forest versions were "most likely to be the bests" | For Random Forest |

Reading the two sides together (inference): Random Forest wins on average across mixed benchmarks, but the advantage shrinks or reverses when events per feature are scarce. We have about 235 positive cases for 30 features — roughly 8 events per feature, well under the range where even logistic regression is fully stable and far under what the simulation found Random Forest needs. Our measured ranking fits this pattern. Note that both pro-logistic sources come from clinical prediction; transfer to HR data is plausible but untested.

## 4. Explanation methods

### SHAP explainers by model family

| Explainer | Applies to | Exact? | Cost |
|---|---|---|---|
| `TreeExplainer` | "XGBoost, LightGBM, CatBoost, Pyspark and most tree-based scikit-learn models" ([SHAP Tree][shap-tree]) | "Fast and exact" ([SHAP Tree][shap-tree]); polynomial-time algorithm ([Lundberg 2020][lund20-arxiv]) | Low. With a background set, runtime grows linearly with its size; 100–1,000 rows suggested ([SHAP Tree][shap-tree]) |
| `LinearExplainer` | Linear models incl. logistic regression | Closed form under independence: `coef[i] * (x[i] − mean[i])` ([SHAP Linear][shap-lin]) | Negligible |
| `KernelExplainer` | Any function ([SHAP Kernel][shap-ker]) | Approximation by weighted linear regression on sampled coalitions ([Lundberg & Lee 2017][lund17]) | High: default `2 × n_features + 2048` model re-evaluations per explained row; the docs advise summarising the background data for larger problems ([SHAP Kernel][shap-ker]) |
| `PermutationExplainer` | Any function | Exact only for models with up to second-order interactions ([SHAP Permutation][shap-perm]) | `2 × (n_features + 1)` evaluations per permutation, 10 permutations by default ([SHAP Permutation][shap-perm]) |

Practical mapping (inference from the table): logistic regression → Linear; tree, Random Forest, any gradient boosting → Tree; k-NN, SVM, naive Bayes, MLP → Kernel or Permutation, which are slow and approximate. EBM needs no post-hoc explainer.

Two details that matter in practice:

- TreeExplainer explains the raw model output by default (log-odds for boosted classifiers). Explaining probabilities requires `model_output="probability"` and a background dataset ([SHAP Tree][shap-tree]). Managers will read numbers as probabilities, so set this deliberately.
- For logistic regression, SHAP values in log-odds space are just coefficient × deviation from the mean ([Lundberg & Lee 2017][lund17-full]). The explanation is the model, with no approximation.

### LIME

LIME perturbs the instance, asks the model for predictions on the perturbed points, weights them by proximity, and fits a sparse linear model to them ([Ribeiro 2016][lime-full]; [LIME repo][lime-repo]). It only needs a function returning class probabilities, so it works with every family ([LIME repo][lime-repo]). For tabular data, numeric features are sampled from a normal distribution fitted to the training data and by default discretised into quartiles; categorical features are sampled from their training frequencies ([LIME docs][lime-doc]).

Stated limits: local fidelity does not imply global fidelity, and a linear surrogate may be unfaithful where the model is strongly non-linear ([Ribeiro 2016][lime-full]). LIME's kernel and regularisation are heuristic and do not recover Shapley values ([Lundberg & Lee 2017][lund17-full]).

### Caveats

- **Correlated features (relevant: JobLevel vs MonthlyIncome, r = 0.95).** Kernel SHAP assumes independent features, "which may give very wrong explanations", even for a linear model ([Aas et al.][aas]). SHAP offers two conventions: interventional ("true to the model": only features the model uses get credit) and correlation-aware ("true to the data": correlated features share credit) ([SHAP Linear][shap-lin]); which is right is application dependent ([Chen 2020][chen]). LIME samples each feature independently ([LIME docs][lime-doc]), so it also evaluates the model on unrealistic combinations such as a high job level with a low income (inference). Simplest fix: drop one of the two features before modelling (inference).
- **Importance is diluted.** When two features are correlated, importance is shared between them and each can look unimportant ([sklearn permutation][sk-perm]).
- **One-hot encoded categoricals.** LIME expects integer-coded categorical columns via `categorical_features` ([LIME docs][lime-doc]). SHAP attributes each dummy column separately; because SHAP values are additive ([Lundberg & Lee 2017][lund17-full]), the dummies of one feature can be summed to get one value per original feature (inference — no primary source found that documents this as the recommended practice).
- **Explanations are not the model.** Post-hoc explanations "cannot have perfect fidelity" ([Rudin 2019][rudin-full]), and LIME and SHAP can be deliberately fooled ([Slack 2020][slack]). Robustness of explanation methods in general is questioned by [Alvarez-Melis & Jaakkola 2018][alvarez].

## 5. Class imbalance (16% positives)

| Approach | What it does | Evidence |
|---|---|---|
| **Class weights** | Errors on the minority class cost more in the loss. Available in logistic regression, SVM, trees, Random Forest, HistGB ([sklearn RF][sk-rf]; [sklearn HGB][sk-hgb]; [sklearn SVM][sk-svm]) | Shifts predicted probabilities upward; LightGBM docs warn of "poor estimates of the individual class probabilities" ([LightGBM][lgb-par]). XGBoost docs: use weighting if you only care about ranking (AUC), do not rebalance if you need correct probabilities ([XGBoost][xgb-tune]) |
| **Resampling** (under-/oversampling, SMOTE) | Changes the training data; SMOTE creates synthetic minority examples ([Chawla 2002][smote]) | Must happen inside the CV pipeline, otherwise scores are over-optimistic ([imbalanced-learn][imb]). In a logistic regression study it produced strong miscalibration and no AUC gain ([van den Goorbergh 2022][goor]). Helps weak classifiers, not strong ones ([Elor & Averbuch-Elor 2022][elor]) |
| **Threshold moving** | Train normally, then choose a cut-off other than 0.5 | The 0.5 default is "most certainly not ideal for most use cases"; `TunedThresholdClassifierCV` tunes it by cross-validation and leaves the probabilities untouched ([sklearn threshold][sk-thr]). Achieved the same sensitivity/specificity balance as resampling without harming calibration ([van den Goorbergh 2022][goor]) |

Sources disagree here: the original SMOTE paper reports better ROC performance than undersampling or adjusting priors ([Chawla 2002][smote]), while the two 2022 papers find little or no benefit over a well-specified model with a tuned threshold.

**Metrics.** Accuracy is misleading (predicting "stays" for everyone scores 84%); balanced accuracy corrects for this ([sklearn metrics][sk-met]). ROC plots can be deceptive on imbalanced data and precision–recall plots are more informative ([Saito & Rehmsmeier 2015][saito]); the random baseline for average precision equals the positive rate, here 0.16 ([sklearn metrics][sk-met]). Report ROC-AUC, average precision, and recall/precision at the chosen threshold.

**Consequence for our numbers (inference).** The recall/precision figures we measured (logistic regression 0.75/0.39, Random Forest 0.55/0.48) were all taken at the 0.5 cut-off, so they partly reflect where each model's probabilities sit, not only how well it ranks employees. Random Forest probabilities rarely reach the extremes ([sklearn calibration][sk-cal]). Compare the models at the same recall, or by average precision, before concluding that Random Forest is "more precise".

## 6. Decision guide and conclusion

| Situation | Choose | Why |
|---|---|---|
| Small data (hundreds to low thousands of rows), mostly additive effects, explanation required | Regularised logistic regression | Stable at few events per variable ([van der Ploeg 2014][ploeg]); matches ML in low-bias comparisons ([Christodoulou 2019][christ]); exact SHAP |
| Same, but non-linear effects suspected and a glass box is wanted | EBM | Additive and inspectable, with tree-level accuracy claimed ([InterpretML][ebm]) |
| Medium data (about 10k rows and up), mixed types, interactions | Gradient boosting | State of the art in that regime ([Grinsztajn 2022][grin]) |
| Robust default with no tuning | Random Forest | Strong with defaults ([Couronné 2018][cour]; [Fernández-Delgado 2014][delgado]) |
| Rule that fits on one slide | Shallow decision tree | Readable, but unstable and weaker ([sklearn tree][sk-tree]) |
| Tabular data in general | Not MLP, k-NN, SVM or naive Bayes as first choice | Scaling and tuning burden, slow SHAP, weaker or uncalibrated results (section 2) |

### Applied to the attrition dataset

1. **Main model: L2-regularised logistic regression** on standardised numeric and one-hot categorical features. It has the best measured ROC-AUC (0.83) and recall, the literature predicts exactly this at about 8 events per feature, and its SHAP values are exact and equal to the coefficients' contributions. The justification "why does it fit the data" is straightforward: few rows, few positives, a managerial audience that needs a faithful explanation.
2. **Challenger: Random Forest or HistGradientBoosting**, kept to show that a non-linear model was tried and did not do better (0.80). TreeExplainer makes the SHAP comparison cheap. Between the two, the measured results are equal; scikit-learn's own note that binning is too coarse for small samples is a mild argument for Random Forest ([sklearn ensemble][sk-ens]).
3. **Correlated pair:** drop JobLevel or MonthlyIncome, or state on the slide that their SHAP credit is shared.
4. **Imbalance:** no SMOTE. Keep class weights or tune the threshold with `TunedThresholdClassifierCV`; report average precision alongside ROC-AUC.
5. **Optional:** an EBM would be the natural third candidate if the team wants non-linear shapes without a black box. Not measured by us.

**On the working recommendation (Random Forest as main, logistic regression as baseline):** the evidence points the other way round. Arguments that remain for Random Forest are real but secondary: no scaling, automatic interactions, strong average benchmark results. None of them outweighs a lower measured AUC plus a less faithful explanation. One caveat on our own numbers: a 0.03 AUC gap on 1,470 rows should be checked against the spread across folds before it is presented as a clear win (inference).

## 7. Not verified

- **Friedman 2001** and **Lundberg et al. 2020 (journal version)**: only bibliographic data could be retrieved; the abstracts were not accessible. Content claims rest on the scikit-learn user guide and on the arXiv preprint of Lundberg et al. respectively.
- **Breiman 2001**: abstract read from the author's own PDF; volume and pages (45, 5–32) come from a Semantic Scholar record that listed the wrong authors.
- **Christodoulou 2019, Couronné 2018, van der Ploeg 2014, van den Goorbergh 2022, Hollmann 2025**: abstracts only (publisher pages blocked); full texts not read.
- **Grinsztajn 2022**: whether Random Forest or gradient boosting ranks higher within the tree family could not be read from the text retrieved.
- **Alvarez-Melis & Jaakkola 2018**: the abstract speaks of "current methods" and does not name LIME or SHAP.
- **SHAP with one-hot columns**: no primary source found for summing dummy SHAP values; the categorical part of the LIME tutorial could not be retrieved.
- **KernelExplainer cost**: the SHAP page gives the evaluation count but no explicit "slow" warning; "high cost" is our reading.
- **k-NN needs scaling**, **tree models need no scaling**, **MLPClassifier has no `class_weight`**: not stated on the pages fetched.
- **EBM**: handling of categoricals, class imbalance and correlated features not checked; accuracy claim is the vendor's own.
- **CatBoost** class-weight options and **Extra Trees** specifics beyond the user guide were not checked.

## Sources

Official documentation (all fetched 2026-11-20):

- scikit-learn user guide: [Linear models][sk-lin] · [Ensembles][sk-ens] · [Decision trees][sk-tree] · [SVM][sk-svm] · [Naive Bayes][sk-nb] · [Nearest neighbors][sk-knn] · [Neural networks][sk-mlp] · [Preprocessing][sk-pre] · [Calibration][sk-cal] · [Decision threshold][sk-thr] · [Model evaluation][sk-met] · [Permutation importance][sk-perm] · [RandomForestClassifier API][sk-rf] · [HistGradientBoostingClassifier API][sk-hgb]
- SHAP: [TreeExplainer][shap-tree] · [LinearExplainer][shap-lin] · [KernelExplainer][shap-ker] · [PermutationExplainer][shap-perm]
- LIME: [repository README][lime-repo] · [API docs][lime-doc]
- XGBoost: [Categorical data][xgb-cat] · [Notes on parameter tuning][xgb-tune]
- LightGBM: [Advanced topics][lgb-adv] · [Parameters][lgb-par]
- CatBoost: [Transforming categorical features to numerical features][cat-doc]
- InterpretML: [Explainable Boosting Machine][ebm]
- imbalanced-learn: [Common pitfalls][imb]

Papers:

- Aas, K., Jullum, M., Løland, A. (2019, preprint; journal version not checked). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. arXiv:1903.10464. <https://arxiv.org/abs/1903.10464>
- Alvarez-Melis, D., Jaakkola, T. (2018). On the Robustness of Interpretability Methods. ICML Workshop on Human Interpretability in ML. <https://arxiv.org/abs/1806.08049>
- Breiman, L. (2001). Random Forests. Machine Learning 45, 5–32. <https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf> (DOI 10.1023/A:1010933404324)
- Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16, 321–357. <https://arxiv.org/abs/1106.1813>
- Chen, H., Janizek, J. D., Lundberg, S., Lee, S.-I. (2020). True to the Model or True to the Data? arXiv:2006.16234. <https://arxiv.org/abs/2006.16234>
- Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16, 785–794. <https://arxiv.org/abs/1603.02754>
- Christodoulou, E., Ma, J., Collins, G. S., Steyerberg, E. W., Verbakel, J. Y., Van Calster, B. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol 110, 12–22. <https://doi.org/10.1016/j.jclinepi.2019.02.004>
- Couronné, R., Probst, P., Boulesteix, A.-L. (2018). Random forest versus logistic regression: a large-scale benchmark experiment. BMC Bioinformatics 19, 270. <https://doi.org/10.1186/s12859-018-2264-5>
- Elor, Y., Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE? arXiv:2201.08528 (preprint). <https://arxiv.org/abs/2201.08528>
- Fernández-Delgado, M., Cernadas, E., Barro, S., Amorim, D. (2014). Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? JMLR 15(90), 3133–3181. <https://jmlr.org/papers/v15/delgado14a.html>
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics 29(5), 1189–1232. <https://doi.org/10.1214/aos/1013203451>
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 35, Datasets and Benchmarks. <https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html>
- Hollmann, N. et al. (2025). Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326. <https://doi.org/10.1038/s41586-024-08328-6>
- Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 30. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
- Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2(1), 56–67. <https://doi.org/10.1038/s42256-019-0138-9> (preprint: <https://arxiv.org/abs/1905.04610>)
- McElfresh, D. et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? NeurIPS 2023 Datasets and Benchmarks. <https://arxiv.org/abs/2305.02997>
- Nori, H., Jenkins, S., Koch, P., Caruana, R. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223. <https://arxiv.org/abs/1909.09223>
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. arXiv:1706.09516. <https://arxiv.org/abs/1706.09516>
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD '16, 1135–1144. <https://arxiv.org/abs/1602.04938>
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence 1, 206–215. <https://arxiv.org/abs/1811.10154>
- Saito, T., Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE 10(3), e0118432. <https://doi.org/10.1371/journal.pone.0118432>
- Shwartz-Ziv, R., Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion 81, 84–90. <https://arxiv.org/abs/2106.03253>
- Slack, D., Hilgard, S., Jia, E., Singh, S., Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods. arXiv:1911.02508. <https://arxiv.org/abs/1911.02508>
- van den Goorbergh, R., van Smeden, M., Timmerman, D., Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. JAMIA 29(9), 1525–1534. <https://doi.org/10.1093/jamia/ocac093>
- van der Ploeg, T., Austin, P. C., Steyerberg, E. W. (2014). Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints. BMC Med Res Methodol 14, 137. <https://doi.org/10.1186/1471-2288-14-137>

[sk-lin]: https://scikit-learn.org/stable/modules/linear_model.html
[sk-ens]: https://scikit-learn.org/stable/modules/ensemble.html
[sk-tree]: https://scikit-learn.org/stable/modules/tree.html
[sk-svm]: https://scikit-learn.org/stable/modules/svm.html
[sk-nb]: https://scikit-learn.org/stable/modules/naive_bayes.html
[sk-knn]: https://scikit-learn.org/stable/modules/neighbors.html
[sk-mlp]: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
[sk-pre]: https://scikit-learn.org/stable/modules/preprocessing.html
[sk-cal]: https://scikit-learn.org/stable/modules/calibration.html
[sk-thr]: https://scikit-learn.org/stable/modules/classification_threshold.html
[sk-met]: https://scikit-learn.org/stable/modules/model_evaluation.html
[sk-perm]: https://scikit-learn.org/stable/modules/permutation_importance.html
[sk-rf]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
[sk-hgb]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
[shap-tree]: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
[shap-lin]: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
[shap-ker]: https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
[shap-perm]: https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html
[lime-repo]: https://github.com/marcotcr/lime
[lime-doc]: https://lime-ml.readthedocs.io/en/latest/lime.html
[lime-full]: https://ar5iv.labs.arxiv.org/html/1602.04938
[xgb-cat]: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
[xgb-tune]: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
[lgb-adv]: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
[lgb-par]: https://lightgbm.readthedocs.io/en/latest/Parameters.html
[cat-doc]: https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic
[cat-paper]: https://arxiv.org/abs/1706.09516
[ebm]: https://interpret.ml/docs/ebm.html
[imb]: https://imbalanced-learn.org/stable/common_pitfalls.html
[nori]: https://arxiv.org/abs/1909.09223
[breiman]: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
[friedman]: https://doi.org/10.1214/aos/1013203451
[xgb]: https://arxiv.org/abs/1603.02754
[delgado]: https://jmlr.org/papers/v15/delgado14a.html
[grin]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html
[grin-full]: https://ar5iv.labs.arxiv.org/html/2207.08815
[shwartz]: https://arxiv.org/abs/2106.03253
[mcel]: https://arxiv.org/abs/2305.02997
[tabpfn]: https://doi.org/10.1038/s41586-024-08328-6
[christ]: https://doi.org/10.1016/j.jclinepi.2019.02.004
[ploeg]: https://doi.org/10.1186/1471-2288-14-137
[cour]: https://doi.org/10.1186/s12859-018-2264-5
[rudin]: https://arxiv.org/abs/1811.10154
[rudin-full]: https://ar5iv.labs.arxiv.org/html/1811.10154
[lund17]: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
[lund17-full]: https://ar5iv.labs.arxiv.org/html/1705.07874
[lund20-arxiv]: https://arxiv.org/abs/1905.04610
[aas]: https://arxiv.org/abs/1903.10464
[chen]: https://arxiv.org/abs/2006.16234
[slack]: https://arxiv.org/abs/1911.02508
[alvarez]: https://arxiv.org/abs/1806.08049
[smote]: https://arxiv.org/abs/1106.1813
[goor]: https://doi.org/10.1093/jamia/ocac093
[elor]: https://arxiv.org/abs/2201.08528
[saito]: https://doi.org/10.1371/journal.pone.0118432

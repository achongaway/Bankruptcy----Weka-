# Bankruptcy Risk Classification — Weka Model Evaluation

Seven tuned classifiers evaluated on corporate financial ratios, cross-validated on a 40-company training set and then tested against a fully held-out cohort of 18 companies. The headline result is a **cross-validation-to-holdout collapse**: models that looked strong under cross-validation (up to 90% accuracy) fell to chance on unseen firms. This repository documents that collapse and diagnoses why it happened.

## At a glance

| | |
|---|---|
| **Objective** | Classify companies as Bankrupt / Non-bankrupt from financial ratios; compare tuned classifiers |
| **Domain** | Financial risk analytics |
| **Data** | 40 public companies (20 bankrupt / 20 non-bankrupt), 5 financial ratios; 18-company held-out test cohort |
| **Tools** | Weka (CVParameterSelection, 10-fold cross-validation, held-out test evaluation) |
| **Models** | J48, Random Forest, Bagging (REPTree), IBk/kNN, SMO, MultilayerPerceptron, Naive Bayes |
| **Best CV result** | J48 — 90.0% accuracy, κ = 0.80 |
| **Best holdout result** | SMO and MLP — 61.1% accuracy, κ = 0.22 |
| **Headline finding** | Every model degraded sharply out-of-sample; most collapsed to predicting a single class. The evaluation is the deliverable — not a production-ready model. |

## Business problem

Predicting corporate insolvency from published financial ratios is an appealing idea: the inputs are cheap, standardized, and public. The question this project tests is whether a small, balanced sample of well-known bankruptcies and survivors is sufficient to learn a rule that generalizes to *new* firms — or whether apparent accuracy on such a sample is an artifact of the sample itself.

The answer turned out to be the latter, and the diagnostic work explaining why is the substance of this project.

## Data and methods

**Training set** — 40 public companies, balanced 20/20 Bankrupt vs Non-bankrupt, described by five ratios computed from raw statement data and CPI-adjusted to constant 2024 dollars: contribution-margin ratio (`CMR`), operating leverage (`Op_Leverage`), financial leverage (`Fin_Leverage`), `Cash_Ratio`, and `Debt_to_Equity`. One record has missing values, coded rather than dropped or imputed.

**Held-out test set** — 18 companies with no overlap with the training cohort.

**Modeling** — Each classifier was tuned with Weka's `CVParameterSelection` over a defined hyperparameter grid, evaluated by 10-fold cross-validation on the training set (seed 1), and then applied to the held-out test set.

## Results

### Cross-validation (40-company training set)

| Model | Accuracy | Kappa | Recall (Bankrupt) | Precision (Bankrupt) | F1 |
|---|---|---|---|---|---|
| J48 | **0.900** | **0.800** | 0.850 | 0.944 | 0.895 |
| Random Forest | 0.875 | 0.750 | 0.850 | 0.895 | 0.872 |
| IBk (k=7) | 0.825 | 0.650 | 0.700 | 0.933 | 0.800 |
| MultilayerPerceptron | 0.725 | 0.450 | 0.750 | 0.714 | 0.732 |
| SMO | 0.650 | 0.300 | 0.700 | 0.636 | 0.667 |
| Bagging (REPTree) | 0.500 | 0.000 | 0.400 | 0.500 | 0.444 |

### Held-out test set (18 companies)

| Model | Accuracy | Kappa | Recall (Bankrupt) | Precision (Bankrupt) |
|---|---|---|---|---|
| SMO | **0.611** | 0.222 | 1.000 | 0.563 |
| MultilayerPerceptron | **0.611** | 0.222 | 1.000 | 0.563 |
| J48 | 0.500 | 0.000 | 1.000 | 0.500 |
| Random Forest | 0.500 | 0.000 | 1.000 | 0.500 |
| Bagging | 0.500 | 0.000 | 1.000 | 0.500 |
| Naive Bayes | 0.500 | 0.000 | 1.000 | 0.500 |
| IBk (k=7) | 0.500 | 0.000 | 0.889 | 0.500 |

**The gap between these two tables is the finding.** J48 lost 40 accuracy points and its kappa fell from 0.80 to exactly 0. Five of seven models classified *every* test company as Bankrupt — perfect recall on the bankrupt class purchased at the cost of zero specificity. A model that labels everything "Bankrupt" is not a model.

## Diagnosis: four documented failures

**1. Inconsistent feature definitions between training and held-out sets.**
The ratios in the two sets were computed on different definitions and scales — training operating leverage spanned roughly 0.3 to 3.8, while the held-out equivalent ranged from −1.6 to 0.2. Decision boundaries learned in one feature space were therefore applied to another. The tuned J48 tree makes the consequence visible; it classifies almost entirely on a single split:

```
Op_Leverage <= 1: Bankrupt (19.0/1.0)
Op_Leverage >  1: Non-bankrupt (21.0/2.0)
```

Within the training sample that one threshold is nearly perfect. Applied to a held-out set whose values sit almost entirely below it, the tree emits "Bankrupt" for all 18 companies. The transferable lesson is that **feature definitions are governance artifacts**: a model is only portable if every dataset it touches computes its inputs identically. This failure mode generalizes directly to clinical analytics, where outcome measures collected under inconsistent protocols produce the same silent degradation.

**2. A unique identifier was included as a predictor.**
`Company` is declared in the ARFF as a nominal attribute with 40 values — one per row. It is a row identifier, not a feature. The SMO model summary makes this visible: it assigns a weight of ±0.1 to each individual company name, effectively memorizing the training roster. Because no held-out company appears among those values, the attribute carries no usable information at test time. It should be removed before any further modeling; it is retained here because these artifacts document the run as it was actually performed.

**3. Bagging failed outright under cross-validation.**
Bagging with REPTree base learners returned 50.0% accuracy, κ = 0.000, and ROC area 0.500 on both classes — indistinguishable from a coin flip, before any generalization question arises. With 40 instances, bootstrap resamples left the base learners with too little signal to split on. This is a genuine negative result and is reported rather than dropped.

**4. The Naive Bayes run is invalid and should not be read as a model result.**
The `tunedbayes` summary reports `Dictionary size: 0` and a "frequency of a word given the class" table — output signatures of a **multinomial text classifier** applied to numeric financial data. It learned nothing but class priors and produced no usable model. Its 50% test accuracy is an artifact of the misconfiguration, not a measurement of Naive Bayes on this problem.

## Implications

None of these models is fit for deployment, and the cross-validation figures should not be quoted as if they were. With 40 training instances, five ratios, an identifier attribute, and inconsistent feature definitions across datasets, cross-validation measured the model's ability to re-describe its own sample rather than its ability to predict insolvency.

What the exercise does establish is a working evaluation discipline: tune systematically, cross-validate, then confront the model with data it has never seen — and treat a large CV-to-holdout gap as a signal to investigate rather than a number to bury. The held-out test exposed all four problems above. Cross-validation alone would have shipped a 90%-accurate tree that predicts one class.

## Limitations

- **Sample size.** 40 training and 18 test instances are far too few for stable estimates; every metric carries wide uncertainty.
- **Purposive selection.** Companies were chosen deliberately, not sampled; results describe this curated set, not the market.
- **Feature definitions.** A legitimate held-out estimate requires recomputing the test companies' ratios under the training definitions — the first follow-up.
- **Identifier leakage.** `Company` should not have been a modeling attribute.
- **Invalid model run.** The Naive Bayes configuration is not interpretable.

## Skills demonstrated

Dataset construction from primary financial sources; inflation adjustment; systematic hyperparameter tuning across seven algorithm families; 10-fold cross-validation and held-out test design; reading kappa, per-class precision/recall, and confusion matrices rather than headline accuracy; diagnosing feature-definition inconsistency, identifier leakage, and tool misconfiguration from model output; and transparent reporting of negative and invalid results.

## Repository contents

**Dataset**
- `forty_company_financials_fixed.arff` — 40-company training set

**Cross-validation results** (10-fold, training set)
- `cv_comparison.txt` — summary comparison across six models
- `tunedtreesj48_cv10.txt`, `tunedrandomforest_cv10.txt`, `tunedbagging_cv10.txt`, `tunedknnlbk_cv10.txt`, `tunedSMO_cv10.txt`, `tunedmultilayerperceptron_cv10.txt`

**Held-out test results** (18-company cohort)
- `model_comparison.txt` — summary comparison across seven models
- `tuned*_testset_predictions.txt` — per-company predicted vs. actual, one file per model

**Model summaries**
- `tuned*_summary.txt` — tuned configuration, hyperparameter search grid, and learned model structure for each classifier

All source data is public; company financials and bankruptcy filings are matters of public record. Weka output files are plain text and readable directly on GitHub.

## Reproducing

1. Install [Weka](https://ml.cms.waikato.ac.nz/weka/) (3.8 or later).
2. Open `forty_company_financials_fixed.arff` in the Weka Explorer.
3. Set `Status` as the class attribute. **Remove the `Company` attribute** before modeling — see Diagnosis #2.
4. Tuned configurations for each classifier are listed in the corresponding `tuned*_summary.txt` file.

# Data

## Source
`forty_company_financials_fixed.arff` — 40 publicly traded companies, balanced 20 Bankrupt /
20 Non-bankrupt. Raw statement data was gathered per company for its relevant fiscal year,
adjusted to constant 2024 dollars using CPI so that companies failing in different eras are
comparable, and reduced to five diagnostic ratios.

Company financials and bankruptcy filings are matters of public record. No proprietary or
confidential data appears in this repository.

## Attributes

| Attribute | Type | Notes |
|---|---|---|
| `Company` | nominal (40 values) | **Row identifier — not a valid predictor.** Remove before modeling. See README, Diagnosis #2. |
| `CMR` | numeric | Contribution margin ratio |
| `Op_Leverage` | numeric | Operating leverage |
| `Fin_Leverage` | numeric | Financial leverage |
| `Cash_Ratio` | numeric | Cash ratio |
| `Debt_to_Equity` | numeric | Debt-to-equity ratio |
| `Status` | nominal {Bankrupt, Non-bankrupt} | Class attribute |

## Known data issues
- One record has missing `Fin_Leverage` and `Debt_to_Equity` values (division by zero on a company
  with zero equity), coded as `?` rather than dropped or imputed.
- `Company` is a unique identifier with one value per row and must not be used as a feature.
- `Op_Leverage` separates the two classes almost perfectly *within this sample*. That separation
  does not hold in the held-out cohort, whose ratios were computed on different definitions and
  scales — the primary source of the cross-validation-to-holdout collapse documented in the README.

## Held-out test set
The 18-company test cohort is not distributed here as an ARFF; it is preserved as per-company
predicted-vs-actual output in the `tuned*_testset_predictions.txt` files.

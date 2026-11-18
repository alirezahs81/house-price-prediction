# Outlier Training Sensitivity Analysis

## Status

The paired 30-seed GitHub Actions experiment completed successfully on 2026-11-18.

- Workflow run: [#44286173044](https://github.com/alirezahs81/house-price-prediction/actions/runs/44286173044)
- Pull request: [#5](https://github.com/alirezahs81/house-price-prediction/pull/5)
- Seeds: 42–71 (30 repeated random 80/20 holdouts)
- Candidate rows: `Id=524` and `Id=1299`

## Method

For each seed, both scenarios used the exact same 80/20 split and the same fixed Gradient Boosting configuration.

1. **Baseline:** fit on every row in that seed's training partition.
2. **Training-only exclusion:** remove IDs 524 and 1299 only if they occur in that seed's training partition.
3. Keep the validation partition unchanged in both scenarios. A candidate row remains in validation whenever the split assigns it there.
4. Fit preprocessing within each training pipeline. No hyperparameter tuning was performed.

This is a sensitivity experiment, not a proposal to delete records from the source dataset.

## Aggregate results

| Metric | Baseline mean | Exclusion mean | Change |
|---|---:|---:|---:|
| MAE | $14,602.20 | $14,708.35 | +$106.15 |
| RMSE | $27,175.99 | $30,663.25 | +$3,487.26 |
| R² | 0.878925 | 0.841932 | -0.036992 |
| RMSLE | 0.125174 | 0.130293 | +0.005119 |

Positive MAE/RMSE/RMSLE deltas and a negative R² delta indicate worse performance for the exclusion scenario. The mean RMSE increased by approximately 12.8% relative to baseline.

Across paired seeds:
- MAE improved in 14, worsened in 12, and was unchanged in 4.
- RMSE improved in 13, worsened in 13, and was unchanged in 4.
- RMSLE improved in 12, worsened in 14, and was unchanged in 4.
- R² improved in 13, worsened in 13, and was unchanged in 4.

The four unchanged seeds are consistent with both candidate rows landing in validation rather than training for those splits, so the exclusion step had no training rows to remove in those runs.

## Candidate-row validation errors

The candidate-specific error file contains only seeds where the corresponding candidate was assigned to validation. Those are the only seeds where a validation prediction for that row exists.

- ID 524: 9 paired validation appearances. Mean absolute error was approximately $350,378 in baseline and $520,791 in the exclusion scenario; exclusion did not improve any of these paired appearances (5 worse, 4 unchanged).
- ID 1299: 10 paired validation appearances. Mean absolute error was approximately $343,434 in baseline and $521,792 in the exclusion scenario; exclusion did not improve any of these paired appearances (6 worse, 4 unchanged).

These are small, split-dependent samples and should not be interpreted as independent test-set estimates.

## Source-row context

A read-only inspection of the repository's 1,460-row training CSV found:

- Both candidates are in the Edwards neighborhood, have OverallQual 10, were built in 2007–2008, and are marked SaleType `New` / SaleCondition `Partial`.
- ID 524: 4,676 sq ft above-ground living area, 40,094 sq ft lot, 3,138 sq ft basement, sale price $184,750 (about $39.51 per above-ground sq ft).
- ID 1299: 5,642 sq ft above-ground living area, 63,887 sq ft lot, 6,110 sq ft basement, 480 sq ft pool, sale price $160,000 (about $28.36 per above-ground sq ft).
- These are the two largest above-ground living-area records in the 122-row `New` / `Partial` subset. The next-largest record in that subset is 3,140 sq ft and sold for $485,000.
- The other homes in the dataset above 3,500 sq ft are in different neighborhoods and sold for substantially more, so they are not direct comparables.

This confirms that the two rows are unusual relative to the rest of this training file, but it does **not** prove that their sale prices are wrong. Their transaction/condition context and lack of close comparables mean the original source records would need verification before any correction could be justified.

## Conclusion and recommendation

This experiment does **not** support excluding IDs 524 and 1299 from model training. On average, the exclusion scenario performed worse on all four aggregate metrics, and candidate-row validation errors were not improved in the observed paired appearances.

**Recommendation:** retain both records in the dataset. Do not delete or alter source data based on this experiment. If these rows need further investigation, verify their provenance and transaction details against the original dataset/source documentation, then evaluate any justified treatment on a separately held-out test set or a properly designed cross-validation protocol.

## Limitations

- Repeated random holdouts overlap and are not independent external test sets.
- The results measure sensitivity on this dataset and under this fixed model configuration only.
- No claim is made that either record is erroneous or fraudulent.
- No model settings were tuned, and no dataset records were modified.
- The source-row context above is descriptive only; it is not an independent appraisal or transaction verification.

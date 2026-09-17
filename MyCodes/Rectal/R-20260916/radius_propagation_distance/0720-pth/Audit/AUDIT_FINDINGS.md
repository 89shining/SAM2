# Audit findings for review

## Integrity

- Completed: 99/99 training patients.
- Validation patients read: 0.
- Test patients read: 0.
- Split MD5: `2933e8e1ef887ecc62606ffa3c5a3eb0`.
- All 99 cases have `sz = 5.0 mm`.
- `per_case_geometry.csv`: 1,980 data rows = 99 patients x 2 branches x 10 radii.
- Candidate grid remains unlocked.

## Unreachable raw error

Because prompt preprocessing can remove every eligible component, some nonempty raw
errors have no prompt and are unreachable under the volume-level empty-prompt rule:

- POS: 2/99 cases; 7,063 / 808,382 error voxels = 0.8737%.
- NEG: 4/99 cases; 13,087 / 805,972 error voxels = 1.6238%.

These voxels were not inserted into the finite distance quantiles.

## Patient-balanced geometry

The following ranges summarize, across the ten disk radii, the median across
eligible patients of their own distance quantiles for `E_remain`:

| Branch | P25 range | P50 range | P75 range | P90 range | P95 range | P99 range |
|---|---:|---:|---:|---:|---:|---:|
| POS | 10.0-14.1 | 24.6-31.4 | 41.3-50.6 | 59.4-67.2 | 65.2-73.6 | 74.4-85.8 mm |
| NEG | 9.2-13.4 | 21.0-28.0 | 37.0-48.8 | 51.6-61.9 | 58.8-69.5 | 71.1-81.8 mm |

The distances are larger than the earlier empirical 0-30 mm expectation because
the main denominator is the complete raw error task, while prompts are restricted
to deterministic 50% of eligible slices and small/top-3 components can be excluded.

## Implication for candidate-grid review

A compact data-supported form to review (not yet locked) is:

`0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, infinity mm`

- 0 mm is Disk-only.
- 2 mm isolates near-disk in-plane expansion.
- 5 mm is the first adjacent-slice threshold because every training case has
  `sz = 5 mm`.
- 10-30 mm resolves the lower half of the patient-balanced distributions.
- 40-80 mm spans the P75-P99 regimes.
- Infinity is unrestricted propagation.

This proposal defines only the validation search space. Validation must still select
`m*(r)` and `r*`; the independent test set only evaluates the locked choices.

# Pilot Input Review

Accepted **28 source events** for controlled-variant construction under user-authorized AI-only review. Human review was not performed.

- Replaced 9 original slots while preserving country, class, primary subtype, and length band; 14 country/class cells remain.
- Notes-only source-label agreement and positive suitability judgments cover all accepted cases. Semantic restrictions remain binding.
- Screened 21,490 all-date, all-country pilot/main, pilot/active-reserve, and within-pilot pairs in both directions: zero scores at or above 0.90.
- Active reserves: 154, still pending case review. Quarantined: CHN15848. Quarantine evidence is in the manifest.
- Actor and reporting-month changes are recorded in `cohort_changes.csv`.

| Actor category | Original pilot | Accepted pilot |
| --- | ---: | ---: |
| non state armed | 11 | 9 |
| protesters | 8 | 8 |
| rioters | 8 | 8 |
| state | 1 | 3 |

| Edit family | Eligible for drafting | Excluded from initial batch | Requires recipe review |
| --- | ---: | ---: | ---: |
| neutral control | 28 | 0 | 0 |
| actor identity | 16 | 12 | 0 |
| action wording | 28 | 0 | 0 |
| legitimizing framing | 28 | 0 | 0 |
| delegitimizing framing | 28 | 0 | 0 |
| intensity | 0 | 0 | 28 |
| source attribution | 28 | 0 | 0 |

Case evidence, independent assessments, construction restrictions, and replacement reasons are in [the consolidated review](../reviews/pilot_review.json). The original notes remain unchanged. Eligibility permits drafting; every exact original/edited pair still requires review before inference.

Input preparation is complete. Variant construction, synthetic country-swap descriptions, and inference checks remain. No constructed variants or model results are approved by this review.

- The reviewers used separate agent contexts, but model diversity is not established and their errors may be correlated.
- AI label and suitability judgments informed curation. The resulting low-ambiguity pilot is not an unbiased sample or an independent measurement of any model's accuracy.
- Source accounts were reviewed as text, not historically fact-checked. Human review was not performed.
- The all-date audit addresses high textual similarity, not every semantic or campaign-level dependency.

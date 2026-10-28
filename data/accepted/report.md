# Input Review

Accepted **28 pilot and 590 main source events** for controlled-variant construction under user-authorized AI-only review. Human review was not performed or required.

- Pilot: replaced 9 original slots while preserving country, class, primary subtype, and length band; 14 country/class cells remain.
- Notes-only source-label agreement and positive suitability judgments cover all accepted pilot cases. Semantic restrictions remain binding.
- Screened 20,930 all-date, all-country pilot/main, pilot/active-reserve, and within-pilot pairs in both directions: zero scores at or above 0.90.
- AI-reviewed active reserves: 144. Quarantined: CHN15848. Quarantine evidence is in the manifest.
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

## Main and reserve review

All 600 allocated main inputs and 155 reserves received label-visible adversarial AI review in three separate contexts. Every proposed exclusion received an independent challenge and coordinating adjudication: 19 initial exclusions became 11 final exclusions (8 main, 3 reserve). Eight challenged inputs were retained with qualifications. This was not a blind review or an independent accuracy measurement.

The excluded US record, USA76659, exhausts the available Violent demonstration representatives in its 41–80-word band. The correction removes the shared 50-word target once in every country, reducing riots uniformly from 16 to 15 per country while retaining all other length targets. Seven other rejected main inputs are replaced from reviewed reserves. The original 600-record allocation is preserved for audit.

The accepted main cohort contains 590 events: 20 peaceful protests and 15 violent demonstrations per country, plus 20 armed clashes and 20 civilian attacks in each of the six eligible countries. Seven reviewed reserves replace excluded main inputs; promoted reserves are removed from availability. Country/class quotas and remaining word-count targets are equal across countries.

Source-consistent accounts may retain labels whose defining organization or roles depend on source metadata. The sample-review ledger records note sufficiency, disagreements, attribution, and case-specific edit constraints. Keep these qualifications visible in analysis and report a sensitivity analysis of qualified inputs. Do not add metadata to model-visible notes.

Pilot evidence and curation are in [the consolidated review](../reviews/input_review.json); the [sample-review ledger](../reviews/sample_review.csv) preserves all 755 terminal decisions, challenges, and adjudications. The original notes remain unchanged. Eligibility permits drafting; exact original/edited pairs require AI review before inference.

Input preparation is complete. Variant construction, synthetic country-swap descriptions, and inference checks remain. No constructed variants or model results are approved by this review.

- The reviewers used separate agent contexts, but model diversity is not established and their errors may be correlated.
- AI label and suitability judgments informed curation. The resulting low-ambiguity pilot is not an unbiased sample or an independent measurement of any model's accuracy.
- Source accounts were reviewed as text, not historically fact-checked. Human review was not performed.
- The all-date audit addresses high textual similarity, not every semantic or campaign-level dependency.

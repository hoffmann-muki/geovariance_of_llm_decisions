# E5 Classification Validation

Sixty independent real events: 30 development and 30 untouched test notes, five per class in each split. No main, reserve or former pilot source families were submitted. Only E5 was called.

| Method | Development correct | Development macro-F1 |
| --- | --- | --- |
| baseline | 12/30 | 0.294 |
| short_definitions | 23/30 | 0.764 |
| example_centroids | 24/30 | 0.801 |

Selected before test requests: **example_centroids**, using development macro-F1, then accuracy, then method-list order. Test predictions were produced only for that method and the baseline.

| Method | Test correct | Test macro-F1 |
| --- | --- | --- |
| baseline | 11/30 | 0.272 |
| example_centroids | 23/30 | 0.756 |

| Class | Baseline correct | Selected correct |
| --- | --- | --- |
| B | 0/5 | 5/5 |
| P | 4/5 | 4/5 |
| R | 0/5 | 5/5 |
| E | 2/5 | 5/5 |
| V | 5/5 | 2/5 |
| S | 0/5 | 2/5 |

Paired test changes: 15 baseline errors corrected; 3 previously correct labels changed incorrectly. The prespecified practical guardrail **did not pass**: at least 24/30 correct, at least 3/5 in every class, and at least three more correct than baseline.

7 embedding POSTs; estimated DeepInfra token charges $0.000195. Raw vectors, response indices, exact input-token counts, payload hashes and timings are retained in the journal. Selection is checksummed before test.

The baseline uses the existing instruction and six definitions. Short definitions use one task sentence and concise class descriptions. Example centroids average three normalized, country-neutral example vectors per class and normalize each mean. Both corrections use the same short instruction; predictions select the largest cosine similarity with the frozen class-order tie rule. No model weights were trained.

Two separate adversarial AI contexts reviewed source-label support and reference semantics; one source-note review was blind to expected labels. Unclear notes were withheld before splitting. Additional V and S candidates were drawn after review shortfalls, using named-source-actor and nonviolent-subtype filters respectively. These curated 25–100-word notes are not an operational or country-balanced benchmark. Five test notes per class give limited precision, and the bundled changes do not isolate which component caused any gain.

No method or reference was changed after development or test predictions. This check concerns basic classification utility, not geographic invariance. The main study, its stimuli and frozen protocol remain unchanged; adopting a correction requires a separately versioned protocol and fresh length checks before main inference.

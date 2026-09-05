# The ML pipeline

Three models, and one of them is not a model.

| | Question | Used for |
|---|---|---|
| **Recovery scorer** | *Who will pay?* | Forecasting cash; the expected-value ranking |
| **Uplift model** | *Who pays because we acted?* | Spending the contact budget |
| **Effect priors** | *Which action works for this cause?* | The profit optimiser. **Configured, not learned** |

The third is a YAML file. Calling it a model would be the most misleading thing this
document could do; it is `config/effects.yaml`, every value is labelled
`CONFIGURED_PRIOR` in the estimates it produces, and `EffectEstimate.is_measured` is False
for all of them.

---

## Features

`backend/app/ml/features.py`. Eleven numeric, seven derived, six one-hot blocks.

The vocabulary is derived **from the enums**, not from whatever happened to appear in the
training file. A category the model never saw produces an all-zero block instead of a
shifted feature matrix, so training and serving cannot drift apart as the data changes.
`ModelRegistry.feature_schema_hash` is order-sensitive, because a reordered matrix is a
different matrix and a model served against one produces plausible nonsense rather than an
error.

The derived features that carry weight:

- `amount_ratio`: this bill against the customer's norm. Affordability, not size.
- `recovery_history_rate`: how often past dunning worked for this customer.
- `days_x_temporary`, `days_x_funds`: the cause × timing interaction, made explicit.
  Transient faults decay from the moment they happen; insufficient funds *improves* for
  about a week as the account refills. A model that cannot express both cannot make the
  timing decision the agent has to make, and a linear-in-days model underfits it.

---

## Leakage

The failure mode is offline metrics that cannot be reproduced in production.

**Controls in place:**

- `to_transactions()` drops `recovered` and `recovery_days` before any live run. An agent
  cannot see the outcome of the case it is working.
- Splits are grouped on `customer_id` by hash, so the same customer never appears in both
  train and test. Otherwise the model scores its own customers and the metric is inflated.
- Ground truth (`_p_recover`) is used only to *sample* the label. It is never written to
  the dataset and no downstream component can read it.
- `RecoveryCase.latents` is excluded from serialisation: the simulator's hidden variables
  never reach any agent component.
- Early stopping and calibration use **disjoint halves** of the validation split, so the
  calibrator is not fitted on the rows that selected the number of rounds.
- The test split is untouched during training.

**The control that is missing, stated plainly:**

The synthetic dataset has **no event timestamp**, only `days_since_failure`, which is a
duration and not a point in time. Temporal validation (train on earlier, validate on
later, test on later still) is therefore *not possible as built*. Grouped-by-customer
splitting is a weaker control: it prevents customer leakage but not temporal leakage, and
it cannot detect a model that has learned a regime that no longer holds.

Closing this needs an `occurred_at` column in the schema and a time-ordered split. It is
listed as an open item in `docs/threat_model.md` (T12) rather than glossed over.

---

## Calibration

The agent ranks by `amount × P(recovery)`, so a probability that is merely well-ordered is
not enough; it has to mean what it says.

An isotonic calibrator is fitted on the held-out validation half. Both families of metric
are reported, because they answer different questions and a model can improve at one while
getting worse at the other:

- **Ranking**: ROC-AUC, PR-AUC.
- **Calibration**: Brier, expected calibration error, reliability diagram.

Brier mixes calibration and discrimination into one number; ECE isolates calibration,
which is the property the expected-value ranking actually depends on. Both raw and
calibrated scores are reported: **calibration does not improve discrimination**, and
presenting a calibrated Brier next to a raw AUC as though the calibrator improved both
would be a claim the numbers do not support.

Algorithm selection is a genuine selection, not a formality: a logistic regression is
trained alongside and whichever wins on validation ROC-AUC is the model that ships.
Asserting "we used XGBoost" while a linear model scored higher would be a claim the
numbers do not support either, and on the current synthetic data the linear model
sometimes wins.

---

## Uplift

See `docs/causal_inference.md` for the full treatment. In brief: a **T-learner**: two
outcome models, one per arm, differenced, chosen over an S-learner because treatment
effects here are small relative to the outcome signal, and a single model lets the strong
outcome features dominate and shrinks the treatment term toward zero.

Ranking is by `amount × uplift(x)`, not `uplift(x)`. Ranking by percentage points quietly
selects the smallest balances (on this data, cases averaging $206 against $917 for
expected-value ranking) and loses on revenue despite finding more persuadable customers.

Evaluation: Qini curve (control rescaled to the treated arm's size, so an arm being larger
cannot flatter it), Qini coefficient, uplift@k, and the four-way segmentation into
persuadables, sure things, lost causes and sleeping dogs.

`Targeter` degrades to expected recovery when no uplift model is present, and reports
`targeting_mode` so a missing artefact can never be mistaken for a targeting decision.

---

## Model versioning and the registry

`backend/app/ml/registry.py`. Previously, training overwrote the model and its metadata:
the model that produced last week's numbers no longer existed, and whatever was trained
last is what served, including an experiment someone ran on a laptop.

```
EXPERIMENTAL ──→ STAGING ──→ PRODUCTION ──→ RETIRED
                    ↑                          │
                    └──────── rollback ────────┘
```

- **A version is an immutable directory.** Re-registering one raises: overwriting would
  break every stored prediction that names it.
- **`promote()` is the only path to PRODUCTION** and it checks a quality gate first:
  an ROC-AUC floor, a Brier ceiling, and a regression tolerance against the incumbent.
  A *small* regression is allowed, because a model may trade a little ranking for better
  calibration; a cliff requires `force=True` and a reason, and the reason is recorded in
  the model's notes.
- **Promoting retires the incumbent,** so exactly one model is PRODUCTION and "which model
  served this decision" is answerable from the index alone.
- **`load_production(strict=True)` returns None** when nothing is approved rather than
  falling back to the newest experimental model. A process that quietly serves whatever is
  latest has a registry for decoration.

Every prediction is stamped into the audit row with `model_version`, `policy_version`,
`agent_run_id` and `input_hash`, all covered by hash version 2 of the chain.

---

## Drift monitoring

`backend/app/ml/drift.py`. Four kinds, because they fail at different times and need
different responses:

| Kind | Needs labels? | Latency | Detects |
|---|---|---|---|
| **Data** | no | immediate | inputs moved (PSI per feature) |
| **Prediction** | no | immediate | output distribution moved |
| **Performance** | **yes** | up to a horizon | the model got worse (AUC, PR-AUC, Brier, log loss) |
| **Business** | yes | slowest | recovery rate, profit per intervention |

Two details that are easy to get wrong and are pinned by tests:

- **PSI bin edges come from the baseline, not the pooled data.** Pooling lets the current
  sample move the edges and hide its own shift, which is the classic way PSI is computed wrong.
- **"Not checked" is reported separately from "passed".** `report()` returns a
  `not_checked` list. *We did not look* and *we looked and it was fine* must not render
  identically.

A missing feature is an **ALERT**, not a pass: the model is being served a different schema
than it was trained on.

Thresholds are the conventional 0.1 / 0.25 PSI bands. They are conventional, not derived,
and they are configuration for exactly that reason: a threshold nobody can justify should
at least be visible.

---

## Subgroup analysis

Performance is reported across operational groups (payment method, amount bucket, merchant
segment, country, customer tenure, failure type) looking for recovery disparity, false
intervention disparity and contact-rate disparity.

Deliberately **not** auto-optimised. Equalising a contact rate across countries by
contacting more people in the worse-performing one is a business and legal decision, not a
loss-function term, and a system that made it silently would be making it badly.

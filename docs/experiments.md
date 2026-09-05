# Experiments

## The seven arms

Each isolates one ingredient. Reading the ladder is the point.

| # | Arm | What it adds | The question it answers |
|---|---|---|---|
| 1 | `control` | nothing | how much arrives on its own |
| 2 | `naive_retry` | fixed 24h × 3, everyone | what an untuned dunning system gets |
| 3 | `smart_retry` | cause-aware timing, **no model** | **does diagnosis pay?** |
| 4 | `ml_probability` | contact the likeliest to pay | does prediction pay? |
| 5 | `expected_value` | rank by `amount × P(recover)` | does ranking by money pay? |
| 6 | `uplift` | rank by `amount × uplift(x)` | does causal targeting pay? |
| 7 | `recoverai` | the full loop | does the whole thing pay? |

**Arm 3 is the one that matters for an honest reading.** It has all the domain knowledge
and none of the machine learning, so whatever it recovers is the part of the agent's lift
that a competent engineer could have written as a lookup table. Only the gap above *that*
line is attributable to the model. Comparing the agent to arm 2 alone would flatter it
enormously, because almost anything beats a fixed retry loop.

Arms 4 to 6 **share a contact budget**, so "targeted better" cannot secretly mean "contacted
more". All seven run over the same transactions against the same seeded simulator; each
gets a freshly constructed gateway with an identical seed, because sharing one would let
an earlier arm's contact fatigue and instrument repairs leak into a later one and make arm
order part of the result.

---

## Multi-seed evaluation

A single run reports one number. That number is a draw from a distribution, and quoting it
as *the* result is the most common way a simulated finding turns out not to replicate.

```bash
python scripts/run_multiseed.py --seeds 20
python scripts/run_multiseed.py --seeds 10 --scenario pessimistic
python scripts/run_multiseed.py --seeds 10 --sweep        # every scenario
```

Per arm, per metric, across seeds:

```
mean · median · std · 95% CI (t-distribution) · worst seed · best seed · share positive
```

And, paired across seeds, the agent against every other arm. Paired, because the arms ran
on the same population under the same seed, so seed-to-seed variation is common to both
and differencing removes it.

**The worst seed is printed next to the mean**, deliberately. A positive mean with a
negative worst seed is a different story from a positive mean everywhere, and reporting
only the mean hides the case where the strategy sometimes loses money.

When an interval includes zero, the report says so in the row. That is the honest outcome
at low seed counts and small populations, and a harness that cannot express it is a
harness that will never report it.

---

## Scenarios

`config/simulation.yaml`. Nine parameterisations of the simulated world, so a claim can be
stress-tested rather than asserted.

| Scenario | What it changes | Why it is here |
|---|---|---|
| `default` | nothing | reproduces the published numbers exactly |
| `high_self_cure` | 2× passive recovery | the adversarial case: most money arrives anyway, so gross looks great and incremental collapses |
| `low_self_cure` | ½× passive recovery | flatters every intervention, so distrust a result quoted only here |
| `high_fatigue` | customers tire fast | punishes strategies that spend contact freely |
| `low_fatigue` | contact is nearly free | the naive blast looks best here |
| `fast_decay` | transient windows close 3× faster | timing becomes the dominant decision |
| `slow_decay` | windows stay open for weeks | timing barely matters |
| `unreliable_rails` | 3× bounces, intermittent gateway | exercises the DLQ and the outage/decline distinction |
| `pessimistic` | every unfavourable assumption at once | the floor |

The loader **refuses** to raise the retry probability of a structurally dead instrument in
any scenario. Allowing it would produce a world where brute force beats diagnosis, which
invalidates every comparison the simulator is used for.

An unknown scenario name raises rather than silently returning `default`. A sweep that
thinks it ran `pessimistic` and ran `default` reports a robustness result it never tested.
A typo'd parameter name raises too.

---

## Randomised assignment

`backend/app/experiments/assignment.py`, for the design a real causal claim would need.

```python
Experiment("recovery_v1",
           [Variant("control", 1.0), Variant("recoverai", 3.0)],
           holdout_fraction=0.10)
```

`SHA256(experiment_id | unit_id | salt)` → [0,1). Random and stable: a unit gets the same
arm forever, on any machine, with no state stored. `balance()` reports realised versus
intended arm sizes with a chi-square statistic, worth checking every run, because a salt
collision or a correlated unit-id scheme shows up there and nowhere else.

The **holdout** is separate from a control *variant*, and uses a separate hash namespace so
changing variant weights does not reshuffle who receives nothing. A control variant answers
"this treatment versus that one"; a holdout answers "any of this versus nothing". Quoting
one as the other is the most common way an experiment result gets overstated.

---

## Reproducibility

Every multi-seed run records:

```
git commit (with a -dirty marker) · random seeds · dataset split and size
model version · uplift model presence · simulation config fingerprint
optimizer on/off · generated_at · provenance + claim string
```

`SimulationConfig.fingerprint()` is a content hash of the entire simulated world, so a
result can be tied to the parameterisation that produced it rather than to a scenario
*name* that may since have been edited.

Every audit row carries `model_version`, `policy_version`, `agent_run_id` and `input_hash`,
and `POLICY_VERSION` is derived from the rules' own source plus the configured limits,
so a decision taken six months ago can be tied to the exact rule set that took it.

---

## Provenance

Every artefact is stamped. `scripts/run_multiseed.py` prints its claim string as the last
line of output:

> `simulation result -- the treatment effect holds inside the simulator and is not evidence about real customers`

`Provenance.combine()` refuses to pool kinds. See `docs/causal_inference.md`.

---

## Running things

```bash
# one seed, full pipeline, writes the database the dashboard reads
python scripts/run_experiment.py --fresh

# many seeds, distribution instead of a point
python scripts/run_multiseed.py --seeds 20

# every scenario
python scripts/run_multiseed.py --seeds 10 --sweep

# with the profit optimiser as arbiter
python scripts/run_multiseed.py --seeds 20 --optimizer

# the real-data track (observational, predictive claims only)
python scripts/build_real_dataset.py && python ml/train_real.py
python scripts/run_real_experiment.py

# verify the decision record
python scripts/verify_audit.py
python scripts/verify_audit.py --case txn_0001234
```

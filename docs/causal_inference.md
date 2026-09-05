# Causal inference, and what this project can and cannot claim

The single most misleading thing this project could do is present a simulated intervention
effect as evidence about the real world. This document is the constraint on every number
elsewhere in the repository.

---

## Three kinds of evidence

`backend/app/domain/provenance.py` makes these a value carried by the result, not a
caveat in prose that someone can drop when writing a summary.

| Kind | Source | Licenses |
|---|---|---|
| `REAL_OBSERVATIONAL_DATA` | Historical records (the UCI credit-default set) | **Prediction only.** Nothing was assigned, so any "effect" is confounded by whatever drove assignment. |
| `SIMULATED_INTERVENTION_DATA` | Our own simulator | A statement about **the simulator**, not about customers. The treatment effect is real *inside the model*. |
| `RANDOMIZED_EXPERIMENT_DATA` | Treatment assigned at random | An unqualified causal claim, over the randomised population only. |

`combine()` **refuses** to pool different kinds. Averaging a simulated effect with an
observational one produces a number that describes neither world.

Every experiment artefact carries `provenance` and a `claim` string, and the reports print
the claim string verbatim rather than paraphrasing it.

---

## What has actually been established

**Established (simulation).** Under the shipped simulator, across independent seeds and
multiple scenarios, the full agent recovers more than a no-touch control and than a fixed
retry baseline, with intervals reported in `data/runs/multiseed.json`. This is a statement
about the simulator's parameterisation. `config/simulation.yaml` contains eight
alternative parameterisations precisely so the claim can be stress-tested rather than
asserted; a result that holds only under `default` is a result about `default`.

**Established (observational).** On the real UCI delinquency data the model ranks who will
pay better than chance, with temporal-style held-out evaluation. This is a **predictive**
result. It says nothing about what an intervention would have done, because no
intervention was assigned.

**Not established.** Everything about real-world causal effect. No randomised experiment
has been run against real customers or a real payment rail. The per-action effect sizes in
`config/effects.yaml` are **priors**, labelled `CONFIGURED_PRIOR` in every estimate they
produce, and `EffectEstimate.is_measured` is False for all of them.

---

## The control arm, and why it is the point

```
                     gross recovered
                    ┌────────────────┐
                    │  would have    │  ← the control arm measures this
                    │  arrived anyway│
                    ├────────────────┤
                    │  caused by us  │  ← the only number worth quoting
                    └────────────────┘
```

The simulator draws a *self-cure hour* once per transaction from a dedicated RNG stream,
so the same case either self-cures or does not, whatever strategy is working it. That is
what makes the control a genuine counterfactual over one population rather than a
separate population, and it is why `check_self_cure` is called by **every** arm. An arm
that acts slowly would otherwise be credited with money that arrived while it waited.

Without a control arm, a system that works cases the customer was going to pay anyway is
indistinguishable from one that recovers money.

---

## Uplift modelling: what it is and what it is not

The recovery model answers *who will pay*. For spending a contact budget that is the wrong
question, and measurably so: on the real data, **44.8% of the top-250 cases ranked by
expected recovery paid on their own**. Roughly half the budget goes to people who needed
nothing.

The right target is the conditional average treatment effect:

```
uplift(x) = P(recover | treated, x) − P(recover | untreated, x)
```

which sorts the population into four groups, only one of which is worth spending on:

| | Pays if contacted | Pays if not |
|---|---|---|
| **Persuadable** | yes | no | ← the entire value of a dunning programme |
| **Sure thing** | yes | yes | ← contacting them is pure cost |
| **Lost cause** | no | no | ← contacting them is pure cost |
| **Sleeping dog** | no | yes | ← contacting them **loses** money |

Sleeping dogs are real in collections: an ill-timed message provokes a dispute or a
cancellation from someone who would otherwise have quietly paid. An outcome model cannot
distinguish any of these; it ranks sure things at the very top.

Implemented as a **T-learner**: two independent outcome models, one per arm, differenced.
Chosen over an S-learner because treatment effects here are small relative to the outcome
signal, and a single model lets the strong outcome features dominate and shrinks the
treatment term toward zero.

**The critical caveat.** A T-learner fitted on non-randomised data estimates a conditional
mean difference, not a treatment effect. `EffectModel.uplift_is_randomised` defaults to
`False`, and only the caller can set it, because only the caller knows how the data was
collected. When it is False, every estimate is labelled `CONFIGURED_PRIOR` and the detail
string says *"fitted on non-randomised data: a conditional mean, not a verified
counterfactual"*.

### Metrics

- **Qini curve**: incremental outcome from targeting the top-ranked fraction, with the
  control arm rescaled to the treated arm's size so an arm being larger cannot flatter it.
- **Qini coefficient**: area between the curve and the random-targeting diagonal. Zero
  means no better than random; negative means actively worse.
- **Uplift@k**: incremental value captured by treating only the top *k*.

`amount × uplift(x)` is the ranking score, not `uplift(x)` alone: ranking by percentage
points quietly selects the smallest balances (on this data, cases averaging $206 against
$917 for expected-value ranking) and loses on revenue despite finding more persuadable
customers. The objective is money, so the score has to carry the money.

---

## Randomised experimentation

`backend/app/experiments/assignment.py` provides the design a real causal claim would need.

Assignment is `SHA256(experiment_id | unit_id | salt)` mapped onto [0,1). This satisfies
two requirements that pull against each other:

- **Random**, or the arms differ by something other than the treatment.
- **Stable**, or a customer who reloads moves between arms, sees two treatments, and lands
  in both denominators. Assignment by `random.random()` at decision time satisfies the
  first and catastrophically fails the second.

Properties the tests pin:

- assignment is identical across processes and machines (not Python's per-process `hash()`);
- arms are uncorrelated across experiments, so one experiment does not contaminate the next;
- the global holdout is stable when variant weights change, so a design edit does not
  reshuffle who is receiving nothing;
- eligibility is evaluated **before** assignment, and a predicate that raises **excludes**
  rather than admits. Post-assignment filtering is how a clean randomisation quietly
  becomes a biased one.

---

## Statistics

`backend/app/experiments/stats.py`. No SciPy; the critical values are tabulated.

- **Multi-seed intervals use the t-distribution.** With five or ten seeds (the realistic
  range for a sweep) the normal approximation is about 25% too narrow, and an interval
  that is too narrow converts *we do not know* into a claim.
- **A single replication yields no interval.** `mean_interval([x])` returns ±∞ with the
  method string `"single replication: no interval is estimable from one draw"`.
- **Rate intervals are Wilson**, not Wald: the normal approximation can produce a lower
  bound below zero, and a negative recovery rate is a nonsense figure that still gets
  copied onto a slide.
- **Arm comparisons are paired across seeds.** The arms ran on the same population under
  the same seed, so seed-to-seed variation is common to both and differencing removes it.
  Treating them as independent inflates the interval with variance the design already
  controlled. `test_paired_comparison_removes_shared_seed_variation` demonstrates a case
  where the unpaired interval is swamped and the paired one is not.
- **Effect size is reported alongside the interval.** With enough simulated cases
  everything is significant; Cohen's *d* is how you tell significant from meaningful.
- **Worst and best seed are reported, not just the mean.** A positive mean with a negative
  worst seed is a different story from a positive mean everywhere, and reporting only the
  mean hides it.

---

## What would be needed for a real causal claim

In ascending order of cost:

1. **A sandbox integration** against a real provider, so the rails are real even if the
   money is not. The `PaymentRail` protocol is the seam; nothing else changes.
2. **A live holdout** in a real dunning programme: a randomly assigned fraction of failed
   payments that receives nothing, running long enough to cover the recovery horizon.
3. **Multi-armed randomisation** across `RETRY`, `EMAIL`, `SMS`, `PAYMENT_LINK`,
   `SUPPORT`. That is the only thing that would replace `config/effects.yaml`'s priors with
   measurements.
4. **Pre-registration** of the primary metric and the analysis, before the data arrives.

Until 2 and 3 exist, every per-action effect in this repository is a prior, and it is
labelled as one in the code, in the config file, and in the estimates it produces.

# Documentation

| Document | Read it for |
|---|---|
| [architecture.md](architecture.md) | The pipeline, the module map, the case lifecycle, and what is deliberately absent |
| [policy_engine.md](policy_engine.md) | The gate: four verdicts, 22 rules, evaluation order, and why hard rules are separate from optimisation |
| [ml.md](ml.md) | Features, leakage (including the control that is *missing*), calibration, the model registry, drift |
| [causal_inference.md](causal_inference.md) | **Start here if you are checking the claims.** What is demonstrated, what is simulated, what is not validated |
| [experiments.md](experiments.md) | The seven arms, multi-seed evaluation, scenarios, randomised assignment, reproducibility |
| [security.md](security.md) | Authentication, authorisation, tenant isolation, webhooks, prompt-injection defence |
| [threat_model.md](threat_model.md) | Thirteen threats with residual risk; nine are not closed |
| [production.md](production.md) | Profiles, configuration, integrating a real rail, and the ten things that must be true before real money moves |

## The three sentences that matter most

**The model proposes; the policy engine disposes.** No ML model and no LLM can cause a
side effect. The executor's input is the gate's output.

**Uncertainty resolves to not acting.** Of the gate's four verdicts, two permit execution.
A rule that crashes produces `HUMAN_REVIEW`, not an approval.

**Simulated is not real.** Every result carries its provenance, `combine()` refuses to
pool kinds, and no causal claim about real customers is available from anything in this
repository.

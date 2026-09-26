---
name: stochastic-training-variance
type: domain-knowledge
topic: Stochastic Training Variance Near a Threshold
token_cost: 90
keywords: [train, training, accuracy, threshold, classifier, fasttext, seed, epoch, hyperparameter, reproducible, deterministic, validation, holdout]
user-invocable: false
---
Training a model is not a single deterministic measurement -- random weight
initialization, negative sampling, data shuffling, and thread scheduling all
introduce run-to-run variance, even with identical hyperparameters and even
with a seed argument set (many training libraries' seed options are
inconsistently applied or not exposed at all in their documented API --
verify this for the specific library in front of you rather than assuming
`--seed` closes the gap). If a task grades a single run's metric against a
fixed pass/fail threshold, do not report or rely on one run's number,
especially when it lands close to that threshold: measure your own
run-to-run spread first (e.g. a couple of reruns, or cross-validation vs. an
independent holdout), and if the threshold sits inside that spread, retrain
across several seeds/resamples and report the best result, or ensemble
predictions across runs, rather than shipping whichever single draw happened
to land in this trial.

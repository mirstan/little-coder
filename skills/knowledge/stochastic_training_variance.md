---
name: stochastic-training-variance
type: domain-knowledge
topic: Stochastic Training Variance Near a Threshold
token_cost: 150
keywords: [train, accuracy, classifier, fasttext, epoch, hyperparameter, seed, cross-validation]
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
independent holdout). If the threshold sits inside that spread, prefer
raising the expected score outright -- more data, more epochs, better
hyperparameters -- so the run clears the threshold with margin instead of
depending on a favorable draw. If that is not practical, retrain across
several seeds/resamples, pick the best-scoring run on a selection split, and
then report its score on a held-out split that played no part in choosing
it: the selection split's own score is still just the most optimistic draw
in the sample, not an unbiased estimate of how the chosen model performs.
Only ensemble predictions across runs when the task's deliverable format
actually accepts multiple models -- a verifier that loads one native model
file from a fixed path will not accept a multi-model wrapper, and scores
that as a failed load rather than a pass.

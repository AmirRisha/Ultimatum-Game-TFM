# Experimental Design

This document describes the experiment as it is currently implemented in code. It is based on the logic in:

- [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs)
- [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs)
- [src/belief_model.mjs](/Users/mac/Documents/New project/src/belief_model.mjs)
- [src/automaton_profile.mjs](/Users/mac/Documents/New project/src/automaton_profile.mjs)
- [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs)
- [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs)

## 1. Implemented Game Structure

The project implements a repeated ultimatum game with two modes:

- Mode 1: `human_responder_vs_bot_proposer`
- Mode 2: `human_proposer_vs_bot_responder`

In both modes:

- a session has a fixed number of rounds
- stake and wealth are fixed for the full session
- acceptance implements the split
- rejection gives both players `0` for that round
- cumulative payoffs are updated additively across rounds

The payoff rule is implemented in [src/payoff_engine.mjs](/Users/mac/Documents/New project/src/payoff_engine.mjs).

## 2. Treatment Variables

The canonical stake levels are:

- `20`
- `200`
- `2000`
- `20000`

Wealth is binary:

- `0`
- `1`

The browser session engine and the step API both default to:

- `rounds = 10`
- `stake = 200`
- `wealth = 1`

## 3. Belief State and Prior Initialization

### 3.1 Responder beliefs

Responder-type beliefs are defined over four types:

- `money_maximizer`
- `fairness_sensitive`
- `stake_sensitive`
- `noisy`

The default responder prior in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs) is:

```json
{
  "money_maximizer": 0.13,
  "fairness_sensitive": 0.78,
  "stake_sensitive": 0.05,
  "noisy": 0.04
}
```

In [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs) and [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs):

- if `priorsSource === "literature"`, responder beliefs are initialized from the literature prior above
- if `priorsSource === "calibration"`, responder beliefs are initialized from `fittedParams.priors?.responder`

This applies to:

- human responder beliefs in mode 1
- bot responder beliefs used in mode 2

### 3.2 Proposer beliefs

Proposer-type beliefs are also represented over four types:

- `money_maximizer`
- `fairness_sensitive`
- `stake_sensitive`
- `noisy`

By default, proposer priors come from `fittedParams.priors?.proposer`, with the fallback in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs) equal to a uniform prior.

### 3.3 Bayesian update rule

The update rule is implemented in [src/belief_model.mjs](/Users/mac/Documents/New project/src/belief_model.mjs):

\[
\tilde{\pi}_{t+1}(k) \propto \pi_t(k)\,L_k
\]

where:

- \(\pi_t(k)\) is the current belief on type \(k\)
- \(L_k\) is the likelihood of the observed action under type \(k\)

This is followed by two blending steps:

1. learning-rate blending toward the pure posterior
2. pullback blending toward an anchor prior

The implemented posterior is therefore:

- pure Bayes update
- then blend with the previous belief using `learningRate`
- then blend with `pullbackPrior` using `pullback`

Defaults in the browser engine and step API are:

- `beliefLearningRate = 0.85`
- `beliefPullback = 0.02`

This means the implemented learning rule is not a pure one-step Bayes update unless the caller explicitly sets `learningRate = 1` and `pullback = 0`.

## 4. Responder Acceptance Model

### 4.1 Baseline reject probability

The baseline responder component is implemented in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs):

\[
P(\text{reject}) =
\sigma(\beta_0 + \beta_1\,wealth + \beta_2\,stake200 + \beta_3\,stake2000 + \beta_4\,stake20000 + \beta_5\,offerShare)
\]

where `stake200`, `stake2000`, and `stake20000` are bucket dummies derived from the stake.

If `policyMode === "nn"` and `fittedParams.nnModel` is available, the baseline reject probability is replaced by the neural-network prediction from [src/nn_inference.mjs](/Users/mac/Documents/New project/src/nn_inference.mjs). The rest of the belief update logic remains unchanged.

### 4.2 Type-specific responder acceptance

The type-conditioned accept probability is `1 - baselineRejectProbability(...)`, modified by type-specific rules.

#### `money_maximizer`

- if `offerShare <= 0`, accept probability is `0.02`
- otherwise:

\[
P_{MM}(\text{accept}) = \text{clamp}(0.98 + 0.015 \cdot baselineAccept,\; 0.02,\; 0.999)
\]

#### `fairness_sensitive`

Threshold:

\[
threshold = \text{clamp}(0.29 + (wealth==0 ? 0.03 : -0.01),\; 0.2,\; 0.4)
\]

Fairness response:

\[
fairnessAccept = \sigma((offerShare - threshold)\cdot 24)
\]

Final acceptance:

\[
P_{FS}(\text{accept}) = \text{clamp}(0.7\cdot fairnessAccept + 0.3\cdot baselineAccept,\; 0.001,\; 0.999)
\]

#### `stake_sensitive`

This type is implemented as a mixture of `fairness_sensitive` and `money_maximizer`.

\[
P_{SS}(\text{accept}) = \text{clamp}(\lambda(stake)\,P_{FS}(\text{accept}) + (1-\lambda(stake))\,P_{MM}(\text{accept}),\; 0.001,\; 0.999)
\]

with:

\[
\lambda(S)=\sigma(\kappa(\ln S_0 - \ln S))
\]

The constants are calibrated in code to satisfy approximately:

- `lambda(20) = 0.90`
- `lambda(20000) = 0.15`

The current implementation documents:

- `kappa ~= 0.5692`
- `S0 ~= 949.56`

#### `noisy`

\[
P_{noisy}(\text{accept}) = \text{clamp}(0.6\cdot baselineAccept + 0.4\cdot 0.5,\; 0.05,\; 0.95)
\]

### 4.3 Belief-weighted acceptance probability

The mixture acceptance probability used by the proposer is:

\[
P(\text{accept}\mid context) = \sum_k \pi(k)\,P_k(\text{accept}\mid context)
\]

implemented in `beliefWeightedAcceptProbability(...)`.

## 5. Proposer Type Representation

### 5.1 Human proposer inference types

Observed human offers in mode 2 are interpreted through proposer likelihoods over:

- `money_maximizer`
- `fairness_sensitive`
- `stake_sensitive`
- `noisy`

The proposer target shares in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs) are:

- `money_maximizer`: `clamp(0.08 + (wealth === 0 ? 0.02 : 0), 0.02, 0.2)`
- `fairness_sensitive`: `clamp(0.4 + (wealth === 0 ? 0.02 : -0.02), 0.25, 0.5)`
- `stake_sensitive`: `clamp(0.34 - 0.055 * log10(stake / 20 + 1), 0.12, 0.36)`
- `noisy`: `0.25`

Likelihood for observed offer share:

- `noisy`: constant likelihood `1`
- all other proposer types: Gaussian density around the target share

Sigma values are:

- `money_maximizer`: `0.06`
- `fairness_sensitive`: `0.08`
- `stake_sensitive`: `0.09`

## 6. Bot Proposer Profile

The bot proposer automaton profile is defined in [src/automaton_profile.mjs](/Users/mac/Documents/New project/src/automaton_profile.mjs).

### 6.1 Session-level proposer type

The bot proposer is sampled once per session from:

```json
{
  "maximizer": 0.5,
  "fairness_sensitive": 0.5
}
```

This draw is fixed for the session in both:

- [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs)
- [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs)

### 6.2 Fairness beta

If the sampled proposer type is `maximizer`, then:

- `beta = 0`

If the sampled proposer type is `fairness_sensitive`, the beta parameter is determined once per session:

- mode `A`: fixed beta
- mode `B`: one draw from `{0.25, 0.6}` with weights `{0.4286, 0.5714}`

Defaults:

- `betaMode = "B"`
- `fixedBeta = 0.6`

## 7. Bot Proposer Offer Policy

### 7.1 Offer grid

The default offer grid is generated in [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs):

- `minShare = 0`
- `maxShare = 0.5`
- `stepShare = 0.025`

Each grid share is converted to:

- `offerAmount = round(stake * gridShare)`
- `offerShare = offerAmount / stake`

The implementation therefore evaluates discrete currency amounts, not the raw share grid alone.

### 7.2 Candidate evaluation

For each candidate offer:

- compute belief-weighted accept probability
- compute proposer objective

For all candidates:

\[
EV_{max} = P(\text{accept})\cdot (stake - offerAmount)
\]

For fairness-sensitive proposers with current session beta:

\[
U_{FS,current} = (stake - offerAmount) - \beta(stake - 2\cdot offerAmount)
\]

\[
EU_{FS,current} = P(\text{accept})\cdot U_{FS,current}
\]

The code also computes two reference utilities:

- `EU` at `beta = 0.25`
- `EU` at `beta = 0.60`

Selection score:

- if proposer automaton type is `maximizer`, `expectedValue = EV_max`
- otherwise, `expectedValue = EU_FS_current`

### 7.3 Offer-selection rule

After scores are computed, the proposer samples an offer according to `selectionMode`.

#### `softmax`

\[
p_i \propto \exp(score_i / temperature)
\]

#### `proportional_ev`

Scores are shifted to nonnegative values:

\[
shifted_i = (score_i - \min_j score_j) + 10^{-9}
\]

then normalized linearly.

#### `normal_around_best`

Let:

- \(i^* = \arg\max_i score_i\)
- \(\mu = offerShare_{i^*}\)
- \(\sigma = normalSigmaSteps \times offerStepShare\)

Then:

\[
p_i \propto \phi(offerShare_i;\mu,\sigma)
\]

The implemented code tries to infer `offerStepShare` from the realized candidate share spacing and falls back to the configured `offerStepShare` if needed.

### 7.4 Exploration and tremble

The proposer policy also includes:

- `trembleProb`
- `epsilon`

Their semantics are the same in all selection modes:

- with probability `trembleProb`, choose a uniformly random candidate
- else with probability `epsilon`, choose a uniformly random candidate
- otherwise sample from the selection-mode distribution

Defaults in the engines:

- `proposerEpsilon = 0.08`
- `proposerTemperature = 0.1`
- `proposerTrembleProb = 0.03`
- `proposerSelectionMode = "softmax"`
- `proposerNormalSigmaSteps = 1.5`

## 8. Bot Responder Policy in Mode 2

The bot responder logic is implemented in [src/bot_responder_policy.mjs](/Users/mac/Documents/New project/src/bot_responder_policy.mjs).

### 8.1 Inputs

The bot responder uses:

- updated beliefs over human proposer types
- `botResponderBeliefs` over responder acceptance types
- stake, wealth, round index, offer share, and offer amount

### 8.2 Acceptance model

First compute:

\[
baseAcceptProb = beliefWeightedAcceptProbability(botResponderBeliefs, context)
\]

Then compute the expected human offer share under proposer beliefs:

\[
E[offerShare] = \sum_k \pi_{prop}(k)\,targetShare_k
\]

Then apply a reputation-style adjustment:

\[
offerSignal = offerShare - E[offerShare]
\]

\[
stakeScale = clamp(\log_{10}(stake/20 + 1)/3,\; 0,\; 1)
\]

\[
reputationAdjustment =
0.25\cdot offerSignal
+ 0.08\cdot fairWeight
- 0.12\cdot selfishWeight
+ 0.04\cdot stakeSensitiveWeight\cdot stakeScale
\]

\[
adjustedAcceptProb = clamp(baseAcceptProb + reputationAdjustment,\; 0.001,\; 0.999)
\]

Decision probability:

\[
decisionProbability = \sigma((adjustedAcceptProb - 0.5) / temperature)
\]

Then:

- draw accept/reject from `decisionProbability`
- with probability `trembleProb`, flip the decision

Defaults:

- `responderTemperature = 0.08`
- `responderTrembleProb = 0.03`

### 8.3 Important asymmetry

The mode-2 bot responder does not use the sampled proposer automaton profile (`bot_profile`) or fairness beta. Those values still exist in state, but they do not enter the responder decision rule.

## 9. Session Integration

### 9.1 Browser session engine

In [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs):

- mode 1:
  - bot offer is generated by `chooseBotOffer(...)`
  - human accept/reject updates responder beliefs
- mode 2:
  - human offer updates proposer beliefs
  - bot acceptance is generated by `decideBotResponse(...)`

The browser session engine uses:

- non-deterministic randomness by default (`Math.random`) unless an RNG is injected
- a mutable class-based state machine

### 9.2 Pure step API

In [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs):

- state is JSON-serializable
- RNG is deterministic from `rng_seed`
- mode 1 is a two-step interaction:
  - `step(state, null)` generates the pending offer
  - `step(new_state, accept_or_reject)` resolves the round
- mode 2 resolves in one call

The step API and session engine are logically aligned on:

- priors
- belief updates
- proposer selection configuration
- proposer automaton sampling

They are not fully identical in all text outputs:

- rationale strings are not constructed in exactly the same way
- debug payloads are similar but not identical in provenance

## 10. Logging and Outputs

The implemented round log schema is fixed by [src/logger.mjs](/Users/mac/Documents/New project/src/logger.mjs):

- `round`
- `mode`
- `stake`
- `wealth`
- `offer_amount`
- `offer_share`
- `accept_reject`
- `bot_action`
- `bot_beliefs`
- `inferred_type`
- `decision_rationale`
- `expected_accept_prob`
- `expected_value`
- `cumulative_payoffs`

Not included in CSV logs:

- `proposerSelectionMode`
- `selectedProb`
- candidate-level proposer grid probabilities

## 11. Implementation Verification Notes

### Confirmed features

- Bayesian updating is implemented for responder observations and proposer-offer observations.
- The implemented update includes both `learningRate` and `pullback`.
- Literature-based responder priors are implemented and are the default in the engines.
- `stake_sensitive` responder behavior is implemented as a smooth stake-dependent mixture of `fairness_sensitive` and `money_maximizer`.
- The bot proposer has a session-level automaton profile with a fixed fairness beta for the full session.
- The bot proposer objective depends on automaton type:
  - maximizer: expected monetary payoff
  - fairness-sensitive: fairness-adjusted expected utility
- Three proposer selection modes are implemented in the proposer policy:
  - `softmax`
  - `proportional_ev`
  - `normal_around_best`
- The pure step API supports proposer selection mode and normal sigma configuration.

### Missing features

- No equilibrium computation layer is implemented.
- No static replay of empirical frequencies is implemented.
- No mode-2 responder automaton profile is implemented; mode 2 uses the explicit responder decision rule in [src/bot_responder_policy.mjs](/Users/mac/Documents/New project/src/bot_responder_policy.mjs).
- No round-log fields are implemented for proposer sampling mode or selected sampling probability.

### Partial implementations and asymmetries

- The bot proposer automaton is only two-type (`maximizer`, `fairness_sensitive`), while proposer belief inference in mode 2 remains four-type.
- Candidate-level `selectionProb` and `scoreUsed` are computed in [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs), but the proposer grids stored by the session engine and step API drop those fields.
- [scripts/stress_test.mjs](/Users/mac/Documents/New project/scripts/stress_test.mjs) validates all three proposer selection modes in a dedicated validation routine, but the main Monte Carlo loop does not expose a CLI argument for changing `proposerSelectionMode`.
- [scripts/example_step_session.mjs](/Users/mac/Documents/New project/scripts/example_step_session.mjs) does not currently pass proposer-selection settings into `createInitialState`, even though the step API supports them.
- The browser session engine defaults to non-deterministic randomness, while the step API is deterministic from `rng_seed`.

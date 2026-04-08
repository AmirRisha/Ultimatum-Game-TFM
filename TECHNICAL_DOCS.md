# Technical Documentation

This document is a code-faithful description of the current repository. It is aligned with [Experimental_Design.md](/Users/mac/Documents/New project/Experimental_Design.md) and is based on direct inspection of the implementation modules rather than prior documentation text.

Primary logic sources:

- [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs)
- [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs)
- [src/belief_model.mjs](/Users/mac/Documents/New project/src/belief_model.mjs)
- [src/automaton_profile.mjs](/Users/mac/Documents/New project/src/automaton_profile.mjs)
- [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs)
- [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs)

## 1. Repository Structure

| Path | Current role in implementation |
|---|---|
| [src/utils.mjs](/Users/mac/Documents/New project/src/utils.mjs) | math, normalization, sampling, seeded RNG |
| [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs) | responder-type acceptance rules, proposer-type target shares, priors, baseline reject model |
| [src/nn_inference.mjs](/Users/mac/Documents/New project/src/nn_inference.mjs) | neural-network reject-probability inference |
| [src/data_loader.mjs](/Users/mac/Documents/New project/src/data_loader.mjs) | CSV parsing, row normalization, offer conversions |
| [src/belief_model.mjs](/Users/mac/Documents/New project/src/belief_model.mjs) | belief initialization, Bayesian update, belief-weighted accept probability |
| [src/automaton_profile.mjs](/Users/mac/Documents/New project/src/automaton_profile.mjs) | bot proposer automaton-type and fairness-beta sampling |
| [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs) | candidate-offer evaluation and proposer action sampling |
| [src/bot_responder_policy.mjs](/Users/mac/Documents/New project/src/bot_responder_policy.mjs) | mode-2 bot responder decision rule |
| [src/payoff_engine.mjs](/Users/mac/Documents/New project/src/payoff_engine.mjs) | ultimatum-game payoff resolution |
| [src/logger.mjs](/Users/mac/Documents/New project/src/logger.mjs) | canonical round log schema and CSV output |
| [src/sim_agents.mjs](/Users/mac/Documents/New project/src/sim_agents.mjs) | synthetic human types for stress testing |
| [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs) | mutable browser/session runtime |
| [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs) | pure JSON-serializable state transition API |
| [scripts/fit_logit.mjs](/Users/mac/Documents/New project/scripts/fit_logit.mjs) | calibration script for baseline reject model and inferred priors |
| [scripts/train_nn.mjs](/Users/mac/Documents/New project/scripts/train_nn.mjs) | neural-network training script |
| [scripts/example_step_session.mjs](/Users/mac/Documents/New project/scripts/example_step_session.mjs) | example `step()` session runner |
| [scripts/stress_test.mjs](/Users/mac/Documents/New project/scripts/stress_test.mjs) | batch simulation and proposer-selection validation |

## 2. Core Mathematical Components

### 2.1 Belief initialization

In [src/belief_model.mjs](/Users/mac/Documents/New project/src/belief_model.mjs), `createInitialBeliefs(typeNames, priors)`:

- returns a uniform distribution if `priors` is absent
- otherwise reads only the provided keys for the requested type set
- normalizes the resulting vector with `normalizeBeliefMap`

This means prior objects may contain additional keys, but only the requested type names are retained.

### 2.2 Bayesian update with inertia and pullback

`bayesUpdate(priorBeliefs, likelihoodByType, options)` performs:

1. unnormalized Bayes step:
   \[
   \tilde{\pi}(k)=\max(\pi(k),10^{-9})\cdot \max(L(k),10^{-9})
   \]
2. normalization to a pure posterior
3. blending of prior and posterior using `learningRate`
4. blending of that result with `pullbackPrior` using `pullback`

Implementation details:

- `learningRate` is clamped to `[0, 1]`
- `pullback` is clamped to `[0, 0.5]`
- `pullbackPrior` is normalized before use

This is not a pure Bayes update unless the caller sets `learningRate = 1` and `pullback = 0`.

### 2.3 Belief update entry points

`updateResponderBeliefs(...)`:

- builds likelihoods from `responderAcceptProbability(type, context, fittedParams, { policyMode })`
- uses accept likelihood if the observed action is accept
- uses reject likelihood `1 - acceptProb` if the observed action is reject

`updateProposerBeliefs(...)`:

- adds `offerShare` into context
- builds proposer-type likelihoods with `proposerOfferLikelihood(type, enrichedContext)`

## 3. Responder Types and Acceptance Functions

The responder types are defined in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs):

- `money_maximizer`
- `fairness_sensitive`
- `stake_sensitive`
- `noisy`

### 3.1 Baseline reject model

`baselineRejectProbability(context, fittedParams, options)`:

- uses a logistic reject model by default
- if `policyMode === "nn"` and `fittedParams.nnModel` exists, uses the neural-network prediction instead

The logistic model uses:

\[
\beta_0 + \beta_1\,wealth + \beta_2\,stake200 + \beta_3\,stake2000 + \beta_4\,stake20000 + \beta_5\,offerShare
\]

Stake dummy variables are generated by the local `stakeBucket()` rule:

- `stake <= 110` maps to bucket `20`
- `stake <= 1100` maps to `200`
- `stake <= 11000` maps to `2000`
- otherwise `20000`

### 3.2 `money_maximizer`

Implemented rule:

- `0.02` if `offerShare <= 0`
- otherwise `clamp(0.98 + 0.015 * baselineAccept, 0.02, 0.999)`

### 3.3 `fairness_sensitive`

Implemented threshold:

- `0.29 + 0.03` if `wealth == 0`
- `0.29 - 0.01` if `wealth == 1`
- then clamped to `[0.2, 0.4]`

Acceptance rule:

- `fairnessAccept = sigmoid((offerShare - threshold) * 24)`
- final probability `clamp(0.7 * fairnessAccept + 0.3 * baselineAccept, 0.001, 0.999)`

### 3.4 `stake_sensitive`

Implemented as a stake-dependent mixture:

- `fsProb = responderAcceptProbability("fairness_sensitive", ...)`
- `mmProb = responderAcceptProbability("money_maximizer", ...)`
- `lam = lambdaFromStake(context.stake)`
- final probability `clamp(lam * fsProb + (1 - lam) * mmProb, 0.001, 0.999)`

`lambdaFromStake(stake)` is defined in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs) using:

\[
\lambda(S)=\sigma(\kappa(\ln S_0 - \ln S))
\]

The file comments explicitly state the anchor calibration:

- `lambda(20) ~= 0.90`
- `lambda(20000) ~= 0.15`
- `kappa ~= 0.5692`
- `S0 ~= 949.56`

### 3.5 `noisy`

Implemented rule:

- `trembleAccept = 0.5`
- final probability `clamp(0.6 * baselineAccept + 0.4 * trembleAccept, 0.05, 0.95)`

## 4. Priors

### 4.1 Default fitted parameter object

`DEFAULT_FITTED_PARAMS` in [src/types.mjs](/Users/mac/Documents/New project/src/types.mjs) contains:

- logit coefficients
- `nnModel: null`
- default responder prior equal to `LITERATURE_PRIOR_2_RESPONDER`
- default proposer prior uniform over the four proposer types

### 4.2 Engine-level prior selection

In [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs) and [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs):

- responder prior source is configurable via `priorsSource`
- default is `"literature"`
- `"calibration"` switches responder priors to `fittedParams.priors?.responder`

Important asymmetry:

- proposer beliefs are always initialized from `fittedParams.priors?.proposer`
- `priorsSource` only changes responder priors

## 5. Proposer Type Models

### 5.1 Proposer latent types for belief inference

`PROPOSER_TYPES` contains four types:

- `money_maximizer`
- `fairness_sensitive`
- `stake_sensitive`
- `noisy`

These are used in:

- `updateProposerBeliefs(...)`
- `proposerTargetShare(...)`
- `proposerOfferLikelihood(...)`
- `sim_agents.mjs`

### 5.2 `proposerTargetShare`

Implemented target shares:

- `money_maximizer`: `clamp(0.08 + (wealth === 0 ? 0.02 : 0), 0.02, 0.2)`
- `fairness_sensitive`: `clamp(0.4 + (wealth === 0 ? 0.02 : -0.02), 0.25, 0.5)`
- `stake_sensitive`: `clamp(0.34 - 0.055 * Math.log10(stake / 20 + 1), 0.12, 0.36)`
- `noisy`: `0.25`

### 5.3 `proposerOfferLikelihood`

Likelihood rule:

- `1` for `noisy`
- Gaussian PDF centered at `proposerTargetShare(type, context)` for other types

Sigma values:

- `0.06` for `money_maximizer`
- `0.08` for `fairness_sensitive`
- `0.09` otherwise

## 6. Bot Proposer Automaton Profile

The bot proposer automaton is separate from the four proposer belief types.

In [src/automaton_profile.mjs](/Users/mac/Documents/New project/src/automaton_profile.mjs):

- implemented automaton types:
  - `maximizer`
  - `fairness_sensitive`
- default automaton mix:
  - `maximizer: 0.5`
  - `fairness_sensitive: 0.5`
- default beta mode:
  - `"B"`
- fixed beta default:
  - `0.6`

`sampleAutomatonType(mix, rng)`:

- reads `mix.maximizer` and `mix.fairness_sensitive`
- passes those values directly to `sampleCategorical`

`sampleFsBetaOnce(betaMode, fixedBeta, rng)`:

- returns `fixedBeta` in mode `"A"`
- samples once from `[0.25, 0.6]` with weights `[0.4286, 0.5714]` in mode `"B"`

Important asymmetry:

- the bot proposer automaton does not include `stake_sensitive` or `noisy`
- the human proposer inference model still does

## 7. Bot Proposer Offer Policy

### 7.1 Candidate generation

In [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs):

- default grid spans `0` to `0.5`
- default step is `0.025`
- each share is clamped and rounded to four decimals

For each grid share:

- `offerAmount = round(stake * gridShare)`
- `offerShare = clamp(offerAmount / stake, 0, 1)`

This means two adjacent grid shares can map to the same rounded amount at small stakes.

### 7.2 Candidate objective values

For each candidate:

- `acceptProb` is belief-weighted over responder types
- `ev_max = acceptProb * (stake - offerAmount)`
- `utility_raw_fs_current = (stake - offerAmount) - betaUsed * (stake - 2 * offerAmount)`
- `eu_fs_current = acceptProb * utility_raw_fs_current`
- `eu_fs_beta025 = acceptProb * ((stake - offerAmount) - 0.25 * (stake - 2 * offerAmount))`
- `eu_fs_beta060 = acceptProb * ((stake - offerAmount) - 0.60 * (stake - 2 * offerAmount))`

Selection score:

- `expectedValue = ev_max` if `automatonType === "maximizer"`
- `expectedValue = eu_fs_current` otherwise

### 7.3 Selection modes

Implemented `selectionMode` values:

- `softmax`
- `proportional_ev`
- `normal_around_best`

`softmax`:

- uses `softmax(scores, temperature)`

`proportional_ev`:

- shifts the score vector by `minScore`
- adds `1e-9`
- normalizes linearly
- falls back to uniform if the total weight is invalid or zero

`normal_around_best`:

- identifies the argmax score index
- centers a Gaussian at that candidate’s `offerShare`
- computes `sigma = normalSigmaSteps * inferredOfferStepShare`
- normalizes the Gaussian weights

The proposer policy computes candidate-level:

- `scoreUsed`
- `selectionProb`

but these fields are only retained inside the proposer policy’s own debug output.

### 7.4 Exploration and tremble

The proposer policy uses:

- `epsilon`
- `trembleProb`

Behavior:

- `trembleProb` overrides all modes with a uniform random offer
- otherwise `epsilon` overrides with a uniform random offer
- otherwise the configured selection-mode distribution is used

Defaults in engine integrations:

- `epsilon = 0.08`
- `temperature = 0.1`
- `trembleProb = 0.03`
- `selectionMode = "softmax"`
- `normalSigmaSteps = 1.5`

## 8. Bot Responder Policy

In [src/bot_responder_policy.mjs](/Users/mac/Documents/New project/src/bot_responder_policy.mjs), mode-2 response is not generated from the automaton profile. Instead it uses:

1. `baseAcceptProb = beliefWeightedAcceptProbability(responderBeliefs, context, fittedParams, policyMode)`
2. proposer-belief expectations and weights
3. a reputation-style adjustment
4. a sigmoid response threshold around `0.5`
5. an optional tremble flip

Computed quantities:

- `expectedOfferShare`
- `reputationAdjustment`
- `adjustedAcceptProb`
- `decisionProbability`
- `expectedValue = adjustedAcceptProb * offerAmount`

Important asymmetry:

- the returned `expectedValue` is logged, but the accept/reject choice is based on `decisionProbability`, not on a direct expected-value comparison

## 9. Session Engine Integration

### 9.1 `RepeatedUltimatumSession`

The browser runtime in [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs):

- keeps mutable state
- uses injected RNG or `Math.random`
- samples `botProfile` once in the constructor
- initializes:
  - `humanResponderBeliefs`
  - `humanProposerBeliefs`
  - `botResponderBeliefs`

Mode 1:

- `startRoundForHumanResponderMode()` calls `chooseBotOffer(...)`
- `submitHumanResponse(accepted)` updates responder beliefs and payoffs

Mode 2:

- `submitHumanOffer(...)` updates proposer beliefs
- then calls `decideBotResponse(...)`

The session debug state includes:

- current beliefs
- inferred type
- proposer grid
- expected accept probability
- `proposerSelectionMode`
- `selectedProb`
- `bot_profile`

But the stored `proposerGrid` deliberately strips:

- `scoreUsed`
- `selectionProb`

### 9.2 Payoffs

`resolvePayoffs(...)` in [src/payoff_engine.mjs](/Users/mac/Documents/New project/src/payoff_engine.mjs) implements standard ultimatum payoffs:

- mode 1 accepted:
  - human gets `offerAmount`
  - bot gets `stake - offerAmount`
- mode 2 accepted:
  - human gets `stake - offerAmount`
  - bot gets `offerAmount`
- rejected:
  - both get `0`

## 10. Step API Integration

The pure logic API in [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs):

- serializes all state as plain JSON
- stores a deterministic RNG state in `rng_seed`
- samples the same proposer automaton profile as the browser engine
- threads proposer-selection configuration into `chooseBotOffer(...)`

Mode 1:

- `step(state, null)` generates a pending offer
- `step(state_with_pending_offer, humanAction)` resolves the round

Mode 2:

- `step(state, humanAction)` resolves the round directly

The step API accepts the following proposer-policy config fields:

- `proposerSelectionMode`
- `proposerNormalSigmaSteps`
- `offerStepShare`

Like the browser engine, it carries:

- `selectedProb` for the chosen offer
- `proposerSelectionMode` in `last_debug`

Like the browser engine, it does not preserve candidate-level `selectionProb` inside the stored proposer grid.

## 11. Data and Training Scripts

### 11.1 Calibration data handling

[src/data_loader.mjs](/Users/mac/Documents/New project/src/data_loader.mjs) provides:

- CSV parsing with quoted-field support
- normalization to rows with:
  - `stake`
  - `offerAmount`
  - `offerShare`
  - `wealth`
  - `accept`
  - `reject`
  - `year`
- `calibrationRows(...)` for the logit fit
- `nnTrainingRows(...)` for neural-network training

### 11.2 `scripts/fit_logit.mjs`

Current behavior:

- reads `data/20100982_DATA.csv` by default
- filters rows with `calibrationRows(...)`
- fits the logistic reject model
- infers responder priors using `responderAcceptProbability(...)`
- infers proposer priors using `proposerOfferLikelihood(...)`
- writes `data/fitted_params.json`

This means calibration priors are endogenous to the currently implemented type rules.

### 11.3 `scripts/train_nn.mjs`

Current behavior:

- uses features:
  - `offerShare`
  - `Math.log(stake)`
  - `wealth`
- trains a one-hidden-layer MLP
- hidden activation: `tanh`
- output activation: `sigmoid`
- optimizer: Adam
- writes `data/nn_model.json`

### 11.4 `src/nn_inference.mjs`

Runtime NN inference:

- rebuilds the same three input features
- normalizes them using the stored means and standard deviations
- performs a forward pass through the saved weights
- returns a clamped reject probability

## 12. Simulation and Validation Scripts

### 12.1 `scripts/example_step_session.mjs`

Implemented behavior:

- demonstrates end-to-end use of the step API
- supports:
  - `mode`
  - `policyMode`
  - `rounds`
  - `stake`
  - `wealth`
  - `seed`
  - output path arguments

Not implemented in this script:

- explicit threading of `proposerSelectionMode`
- explicit threading of `proposerNormalSigmaSteps`

### 12.2 `scripts/stress_test.mjs`

Current behavior has two distinct parts:

1. main stress-test loop:
   - runs both experiment modes
   - varies stake and wealth
   - uses synthetic human agents
   - writes:
     - `stress_sessions.csv`
     - `stress_summary.csv`
2. proposer-selection validation routine:
   - runs mode 1 only
   - iterates over stakes
   - iterates over:
     - `softmax`
     - `proportional_ev`
     - `normal_around_best`
   - checks:
     - finite `expectedAcceptProb`
     - finite `expectedValue`
     - round completion
     - candidate selection probabilities sum to approximately one if they are present
   - writes:
     - `selection_mode_validation.csv`

Important limitation:

- the validation routine reads proposer grids from the session engine, and those grids do not currently carry candidate-level `selectionProb`
- therefore the validation script can legitimately record `selection_prob_check = "not_logged"`

## 13. Implementation Verification Notes

### Confirmed features

- `src/types.mjs` implements a stake-dependent `lambdaFromStake` function and uses it inside the `stake_sensitive` responder rule.
- `src/bot_proposer_policy.mjs` implements EV for maximizers and fairness-adjusted EU for fairness-sensitive proposers.
- `src/bot_proposer_policy.mjs` implements three proposer sampling modes.
- `src/belief_model.mjs` implements Bayesian updating with additional learning-rate and pullback terms.
- `src/automaton_profile.mjs` samples the bot proposer type and fairness beta once per session.
- `src/session_engine.mjs` and `src/step_api.mjs` both thread proposer selection settings into the proposer policy.

### Missing features

- No equilibrium or static-replay module exists in the repository.
- No candidate-level selection probabilities are written to CSV logs.
- No mode-2 responder automaton profile is used in the responder decision function.

### Partial implementations and inconsistencies

- The session engine and step API preserve `selectedProb` for the chosen offer, but not `selectionProb` for every candidate in the stored proposer grid.
- The bot proposer automaton space is two-type, while proposer belief inference for human offers remains four-type.
- `priorsSource` only governs responder priors; proposer priors do not have an analogous literature/calibration switch.
- [scripts/example_step_session.mjs](/Users/mac/Documents/New project/scripts/example_step_session.mjs) does not expose the full proposer-selection configuration even though the underlying step API supports it.

# Adaptive Repeated Ultimatum Game

This repository implements a repeated ultimatum game with adaptive belief updates and two playable modes:

- `human_responder_vs_bot_proposer`
- `human_proposer_vs_bot_responder`

The current implementation is documented from the codebase, not from prior design notes:

- [Experimental_Design.md](/Users/mac/Documents/New project/Experimental_Design.md)
- [TECHNICAL_DOCS.md](/Users/mac/Documents/New project/TECHNICAL_DOCS.md)

## Implemented Scope

- Fixed-within-session treatment variables:
  - canonical/UI stake levels: `20`, `200`, `2000`, `20000`
  - wealth: `0` or `1`
- Online belief updating over latent types for:
  - the human responder in mode 1
  - the human proposer in mode 2
- Belief updates use a Bayesian likelihood step with:
  - `beliefLearningRate`
  - `beliefPullback`
  - prior anchors
- Optional responder likelihood backend:
  - `belief` mode: logit baseline reject model
  - `nn` mode: neural-network baseline reject model, if `data/nn_model.json` is present
- Bot proposer automaton profile:
  - `maximizer`
  - `fairness_sensitive`
- Bot proposer offer-selection rules:
  - `softmax`
  - `proportional_ev`
  - `normal_around_best`
- Pure logic API in [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs)

## Not Implemented

- No equilibrium solver or equilibrium simulation layer
- No static frequency replay
- No closed-form game-theoretic solution module
- No per-round re-drawing of the bot proposer automaton within a session
- No CSV fields for proposer selection mode or selected sampling probability

## Important Asymmetries

- The bot proposer automaton is sampled from only two types (`maximizer`, `fairness_sensitive`), but human proposer belief inference still uses four proposer types (`money_maximizer`, `fairness_sensitive`, `stake_sensitive`, `noisy`).
- The bot responder policy in mode 2 does not use the sampled `bot_profile` automaton or fairness beta, even though `bot_profile` still exists in session state and debug output.
- Candidate-level proposer sampling probabilities are computed inside [src/bot_proposer_policy.mjs](/Users/mac/Documents/New project/src/bot_proposer_policy.mjs), but they are not preserved in the proposer grid exposed by [src/session_engine.mjs](/Users/mac/Documents/New project/src/session_engine.mjs) or [src/step_api.mjs](/Users/mac/Documents/New project/src/step_api.mjs).

## Run

Browser experiment:

```bash
cd "/Users/mac/Documents/New project"
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

Example pure-logic session:

```bash
cd "/Users/mac/Documents/New project"
node scripts/example_step_session.mjs --mode=mode1 --policyMode=belief --rounds=10 --stake=200 --wealth=1
```

Stress test:

```bash
cd "/Users/mac/Documents/New project"
node scripts/stress_test.mjs --sessions=1000 --rounds=10 --seed=20260218 --outDir=outputs
```

## Implementation Verification Notes

Confirmed features:

- Literature-based responder priors are implemented and can be overridden by `priorsSource: "calibration"`.
- `stake_sensitive` responder acceptance is implemented as a smooth mixture of `fairness_sensitive` and `money_maximizer` using `lambdaFromStake`.
- Fairness-sensitive bot proposers optimize a fairness-adjusted expected utility, not pure proposer payoff.
- Three proposer selection modes are implemented in the proposer policy module and threaded through the browser session engine and the pure step API.

Missing or partial features:

- `Experimental_Design.md` did not previously exist in the repository; this documentation file now serves as the code-faithful experimental design reference.
- The main Monte Carlo loop in [scripts/stress_test.mjs](/Users/mac/Documents/New project/scripts/stress_test.mjs) does not vary `proposerSelectionMode`; only the validation subroutine does.
- [scripts/example_step_session.mjs](/Users/mac/Documents/New project/scripts/example_step_session.mjs) does not pass proposer-selection configuration into `createInitialState`, even though the step API supports it.

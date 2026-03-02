# Adaptive Repeated Ultimatum Game (Andersen et al., 2011 Grounded)

This project implements a repeated Ultimatum Game web experiment with adaptive bots and online belief updates.

Detailed code-level documentation:
- `TECHNICAL_DOCS.md`

Modes:
- `human_responder_vs_bot_proposer` (Mode 1)
- `human_proposer_vs_bot_responder` (Mode 2)

It is calibrated using:
- Paper: Andersen et al. (2011), "Stakes Matter in Ultimatum Games"
- Data: `data/20100982_DATA.csv`

## Core Design

- Session has `R` rounds (default `10`, configurable).
- Treatment parameters are fixed within session:
  - stake in `{20, 200, 2000, 20000}`
  - wealth in `{0, 1}`
- Offer input supports rupees or percent toggle.
- Every round logs beliefs and rationales.
- Final CSV download includes:
  - `round, mode, stake, wealth, offer_amount, offer_share, accept_reject, bot_action, bot_beliefs, inferred_type, decision_rationale, expected_accept_prob, expected_value, cumulative_payoffs`

## Adaptive Bot Model

Latent types:
- `money_maximizer`
- `fairness_sensitive`
- `stake_sensitive`
- `noisy`

Belief update each round:
- Posterior update via likelihood weighting:
  - `posterior(type) ∝ prior(type) * P(observed_action | type, context)`
- Context includes stake, wealth, offer share, and round.

Default policy is belief-based and explainable:
- Bot proposer maximizes expected value under current responder-type beliefs.
- Bot responder uses belief-weighted acceptance plus proposer-type inference from observed offers.
- Every bot action emits a rationale string.
- `policy_mode` switch:
  - `belief` (default): logit-based responder likelihood component
  - `nn` (optional): MLP-based responder likelihood component from `data/nn_model.json`
- In `nn` mode, adaptation is still active:
  - beliefs are still updated every round via Bayesian likelihood weighting
  - beliefs still drive future proposer/responder decisions
- Belief updates use Bayesian likelihood with:
  - `beliefLearningRate` (default `0.85`) for gradual updates
  - `beliefPullback` (default `0.02`) toward initial priors to avoid collapse
- Explicit tremble noise is enabled:
  - proposer tremble probability (default `0.03`)
  - responder tremble probability (default `0.03`)

## Calibration

`scripts/fit_logit.mjs` fits:

`P(reject) = logistic(β0 + β1*wealth + β2*stake200 + β3*stake2000 + β4*stake20000 + β5*offer_share)`

Estimated coefficients and inferred priors are saved to:
- `data/fitted_params.json`

Run:

```bash
node scripts/fit_logit.mjs
```

Optional args:

```bash
node scripts/fit_logit.mjs --data=data/20100982_DATA.csv --out=data/fitted_params.json
```

Optional NN responder likelihood model:

`scripts/train_nn.mjs` trains a tiny MLP with input features:
- `[offer_share, log(stake), wealth]`

Run:

```bash
node scripts/train_nn.mjs
```

Optional args:

```bash
node scripts/train_nn.mjs --data=data/20100982_DATA.csv --out=data/nn_model.json --hidden=8 --epochs=5000 --lr=0.02 --seed=20260218
```

## Run Web Experiment Locally

From project root:

```bash
python -m http.server 8000
```

Open:
- `http://localhost:8000`

Debug panel (in UI) displays:
- current belief distribution over latent types
- current inferred type
- proposer-mode offer grid with expected acceptance and expected value

## Headless Stress Test (Reproducible)

`scripts/stress_test.mjs` runs deterministic simulations with seeded RNG.

Default behavior:
- 1000 sessions per treatment cell
- both modes
- stake levels `{20,200,2000,20000}`
- wealth levels `{0,1}`
- 10 rounds/session

Run:

```bash
node scripts/stress_test.mjs
```

Optional args:

```bash
node scripts/stress_test.mjs --sessions=1000 --rounds=10 --seed=20260218 --wealth=both --stakes=20,200,2000,20000 --outDir=outputs --policyMode=belief
```

Outputs:
- `outputs/stress_sessions.csv`
- `outputs/stress_summary.csv`

## Pure Logic API (oTree-ready)

`/src/step_api.mjs` provides a DOM-free, serializable logic interface:

- `createInitialState(config, fittedParams)` -> JSON state
- `step(state, human_action)` -> `{ new_state, bot_action, log_row }`
- `stateLogsToCsv(stateOrRows)` -> CSV string

Mode 1 (`human_responder_vs_bot_proposer`) uses a two-stage pattern:
1. `step(state, null)` to get the bot offer (`bot_action`) and pending state.
2. `step(new_state, { decision: "accept" | "reject" })` to resolve the round.

Mode 2 (`human_proposer_vs_bot_responder`) is single-stage:
- `step(state, { offer_share: 0.25 })` or `step(state, { offer_amount: 50 })`

All state fields are plain JSON values, suitable for persistence in oTree.

### Full-session step() example

Run:

```bash
node scripts/example_step_session.mjs --mode=mode1 --policyMode=belief --rounds=10 --stake=200 --wealth=1
```

Or NN mode:

```bash
node scripts/example_step_session.mjs --mode=mode2 --policyMode=nn --rounds=10 --stake=200 --wealth=1
```

## File Structure

- `/Users/mac/Documents/New project/TECHNICAL_DOCS.md`
- `/Users/mac/Documents/New project/index.html`
- `/Users/mac/Documents/New project/styles.css`
- `/Users/mac/Documents/New project/app.js`
- `/Users/mac/Documents/New project/src/data_loader.mjs`
- `/Users/mac/Documents/New project/src/belief_model.mjs`
- `/Users/mac/Documents/New project/src/types.mjs`
- `/Users/mac/Documents/New project/src/bot_proposer_policy.mjs`
- `/Users/mac/Documents/New project/src/bot_responder_policy.mjs`
- `/Users/mac/Documents/New project/src/payoff_engine.mjs`
- `/Users/mac/Documents/New project/src/logger.mjs`
- `/Users/mac/Documents/New project/src/session_engine.mjs`
- `/Users/mac/Documents/New project/src/sim_agents.mjs`
- `/Users/mac/Documents/New project/src/step_api.mjs`
- `/Users/mac/Documents/New project/data/20100982_DATA.csv`
- `/Users/mac/Documents/New project/data/fitted_params.json`
- `/Users/mac/Documents/New project/data/nn_model.json`
- `/Users/mac/Documents/New project/scripts/fit_logit.mjs`
- `/Users/mac/Documents/New project/scripts/train_nn.mjs`
- `/Users/mac/Documents/New project/scripts/example_step_session.mjs`
- `/Users/mac/Documents/New project/scripts/stress_test.mjs`
- `/Users/mac/Documents/New project/src/nn_inference.mjs`

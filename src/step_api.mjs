import { updateProposerBeliefs, updateResponderBeliefs, createInitialBeliefs } from "./belief_model.mjs";
import { chooseBotOffer, buildOfferGrid } from "./bot_proposer_policy.mjs";
import { decideBotResponse } from "./bot_responder_policy.mjs";
import { inferAmountFromShare, inferShareFromAmount } from "./data_loader.mjs";
import {
  AUTOMATON_TYPES,
  DEFAULT_AUTOMATON_MIX,
  DEFAULT_BETA_MODE,
  DEFAULT_FIXED_BETA,
  sampleAutomatonType,
  sampleFsBetaOnce,
} from "./automaton_profile.mjs";
import { MODES, resolvePayoffs } from "./payoff_engine.mjs";
import {
  DEFAULT_FITTED_PARAMS,
  LITERATURE_PRIOR_2_RESPONDER,
  PROPOSER_TYPES,
  RESPONDER_TYPES,
} from "./types.mjs";
import { argmaxBelief, roundTo } from "./utils.mjs";

function csvEscape(value) {
  const stringValue = value === null || value === undefined ? "" : String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

function normalizePolicyMode(mode) {
  return mode === "nn" ? "nn" : "belief";
}

function normalizeConfig(config = {}) {
  const offerStepShare = Number(config.offerStepShare ?? 0.025);
  return {
    mode: config.mode ?? MODES.HUMAN_RESPONDER,
    policyMode: normalizePolicyMode(config.policyMode),
    priorsSource: config.priorsSource === "calibration" ? "calibration" : "literature",
    rounds: Math.max(1, Number(config.rounds ?? 10)),
    stake: Number(config.stake ?? 200),
    wealth: config.wealth === 0 ? 0 : 1,
    proposerEpsilon: Number(config.proposerEpsilon ?? 0.08),
    proposerTemperature: Number(config.proposerTemperature ?? 0.1),
    proposerTrembleProb: Number(config.proposerTrembleProb ?? 0.03),
    proposerSelectionMode: config.proposerSelectionMode ?? "softmax",
    proposerNormalSigmaSteps: Number(config.proposerNormalSigmaSteps ?? 1.5),
    responderTemperature: Number(config.responderTemperature ?? 0.08),
    responderTrembleProb: Number(config.responderTrembleProb ?? 0.03),
    beliefLearningRate: Number(config.beliefLearningRate ?? 0.85),
    beliefPullback: Number(config.beliefPullback ?? 0.02),
    automatonMix: {
      ...DEFAULT_AUTOMATON_MIX,
      ...(config.automatonMix ?? {}),
    },
    betaMode: config.betaMode ?? DEFAULT_BETA_MODE,
    fixedBeta: Number(config.fixedBeta ?? DEFAULT_FIXED_BETA),
    offerStepShare,
    offerGrid: Array.isArray(config.offerGrid)
      ? config.offerGrid.map((value) => Number(value))
      : buildOfferGrid({ stepShare: offerStepShare }),
    seed: Number.isFinite(Number(config.seed)) ? Number(config.seed) : 20260218,
  };
}

function normalizeFittedParams(rawParams = DEFAULT_FITTED_PARAMS) {
  if (!rawParams) {
    return { ...DEFAULT_FITTED_PARAMS };
  }
  if (rawParams.coefficients) {
    return {
      ...DEFAULT_FITTED_PARAMS,
      ...rawParams.coefficients,
      priors: rawParams.priors ?? DEFAULT_FITTED_PARAMS.priors,
      nnModel: rawParams.nnModel ?? DEFAULT_FITTED_PARAMS.nnModel,
    };
  }
  return {
    ...DEFAULT_FITTED_PARAMS,
    ...rawParams,
    priors: rawParams.priors ?? DEFAULT_FITTED_PARAMS.priors,
    nnModel: rawParams.nnModel ?? DEFAULT_FITTED_PARAMS.nnModel,
  };
}

function normalizeStatePolicyMode(state) {
  if (state.config.policyMode === "nn" && !state.fitted_params.nnModel) {
    return "belief";
  }
  return state.config.policyMode;
}

function makeRngFromSeed(seed) {
  const seedHolder = {
    value: (Number(seed) >>> 0) || 20260218,
  };
  const rng = () => {
    seedHolder.value = (Math.imul(seedHolder.value, 1664525) + 1013904223) >>> 0;
    return seedHolder.value / 4294967296;
  };
  return { rng, seedHolder };
}

function parseAcceptReject(humanAction) {
  if (humanAction === null || humanAction === undefined) {
    return null;
  }
  if (typeof humanAction === "boolean") {
    return humanAction;
  }
  if (typeof humanAction === "string") {
    const normalized = humanAction.trim().toLowerCase();
    if (["accept", "a", "yes", "1", "true"].includes(normalized)) {
      return true;
    }
    if (["reject", "r", "no", "0", "false"].includes(normalized)) {
      return false;
    }
  }
  if (typeof humanAction === "object") {
    if (typeof humanAction.accepted === "boolean") {
      return humanAction.accepted;
    }
    if (typeof humanAction.accept === "boolean") {
      return humanAction.accept;
    }
    if (typeof humanAction.decision === "string") {
      return parseAcceptReject(humanAction.decision);
    }
    if (typeof humanAction.action === "string") {
      return parseAcceptReject(humanAction.action);
    }
  }
  throw new Error("Invalid human responder action. Use accept/reject.");
}

function parseHumanOffer(humanAction, stake) {
  if (humanAction === null || humanAction === undefined || typeof humanAction !== "object") {
    throw new Error("Human proposer action must include offer amount/share.");
  }

  if (Number.isFinite(Number(humanAction.offer_amount))) {
    const offerAmount = Math.max(0, Math.min(stake, Math.round(Number(humanAction.offer_amount))));
    const offerShare = inferShareFromAmount(offerAmount, stake);
    return { offerAmount, offerShare };
  }
  if (Number.isFinite(Number(humanAction.offerAmount))) {
    const offerAmount = Math.max(0, Math.min(stake, Math.round(Number(humanAction.offerAmount))));
    const offerShare = inferShareFromAmount(offerAmount, stake);
    return { offerAmount, offerShare };
  }

  if (Number.isFinite(Number(humanAction.offer_share))) {
    const share = Math.max(0, Math.min(1, Number(humanAction.offer_share)));
    const offerAmount = inferAmountFromShare(share, stake);
    return { offerAmount, offerShare: inferShareFromAmount(offerAmount, stake) };
  }
  if (Number.isFinite(Number(humanAction.offerShare))) {
    const share = Math.max(0, Math.min(1, Number(humanAction.offerShare)));
    const offerAmount = inferAmountFromShare(share, stake);
    return { offerAmount, offerShare: inferShareFromAmount(offerAmount, stake) };
  }
  if (Number.isFinite(Number(humanAction.offer_percent))) {
    const share = Math.max(0, Math.min(1, Number(humanAction.offer_percent) / 100));
    const offerAmount = inferAmountFromShare(share, stake);
    return { offerAmount, offerShare: inferShareFromAmount(offerAmount, stake) };
  }
  if (Number.isFinite(Number(humanAction.offerPercent))) {
    const share = Math.max(0, Math.min(1, Number(humanAction.offerPercent) / 100));
    const offerAmount = inferAmountFromShare(share, stake);
    return { offerAmount, offerShare: inferShareFromAmount(offerAmount, stake) };
  }

  throw new Error("Invalid human proposer action. Provide offer_amount or offer_share.");
}

function createDefaultStateDebug(config, beliefs, botProfile) {
  const inferred =
    config.mode === MODES.HUMAN_RESPONDER
      ? argmaxBelief(beliefs.human_responder).type
      : argmaxBelief(beliefs.human_proposer).type;
  return {
    mode: config.mode,
    policyMode: config.policyMode,
    round: 1,
    inferredType: inferred,
    expectedAcceptProb: null,
    proposerGrid: [],
    proposerSelectionMode: config.proposerSelectionMode,
    selectedProb: null,
    bot_profile: { ...botProfile },
    beliefs:
      config.mode === MODES.HUMAN_RESPONDER
        ? { ...beliefs.human_responder }
        : { ...beliefs.human_proposer },
  };
}

export function createInitialState(config = {}, fittedParams = DEFAULT_FITTED_PARAMS) {
  const normalizedConfig = normalizeConfig(config);
  const normalizedFittedParams = normalizeFittedParams(fittedParams);
  if (normalizedConfig.policyMode === "nn" && !normalizedFittedParams.nnModel) {
    normalizedConfig.policyMode = "belief";
  }
  const { rng, seedHolder } = makeRngFromSeed(normalizedConfig.seed);
  const automatonType = sampleAutomatonType(normalizedConfig.automatonMix, rng);
  const betaMode = normalizedConfig.betaMode;
  const beta =
    automatonType === AUTOMATON_TYPES.MAXIMIZER
      ? 0
      : sampleFsBetaOnce(betaMode, normalizedConfig.fixedBeta, rng);
  const botProfile = {
    automatonType,
    betaMode,
    beta,
  };
  const responderPriorConfig =
    normalizedConfig.priorsSource === "calibration"
      ? normalizedFittedParams.priors?.responder
      : LITERATURE_PRIOR_2_RESPONDER;

  const anchorHumanResponderBeliefs = createInitialBeliefs(
    RESPONDER_TYPES,
    responderPriorConfig
  );
  const anchorHumanProposerBeliefs = createInitialBeliefs(
    PROPOSER_TYPES,
    normalizedFittedParams.priors?.proposer
  );
  const beliefs = {
    human_responder: { ...anchorHumanResponderBeliefs },
    human_proposer: { ...anchorHumanProposerBeliefs },
    bot_responder: createInitialBeliefs(
      RESPONDER_TYPES,
      responderPriorConfig
    ),
  };

  const state = {
    schema_version: "ultimatum_step_state_v1",
    config: normalizedConfig,
    fitted_params: normalizedFittedParams,
    round_index: 1,
    complete: false,
    cumulative_payoffs: {
      human: 0,
      bot: 0,
    },
    anchors: {
      human_responder: anchorHumanResponderBeliefs,
      human_proposer: anchorHumanProposerBeliefs,
    },
    bot_profile: botProfile,
    beliefs,
    pending_offer: null,
    logs: [],
    last_debug: createDefaultStateDebug(normalizedConfig, beliefs, botProfile),
    rng_seed: seedHolder.value,
  };

  return JSON.parse(JSON.stringify(state));
}

function buildLogRow(entry) {
  return {
    round: entry.round,
    mode: entry.mode,
    stake: entry.stake,
    wealth: entry.wealth,
    offer_amount: entry.offer_amount,
    offer_share: entry.offer_share,
    accept_reject: entry.accept_reject,
    bot_action: entry.bot_action,
    bot_beliefs: entry.bot_beliefs,
    inferred_type: entry.inferred_type,
    decision_rationale: entry.decision_rationale,
    expected_accept_prob: entry.expected_accept_prob,
    expected_value: entry.expected_value,
    cumulative_payoffs: entry.cumulative_payoffs,
  };
}

function mode1GeneratePending(state, rng) {
  const config = state.config;
  const offerDecision = chooseBotOffer({
    stake: config.stake,
    wealth: config.wealth,
    roundIndex: state.round_index,
    responderBeliefs: state.beliefs.human_responder,
    fittedParams: state.fitted_params,
    policyMode: normalizeStatePolicyMode(state),
    automatonType: state.bot_profile?.automatonType ?? AUTOMATON_TYPES.MAXIMIZER,
    betaUsed: Number(state.bot_profile?.beta ?? 0),
    offerGrid: config.offerGrid,
    epsilon: config.proposerEpsilon,
    temperature: config.proposerTemperature,
    trembleProb: config.proposerTrembleProb,
    selectionMode: config.proposerSelectionMode,
    normalSigmaSteps: config.proposerNormalSigmaSteps,
    offerStepShare: Number(config.offerStepShare ?? 0.025),
    rng,
  });

  return {
    round: state.round_index,
    offerAmount: offerDecision.offerAmount,
    offerShare: offerDecision.offerShare,
    expectedAcceptProb: offerDecision.expectedAcceptProb,
    expectedValue: offerDecision.expectedValue,
    inferredType: offerDecision.inferredType,
    rationale: offerDecision.rationale,
    proposerSelectionMode: offerDecision.selectionMode ?? config.proposerSelectionMode,
    selectedProb: offerDecision.selectedProb ?? null,
    proposerGrid: offerDecision.debug.candidateEvaluations.map((item) => ({
      offerAmount: item.offerAmount,
      offerShare: item.offerShare,
      acceptProb: item.acceptProb,
      expectedAcceptProb: item.expectedAcceptProb ?? item.acceptProb,
      expectedValue: item.expectedValue,
      ev_max: item.ev_max,
      eu_fs_current: item.eu_fs_current,
      eu_fs_beta025: item.eu_fs_beta025,
      eu_fs_beta060: item.eu_fs_beta060,
      utility_raw_fs_current: item.utility_raw_fs_current,
      betaUsed: item.betaUsed,
    })),
  };
}

function mode1OfferBotAction(pending, state) {
  return {
    role: "bot_proposer",
    action: "offer",
    offer_amount: pending.offerAmount,
    offer_share: pending.offerShare,
    expected_accept_prob: pending.expectedAcceptProb,
    expected_value: pending.expectedValue,
    inferred_type: pending.inferredType,
    decision_rationale: pending.rationale,
    proposer_selection_mode: pending.proposerSelectionMode ?? state.config.proposerSelectionMode,
    selected_prob: pending.selectedProb,
    policy_mode: normalizeStatePolicyMode(state),
    bot_profile: { ...(state.bot_profile ?? {}) },
    offer_grid: pending.proposerGrid,
  };
}

function mode1ResolveRound(state, pending, humanActionAccepted) {
  const config = state.config;
  const policyMode = normalizeStatePolicyMode(state);
  const context = {
    offerShare: pending.offerShare,
    stake: config.stake,
    wealth: config.wealth,
    roundIndex: state.round_index,
  };
  const beliefUpdate = updateResponderBeliefs({
    priorBeliefs: state.beliefs.human_responder,
    accepted: humanActionAccepted,
    context,
    fittedParams: state.fitted_params,
    policyMode,
    learningRate: config.beliefLearningRate,
    pullback: config.beliefPullback,
    pullbackPrior: state.anchors.human_responder,
  });

  const payoffResult = resolvePayoffs({
    mode: config.mode,
    stake: config.stake,
    offerAmount: pending.offerAmount,
    accepted: humanActionAccepted,
    cumulativePayoffs: state.cumulative_payoffs,
  });

  const posteriorComment = `Observed human ${
    humanActionAccepted ? "accept" : "reject"
  }; posterior now ${beliefUpdate.inferred.type} (${roundTo(
    beliefUpdate.inferred.probability,
    2
  )}).`;
  const rationale = `${pending.rationale} ${posteriorComment}`;
  const logRow = buildLogRow({
    round: state.round_index,
    mode: config.mode,
    stake: config.stake,
    wealth: config.wealth,
    offer_amount: pending.offerAmount,
    offer_share: roundTo(pending.offerShare, 4),
    accept_reject: humanActionAccepted ? "accept" : "reject",
    bot_action: `offer_${pending.offerAmount}`,
    bot_beliefs: JSON.stringify(beliefUpdate.posterior),
    inferred_type: beliefUpdate.inferred.type,
    decision_rationale: rationale,
    expected_accept_prob: roundTo(pending.expectedAcceptProb, 4),
    expected_value: roundTo(pending.expectedValue, 4),
    cumulative_payoffs: JSON.stringify(payoffResult.cumulative),
  });

  const nextRound = state.round_index + 1;
  const newState = {
    ...state,
    config: { ...config, policyMode },
    round_index: nextRound,
    complete: nextRound > config.rounds,
    pending_offer: null,
    cumulative_payoffs: payoffResult.cumulative,
    beliefs: {
      ...state.beliefs,
      human_responder: beliefUpdate.posterior,
    },
    logs: [...state.logs, logRow],
    last_debug: {
      mode: config.mode,
      policyMode,
      round: state.round_index,
      inferredType: beliefUpdate.inferred.type,
      expectedAcceptProb: pending.expectedAcceptProb,
      proposerGrid: pending.proposerGrid,
      proposerSelectionMode: pending.proposerSelectionMode ?? config.proposerSelectionMode,
      selectedProb: pending.selectedProb ?? null,
      bot_profile: { ...(state.bot_profile ?? {}) },
      beliefs: beliefUpdate.posterior,
    },
  };

  const botAction = {
    role: "bot_proposer",
    action: "offer_resolved",
    offer_amount: pending.offerAmount,
    offer_share: pending.offerShare,
    proposer_selection_mode: pending.proposerSelectionMode ?? config.proposerSelectionMode,
    selected_prob: pending.selectedProb,
    inferred_type: beliefUpdate.inferred.type,
    decision_rationale: rationale,
    policy_mode: policyMode,
    bot_profile: { ...(state.bot_profile ?? {}) },
  };

  return {
    new_state: JSON.parse(JSON.stringify(newState)),
    bot_action: botAction,
    log_row: logRow,
  };
}

function mode2ResolveRound(state, humanAction, rng) {
  const config = state.config;
  const policyMode = normalizeStatePolicyMode(state);
  const parsedOffer = parseHumanOffer(humanAction, config.stake);
  const proposerUpdate = updateProposerBeliefs({
    priorBeliefs: state.beliefs.human_proposer,
    observedOfferShare: parsedOffer.offerShare,
    context: {
      stake: config.stake,
      wealth: config.wealth,
      roundIndex: state.round_index,
    },
    learningRate: config.beliefLearningRate,
    pullback: config.beliefPullback,
    pullbackPrior: state.anchors.human_proposer,
  });

  const botDecision = decideBotResponse({
    offerShare: parsedOffer.offerShare,
    offerAmount: parsedOffer.offerAmount,
    stake: config.stake,
    wealth: config.wealth,
    roundIndex: state.round_index,
    proposerBeliefs: proposerUpdate.posterior,
    responderBeliefs: state.beliefs.bot_responder,
    fittedParams: state.fitted_params,
    policyMode,
    temperature: config.responderTemperature,
    trembleProb: config.responderTrembleProb,
    rng,
  });

  const payoffResult = resolvePayoffs({
    mode: config.mode,
    stake: config.stake,
    offerAmount: parsedOffer.offerAmount,
    accepted: botDecision.accepted,
    cumulativePayoffs: state.cumulative_payoffs,
  });

  const proposerPosteriorComment = `Observed human offer ${roundTo(
    parsedOffer.offerShare * 100,
    1
  )}%; inferred proposer type ${proposerUpdate.inferred.type} (${roundTo(
    proposerUpdate.inferred.probability,
    2
  )}).`;
  const rationale = `${proposerPosteriorComment} ${botDecision.rationale}`;
  const logRow = buildLogRow({
    round: state.round_index,
    mode: config.mode,
    stake: config.stake,
    wealth: config.wealth,
    offer_amount: parsedOffer.offerAmount,
    offer_share: roundTo(parsedOffer.offerShare, 4),
    accept_reject: botDecision.accepted ? "accept" : "reject",
    bot_action: botDecision.botAction,
    bot_beliefs: JSON.stringify(proposerUpdate.posterior),
    inferred_type: proposerUpdate.inferred.type,
    decision_rationale: rationale,
    expected_accept_prob: roundTo(botDecision.expectedAcceptProb, 4),
    expected_value: roundTo(botDecision.expectedValue, 4),
    cumulative_payoffs: JSON.stringify(payoffResult.cumulative),
  });

  const nextRound = state.round_index + 1;
  const newState = {
    ...state,
    config: { ...config, policyMode },
    round_index: nextRound,
    complete: nextRound > config.rounds,
    pending_offer: null,
    cumulative_payoffs: payoffResult.cumulative,
    beliefs: {
      ...state.beliefs,
      human_proposer: proposerUpdate.posterior,
    },
    logs: [...state.logs, logRow],
    last_debug: {
      mode: config.mode,
      policyMode,
      round: state.round_index,
      inferredType: proposerUpdate.inferred.type,
      expectedAcceptProb: botDecision.expectedAcceptProb,
      proposerGrid: [],
      proposerSelectionMode: config.proposerSelectionMode,
      selectedProb: null,
      bot_profile: { ...(state.bot_profile ?? {}) },
      beliefs: proposerUpdate.posterior,
    },
  };

  const botAction = {
    role: "bot_responder",
    action: botDecision.botAction,
    accepted: botDecision.accepted,
    expected_accept_prob: botDecision.expectedAcceptProb,
    expected_value: botDecision.expectedValue,
    inferred_type: proposerUpdate.inferred.type,
    decision_rationale: rationale,
    policy_mode: policyMode,
    bot_profile: { ...(state.bot_profile ?? {}) },
  };

  return {
    new_state: JSON.parse(JSON.stringify(newState)),
    bot_action: botAction,
    log_row: logRow,
  };
}

export function step(state, humanAction) {
  if (!state || typeof state !== "object") {
    throw new Error("State must be a JSON-serializable object.");
  }
  if (state.complete) {
    return {
      new_state: JSON.parse(JSON.stringify(state)),
      bot_action: null,
      log_row: null,
    };
  }

  const { rng, seedHolder } = makeRngFromSeed(state.rng_seed);
  const policyMode = normalizeStatePolicyMode(state);

  if (state.config.mode === MODES.HUMAN_RESPONDER) {
    let pending = state.pending_offer;
    let generatedNow = false;
    if (!pending) {
      pending = mode1GeneratePending(state, rng);
      generatedNow = true;
    }

    const actionAccepted = parseAcceptReject(humanAction);
    if (actionAccepted === null) {
      const newState = {
        ...state,
        config: { ...state.config, policyMode },
        pending_offer: pending,
        last_debug: {
          mode: state.config.mode,
          policyMode,
          round: state.round_index,
          inferredType: pending.inferredType,
          expectedAcceptProb: pending.expectedAcceptProb,
          proposerGrid: pending.proposerGrid,
          proposerSelectionMode:
            pending.proposerSelectionMode ?? state.config.proposerSelectionMode,
          selectedProb: pending.selectedProb ?? null,
          bot_profile: { ...(state.bot_profile ?? {}) },
          beliefs: state.beliefs.human_responder,
        },
        rng_seed: generatedNow ? seedHolder.value : state.rng_seed,
      };
      return {
        new_state: JSON.parse(JSON.stringify(newState)),
        bot_action: mode1OfferBotAction(pending, newState),
        log_row: null,
      };
    }

    const resolved = mode1ResolveRound(state, pending, actionAccepted);
    resolved.new_state.rng_seed = generatedNow ? seedHolder.value : state.rng_seed;
    resolved.new_state.config.policyMode = policyMode;
    return resolved;
  }

  const resolved = mode2ResolveRound(
    {
      ...state,
      config: { ...state.config, policyMode },
    },
    humanAction,
    rng
  );
  resolved.new_state.rng_seed = seedHolder.value;
  return resolved;
}

export function stateLogsToCsv(stateOrRows) {
  const rows = Array.isArray(stateOrRows) ? stateOrRows : stateOrRows?.logs ?? [];
  const headers = [
    "round",
    "mode",
    "stake",
    "wealth",
    "offer_amount",
    "offer_share",
    "accept_reject",
    "bot_action",
    "bot_beliefs",
    "inferred_type",
    "decision_rationale",
    "expected_accept_prob",
    "expected_value",
    "cumulative_payoffs",
  ];
  const body = rows.map((row) => headers.map((header) => csvEscape(row[header])).join(","));
  return [headers.join(","), ...body].join("\n");
}

import {
  PROPOSER_TYPES,
  RESPONDER_TYPES,
  proposerOfferLikelihood,
  responderAcceptProbability,
} from "./types.mjs";
import { argmaxBelief, normalizeBeliefMap } from "./utils.mjs";

function defaultUniformBeliefs(typeNames) {
  const value = 1 / typeNames.length;
  return Object.fromEntries(typeNames.map((type) => [type, value]));
}

export function createInitialBeliefs(typeNames, priors = null) {
  if (!priors) {
    return defaultUniformBeliefs(typeNames);
  }
  const candidate = Object.fromEntries(
    typeNames.map((type) => [type, Number(priors[type] ?? 0)])
  );
  return normalizeBeliefMap(candidate);
}

function blendBeliefs(baseBeliefs, targetBeliefs, weightOnTarget) {
  const weight = Math.max(0, Math.min(1, Number(weightOnTarget)));
  const merged = {};
  const allTypes = new Set([...Object.keys(baseBeliefs), ...Object.keys(targetBeliefs)]);
  for (const type of allTypes) {
    const base = Number(baseBeliefs[type] ?? 0);
    const target = Number(targetBeliefs[type] ?? 0);
    merged[type] = (1 - weight) * base + weight * target;
  }
  return normalizeBeliefMap(merged);
}

export function bayesUpdate(priorBeliefs, likelihoodByType, options = {}) {
  const learningRate = Math.max(0, Math.min(1, Number(options.learningRate ?? 1)));
  const pullback = Math.max(0, Math.min(0.5, Number(options.pullback ?? 0)));
  const pullbackPrior = normalizeBeliefMap(
    options.pullbackPrior ?? Object.fromEntries(Object.keys(priorBeliefs).map((k) => [k, 1]))
  );

  const unnormalized = {};
  for (const [type, prior] of Object.entries(priorBeliefs)) {
    const likelihood = Math.max(Number(likelihoodByType[type] ?? 0), 1e-9);
    unnormalized[type] = Math.max(prior, 1e-9) * likelihood;
  }

  const purePosterior = normalizeBeliefMap(unnormalized);
  const learnedPosterior = blendBeliefs(priorBeliefs, purePosterior, learningRate);
  return blendBeliefs(learnedPosterior, pullbackPrior, pullback);
}

export function updateResponderBeliefs({
  priorBeliefs,
  accepted,
  context,
  fittedParams,
  policyMode = "belief",
  learningRate = 1,
  pullback = 0,
  pullbackPrior = null,
}) {
  const likelihoodByType = {};
  for (const type of RESPONDER_TYPES) {
    const acceptProb = responderAcceptProbability(type, context, fittedParams, {
      policyMode,
    });
    likelihoodByType[type] = accepted ? acceptProb : 1 - acceptProb;
  }
  const posterior = bayesUpdate(priorBeliefs, likelihoodByType, {
    learningRate,
    pullback,
    pullbackPrior,
  });
  return {
    posterior,
    likelihoodByType,
    inferred: argmaxBelief(posterior),
  };
}

export function updateProposerBeliefs({
  priorBeliefs,
  observedOfferShare,
  context,
  learningRate = 1,
  pullback = 0,
  pullbackPrior = null,
}) {
  const enrichedContext = { ...context, offerShare: observedOfferShare };
  const likelihoodByType = {};
  for (const type of PROPOSER_TYPES) {
    likelihoodByType[type] = proposerOfferLikelihood(type, enrichedContext);
  }
  const posterior = bayesUpdate(priorBeliefs, likelihoodByType, {
    learningRate,
    pullback,
    pullbackPrior,
  });
  return {
    posterior,
    likelihoodByType,
    inferred: argmaxBelief(posterior),
  };
}

export function beliefWeightedAcceptProbability({
  responderBeliefs,
  context,
  fittedParams,
  policyMode = "belief",
}) {
  let probability = 0;
  for (const type of RESPONDER_TYPES) {
    const belief = responderBeliefs[type] ?? 0;
    probability += belief * responderAcceptProbability(type, context, fittedParams, { policyMode });
  }
  return Math.min(0.999, Math.max(0.001, probability));
}

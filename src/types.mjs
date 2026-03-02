import { clamp, gaussianPdf, sigmoid } from "./utils.mjs";
import { predictRejectProbability } from "./nn_inference.mjs";

export const TARGET_STAKES = [20, 200, 2000, 20000];

export const LITERATURE_PRIOR_2_RESPONDER = {
  money_maximizer: 0.13,
  fairness_sensitive: 0.78,
  stake_sensitive: 0.05,
  noisy: 0.04,
};

export const RESPONDER_TYPES = [
  "money_maximizer",
  "fairness_sensitive",
  "stake_sensitive",
  "noisy",
];

export const PROPOSER_TYPES = [
  "money_maximizer",
  "fairness_sensitive",
  "stake_sensitive",
  "noisy",
];

export const DEFAULT_FITTED_PARAMS = {
  beta0: 0.55,
  beta1: -0.2,
  beta2: -0.15,
  beta3: -0.4,
  beta4: -0.7,
  beta5: -4.2,
  nnModel: null,
  priors: {
    responder: { ...LITERATURE_PRIOR_2_RESPONDER },
    proposer: {
      money_maximizer: 0.25,
      fairness_sensitive: 0.25,
      stake_sensitive: 0.25,
      noisy: 0.25,
    },
  },
};

function stakeBucket(stake) {
  if (stake <= 110) {
    return 20;
  }
  if (stake <= 1100) {
    return 200;
  }
  if (stake <= 11000) {
    return 2000;
  }
  return 20000;
}

function getStakeDummies(stake) {
  const bucket = stakeBucket(stake);
  return {
    stake200: bucket === 200 ? 1 : 0,
    stake2000: bucket === 2000 ? 1 : 0,
    stake20000: bucket === 20000 ? 1 : 0,
  };
}

export function baselineRejectProbability(
  context,
  fittedParams = DEFAULT_FITTED_PARAMS,
  options = {}
) {
  const policyMode = options.policyMode === "nn" ? "nn" : "belief";
  if (policyMode === "nn" && fittedParams?.nnModel) {
    const nnPrediction = predictRejectProbability(fittedParams.nnModel, context);
    if (Number.isFinite(nnPrediction)) {
      return clamp(nnPrediction, 0.001, 0.999);
    }
  }

  const offerShare = clamp(context.offerShare ?? 0, 0, 1);
  const wealth = context.wealth === 1 ? 1 : 0;
  const dummies = getStakeDummies(context.stake ?? 20);
  const z =
    (fittedParams.beta0 ?? DEFAULT_FITTED_PARAMS.beta0) +
    (fittedParams.beta1 ?? DEFAULT_FITTED_PARAMS.beta1) * wealth +
    (fittedParams.beta2 ?? DEFAULT_FITTED_PARAMS.beta2) * dummies.stake200 +
    (fittedParams.beta3 ?? DEFAULT_FITTED_PARAMS.beta3) * dummies.stake2000 +
    (fittedParams.beta4 ?? DEFAULT_FITTED_PARAMS.beta4) * dummies.stake20000 +
    (fittedParams.beta5 ?? DEFAULT_FITTED_PARAMS.beta5) * offerShare;
  return clamp(sigmoid(z), 0.001, 0.999);
}

export function responderAcceptProbability(
  type,
  context,
  fittedParams = DEFAULT_FITTED_PARAMS,
  options = {}
) {
  const offerShare = clamp(context.offerShare ?? 0, 0, 1);
  const stake = context.stake ?? 20;
  const wealth = context.wealth === 1 ? 1 : 0;
  const baselineAccept =
    1 - baselineRejectProbability({ ...context, offerShare }, fittedParams, options);
  const stakeLogScale = Math.log10(stake / 20 + 1);
  const roundIndex = Math.max(1, context.roundIndex ?? 1);

  if (type === "money_maximizer") {
    if (offerShare <= 0) {
      return 0.02;
    }
    return clamp(0.98 + 0.015 * baselineAccept, 0.02, 0.999);
  }

  if (type === "fairness_sensitive") {
    const threshold = clamp(0.29 + (wealth === 0 ? 0.03 : -0.01), 0.2, 0.4);
    const fairnessAccept = sigmoid((offerShare - threshold) * 24);
    return clamp(0.7 * fairnessAccept + 0.3 * baselineAccept, 0.001, 0.999);
  }

  if (type === "stake_sensitive") {
    const dynamicThreshold = clamp(0.34 - 0.07 * stakeLogScale, 0.1, 0.36);
    const fatigue = clamp((roundIndex - 1) * 0.006, 0, 0.05);
    const stakeAccept = sigmoid((offerShare - (dynamicThreshold - fatigue)) * 20);
    return clamp(0.75 * stakeAccept + 0.25 * baselineAccept, 0.001, 0.999);
  }

  if (type === "noisy") {
    const trembleAccept = 0.5;
    return clamp(0.6 * baselineAccept + 0.4 * trembleAccept, 0.05, 0.95);
  }

  return baselineAccept;
}

export function proposerTargetShare(type, context) {
  const stake = context.stake ?? 20;
  const wealth = context.wealth === 1 ? 1 : 0;
  const stakeLogScale = Math.log10(stake / 20 + 1);

  if (type === "money_maximizer") {
    return clamp(0.08 + (wealth === 0 ? 0.02 : 0), 0.02, 0.2);
  }

  if (type === "fairness_sensitive") {
    return clamp(0.4 + (wealth === 0 ? 0.02 : -0.02), 0.25, 0.5);
  }

  if (type === "stake_sensitive") {
    return clamp(0.34 - 0.055 * stakeLogScale, 0.12, 0.36);
  }

  if (type === "noisy") {
    return 0.25;
  }

  return 0.25;
}

export function proposerOfferLikelihood(type, context) {
  const offerShare = clamp(context.offerShare ?? 0, 0, 1);
  if (type === "noisy") {
    return 1;
  }
  const mean = proposerTargetShare(type, context);
  const sigma =
    type === "money_maximizer" ? 0.06 : type === "fairness_sensitive" ? 0.08 : 0.09;
  return Math.max(gaussianPdf(offerShare, mean, sigma), 1e-6);
}

import { PROPOSER_TYPES, RESPONDER_TYPES, proposerTargetShare, responderAcceptProbability } from "./types.mjs";
import { clamp, sampleCategorical } from "./utils.mjs";

function sampleType(typeNames, priorBeliefs, rng = Math.random) {
  const probabilities = typeNames.map((type) => priorBeliefs[type] ?? 0);
  const index = sampleCategorical(probabilities, rng);
  return typeNames[index];
}

function sampleGaussian(rng = Math.random) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function sampleHumanResponderType(priors, rng = Math.random) {
  return sampleType(RESPONDER_TYPES, priors, rng);
}

export function sampleHumanProposerType(priors, rng = Math.random) {
  return sampleType(PROPOSER_TYPES, priors, rng);
}

export function humanResponderDecision({
  responderType,
  context,
  fittedParams,
  rng = Math.random,
}) {
  const acceptProb = responderAcceptProbability(responderType, context, fittedParams);
  const tremble = rng() < 0.02;
  const accepted = tremble ? rng() < 0.5 : rng() < acceptProb;
  return {
    accepted,
    acceptProb,
  };
}

export function humanProposerOffer({
  proposerType,
  context,
  rng = Math.random,
}) {
  const meanShare = proposerTargetShare(proposerType, context);
  const sigma =
    proposerType === "money_maximizer"
      ? 0.04
      : proposerType === "fairness_sensitive"
      ? 0.06
      : proposerType === "stake_sensitive"
      ? 0.07
      : 0.18;
  const sampledShare = clamp(meanShare + sampleGaussian(rng) * sigma, 0, 0.5);
  return sampledShare;
}

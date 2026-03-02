import { beliefWeightedAcceptProbability } from "./belief_model.mjs";
import { proposerTargetShare } from "./types.mjs";
import { argmaxBelief, clamp, formatShare, roundTo, sigmoid } from "./utils.mjs";

function expectedOfferFromProposerBeliefs(proposerBeliefs, context) {
  let expectedShare = 0;
  for (const [type, belief] of Object.entries(proposerBeliefs)) {
    expectedShare += belief * proposerTargetShare(type, context);
  }
  return clamp(expectedShare, 0, 1);
}

export function decideBotResponse({
  offerShare,
  offerAmount,
  stake,
  wealth,
  roundIndex,
  proposerBeliefs,
  responderBeliefs,
  fittedParams,
  policyMode = "belief",
  temperature = 0.08,
  trembleProb = 0.03,
  rng = Math.random,
}) {
  const context = { offerShare, stake, wealth, roundIndex };
  const baseAcceptProb = beliefWeightedAcceptProbability({
    responderBeliefs,
    context,
    fittedParams,
    policyMode,
  });

  const expectedOfferShare = expectedOfferFromProposerBeliefs(proposerBeliefs, context);
  const selfishWeight = proposerBeliefs.money_maximizer ?? 0;
  const fairWeight = proposerBeliefs.fairness_sensitive ?? 0;
  const stakeSensitiveWeight = proposerBeliefs.stake_sensitive ?? 0;
  const stakeScale = clamp(Math.log10(stake / 20 + 1) / 3, 0, 1);

  const offerSignal = offerShare - expectedOfferShare;
  const reputationAdjustment =
    0.25 * offerSignal + 0.08 * fairWeight - 0.12 * selfishWeight + 0.04 * stakeSensitiveWeight * stakeScale;
  const adjustedAcceptProb = clamp(baseAcceptProb + reputationAdjustment, 0.001, 0.999);

  const decisionProbability = sigmoid((adjustedAcceptProb - 0.5) / Math.max(temperature, 1e-4));
  let accepted = rng() < decisionProbability;
  const trembleUsed = rng() < trembleProb;
  if (trembleUsed) {
    accepted = !accepted;
  }
  const topBelief = argmaxBelief(proposerBeliefs);
  const coreRationale = `Belief ${roundTo(topBelief.probability, 2)} ${topBelief.type}; offer ${formatShare(
    offerShare
  )} vs expected ${formatShare(expectedOfferShare)} -> accept prob ${roundTo(adjustedAcceptProb, 2)}; ${
    accepted ? "accept" : "reject"
  }.`;
  const rationale = trembleUsed
    ? `${coreRationale} Tremble (${roundTo(trembleProb, 2)}) flipped baseline response.`
    : coreRationale;

  return {
    accepted,
    botAction: accepted ? "accept" : "reject",
    expectedAcceptProb: adjustedAcceptProb,
    expectedValue: adjustedAcceptProb * offerAmount,
    inferredType: topBelief.type,
    rationale,
    debug: {
      baseAcceptProb,
      expectedOfferShare,
      decisionProbability,
      reputationAdjustment,
      trembleUsed,
    },
  };
}

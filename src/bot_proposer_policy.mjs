import { beliefWeightedAcceptProbability } from "./belief_model.mjs";
import { argmaxBelief, clamp, formatShare, roundCurrency, roundTo, sampleCategorical, softmax } from "./utils.mjs";

export function buildOfferGrid({
  minShare = 0,
  maxShare = 0.5,
  stepShare = 0.025,
} = {}) {
  const grid = [];
  const safeStep = Math.max(stepShare, 0.001);
  for (let share = minShare; share <= maxShare + 1e-9; share += safeStep) {
    grid.push(roundTo(clamp(share, 0, 1), 4));
  }
  return grid;
}

export function chooseBotOffer({
  stake,
  wealth,
  roundIndex,
  responderBeliefs,
  fittedParams,
  policyMode = "belief",
  automatonType = "maximizer",
  betaUsed = 0,
  offerGrid = buildOfferGrid(),
  epsilon = 0.08,
  temperature = 0.1,
  trembleProb = 0.03,
  rng = Math.random,
}) {
  const fsBeta = Number.isFinite(Number(betaUsed)) ? Number(betaUsed) : 0;
  const candidateEvaluations = offerGrid.map((gridShare) => {
    const offerAmount = roundCurrency(stake * gridShare);
    const offerShare = clamp(offerAmount / stake, 0, 1);
    const acceptProb = beliefWeightedAcceptProbability({
      responderBeliefs,
      fittedParams,
      policyMode,
      context: {
        offerShare,
        stake,
        wealth,
        roundIndex,
      },
    });
    const ev_max = acceptProb * (stake - offerAmount);
    const utility_raw_fs_current =
      (stake - offerAmount) - fsBeta * (stake - 2 * offerAmount);
    const eu_fs_current = acceptProb * utility_raw_fs_current;
    const eu_fs_beta025 =
      acceptProb * ((stake - offerAmount) - 0.25 * (stake - 2 * offerAmount));
    const eu_fs_beta060 =
      acceptProb * ((stake - offerAmount) - 0.6 * (stake - 2 * offerAmount));
    const expectedValue =
      automatonType === "maximizer" ? ev_max : eu_fs_current;
    const debugMetrics = {
      acceptProb,
      ev_max,
      eu_fs_current,
      eu_fs_beta025,
      eu_fs_beta060,
      utility_raw_fs_current,
      betaUsed: fsBeta,
    };
    return {
      offerAmount,
      offerShare,
      ...debugMetrics,
      expectedAcceptProb: acceptProb,
      expectedValue,
    };
  });

  const expectedValues = candidateEvaluations.map((candidate) => candidate.expectedValue);
  const maxExpectedValue = Math.max(...expectedValues);
  const bestCandidate = candidateEvaluations.find(
    (candidate) => candidate.expectedValue === maxExpectedValue
  );

  let selectedIndex = 0;
  let trembleUsed = false;
  let explorationUsed = false;
  if (rng() < trembleProb) {
    selectedIndex = Math.floor(rng() * candidateEvaluations.length);
    trembleUsed = true;
  } else if (rng() < epsilon) {
    selectedIndex = Math.floor(rng() * candidateEvaluations.length);
    explorationUsed = true;
  } else {
    const weights = softmax(expectedValues, temperature);
    selectedIndex = sampleCategorical(weights, rng);
  }

  const selected = candidateEvaluations[selectedIndex];
  const topBelief = argmaxBelief(responderBeliefs);
  const isFairnessSensitive = automatonType === "fairness_sensitive";
  const objectiveLabel = isFairnessSensitive ? `EU (β=${roundTo(fsBeta, 4)})` : "EV";
  const selectedObjective = isFairnessSensitive
    ? selected.expectedValue
    : (selected.ev_max ?? selected.expectedValue);
  const bestObjective = maxExpectedValue;
  const rationale = `Highest posterior type: ${topBelief.type} (${Number(
    topBelief.probability
  ).toFixed(2)}). Mixture-based P(accept)=${Number(selected.acceptProb).toFixed(4)}. Offer ${formatShare(
    selected.offerShare
  )} yields ${objectiveLabel}=${Number(selectedObjective).toFixed(2)} (best ${objectiveLabel}=${Number(
    bestObjective
  ).toFixed(2)}). Decision uses full belief distribution.`;

  return {
    ...selected,
    botAction: `propose_${selected.offerAmount}`,
    expectedAcceptProb: selected.acceptProb,
    expectedValue: selected.expectedValue,
    inferredType: topBelief.type,
    rationale,
    debug: {
      bestCandidate,
      candidateEvaluations,
      trembleUsed,
      explorationUsed,
    },
  };
}

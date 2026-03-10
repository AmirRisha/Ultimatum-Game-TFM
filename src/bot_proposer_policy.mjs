import { beliefWeightedAcceptProbability } from "./belief_model.mjs";
import {
  argmaxBelief,
  clamp,
  formatShare,
  gaussianPdf,
  roundCurrency,
  roundTo,
  sampleCategorical,
  softmax,
} from "./utils.mjs";

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
  selectionMode = "softmax",
  normalSigmaSteps = 1.5,
  offerStepShare = 0.025,
  trembleProb = 0.03,
  rng = Math.random,
}) {
  const uniformProbabilities = (length) => {
    if (length <= 0) {
      return [];
    }
    const uniform = 1 / length;
    return Array.from({ length }, () => uniform);
  };

  const normalizeWeights = (weights) => {
    const safeWeights = weights.map((weight) =>
      Number.isFinite(Number(weight)) ? Math.max(0, Number(weight)) : 0
    );
    const total = safeWeights.reduce((sum, weight) => sum + weight, 0);
    if (!Number.isFinite(total) || total <= 0) {
      return uniformProbabilities(safeWeights.length);
    }
    return safeWeights.map((weight) => weight / total);
  };

  const inferOfferStepShare = (shares, fallbackStep) => {
    if (!Array.isArray(shares) || shares.length < 2) {
      return Math.max(Number(fallbackStep) || 0.025, 1e-6);
    }
    const sorted = [...shares]
      .map((share) => Number(share))
      .filter((share) => Number.isFinite(share))
      .sort((a, b) => a - b);
    if (sorted.length < 2) {
      return Math.max(Number(fallbackStep) || 0.025, 1e-6);
    }
    let minPositiveDiff = Infinity;
    for (let i = 1; i < sorted.length; i += 1) {
      const diff = sorted[i] - sorted[i - 1];
      if (diff > 1e-9 && diff < minPositiveDiff) {
        minPositiveDiff = diff;
      }
    }
    if (!Number.isFinite(minPositiveDiff)) {
      return Math.max(Number(fallbackStep) || 0.025, 1e-6);
    }
    return Math.max(minPositiveDiff, 1e-6);
  };

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

  const scores = candidateEvaluations.map((candidate) => Number(candidate.expectedValue));
  const safeScores = scores.map((score) => (Number.isFinite(score) ? score : 0));
  const bestIndex = safeScores.reduce(
    (best, score, index) => (score > safeScores[best] ? index : best),
    0
  );
  const maxExpectedValue = safeScores[bestIndex];
  const bestCandidate = candidateEvaluations[bestIndex];

  const resolvedSelectionMode = [
    "softmax",
    "proportional_ev",
    "normal_around_best",
  ].includes(selectionMode)
    ? selectionMode
    : "softmax";

  // Exploitation distribution by configured proposer selection mode.
  let selectionProbabilities;
  if (resolvedSelectionMode === "proportional_ev") {
    const minScore = Math.min(...safeScores);
    const shiftedScores = safeScores.map((score) => score - minScore + 1e-9);
    selectionProbabilities = normalizeWeights(shiftedScores);
  } else if (resolvedSelectionMode === "normal_around_best") {
    const mu = Number(candidateEvaluations[bestIndex]?.offerShare ?? 0);
    const gridStep = inferOfferStepShare(
      candidateEvaluations.map((candidate) => candidate.offerShare),
      offerStepShare
    );
    const sigma = Math.max(Number(normalSigmaSteps) || 1.5, 1e-6) * Math.max(gridStep, 1e-6);
    const gaussianWeights = candidateEvaluations.map((candidate) =>
      gaussianPdf(Number(candidate.offerShare), mu, sigma)
    );
    selectionProbabilities = normalizeWeights(gaussianWeights);
  } else {
    selectionProbabilities = normalizeWeights(softmax(safeScores, temperature));
  }

  let selectedIndex = 0;
  let trembleUsed = false;
  let explorationUsed = false;
  let selectedProb = null;
  const uniformSelectionProb = candidateEvaluations.length > 0 ? 1 / candidateEvaluations.length : 0;
  // Keep tremble/epsilon semantics unchanged: both override exploitation with uniform random pick.
  if (rng() < trembleProb) {
    selectedIndex = Math.floor(rng() * candidateEvaluations.length);
    trembleUsed = true;
    selectedProb = uniformSelectionProb;
  } else if (rng() < epsilon) {
    selectedIndex = Math.floor(rng() * candidateEvaluations.length);
    explorationUsed = true;
    selectedProb = uniformSelectionProb;
  } else {
    selectedIndex = sampleCategorical(selectionProbabilities, rng);
    selectedProb = Number(selectionProbabilities[selectedIndex] ?? 0);
  }

  const candidateEvaluationsWithSelection = candidateEvaluations.map((candidate, index) => ({
    ...candidate,
    scoreUsed: safeScores[index],
    selectionProb: Number(selectionProbabilities[index] ?? 0),
  }));
  const selected = candidateEvaluationsWithSelection[selectedIndex];
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
    selectionMode: resolvedSelectionMode,
    selectedProb,
    inferredType: topBelief.type,
    rationale,
    debug: {
      bestCandidate,
      candidateEvaluations: candidateEvaluationsWithSelection,
      selectionMode: resolvedSelectionMode,
      trembleUsed,
      explorationUsed,
    },
  };
}

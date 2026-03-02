import { sampleCategorical } from "./utils.mjs";

export const AUTOMATON_TYPES = {
  MAXIMIZER: "maximizer",
  FAIRNESS: "fairness_sensitive",
};

export const BETA_MODES = {
  A_FIXED: "A",
  B_DRAW_ONCE: "B",
};

export const FS_BETA_VALUES = [0.25, 0.6];

export const FS_BETA_WEIGHTS = [0.4286, 0.5714];

export const DEFAULT_AUTOMATON_MIX = {
  maximizer: 0.5,
  fairness_sensitive: 0.5,
};

export const DEFAULT_BETA_MODE = "B";

export const DEFAULT_FIXED_BETA = 0.6;

export function sampleAutomatonType(mix, rng = Math.random) {
  const probabilities = [
    Number(mix?.[AUTOMATON_TYPES.MAXIMIZER] ?? 0),
    Number(mix?.[AUTOMATON_TYPES.FAIRNESS] ?? 0),
  ];
  const index = sampleCategorical(probabilities, rng);
  return index === 0 ? AUTOMATON_TYPES.MAXIMIZER : AUTOMATON_TYPES.FAIRNESS;
}

export function sampleFsBetaOnce(betaMode, fixedBeta = DEFAULT_FIXED_BETA, rng = Math.random) {
  if (betaMode === BETA_MODES.A_FIXED) {
    return Number(fixedBeta);
  }
  if (betaMode === BETA_MODES.B_DRAW_ONCE) {
    const index = sampleCategorical(FS_BETA_WEIGHTS, rng);
    return FS_BETA_VALUES[index];
  }
  return Number(fixedBeta);
}

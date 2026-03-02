export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function softmax(values, temperature = 1) {
  const safeTemperature = Math.max(temperature, 1e-6);
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp((value - maxValue) / safeTemperature));
  const total = exps.reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    const uniform = 1 / values.length;
    return values.map(() => uniform);
  }
  return exps.map((value) => value / total);
}

export function sampleCategorical(probabilities, rng = Math.random) {
  const draw = rng();
  let cumulative = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index];
    if (draw <= cumulative || index === probabilities.length - 1) {
      return index;
    }
  }
  return probabilities.length - 1;
}

export function roundTo(value, decimals = 4) {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

export function roundCurrency(value) {
  return Math.round(value);
}

export function formatShare(share) {
  return `${roundTo(share * 100, 1)}%`;
}

export function normalizeBeliefMap(beliefs) {
  const entries = Object.entries(beliefs);
  const total = entries.reduce((sum, [, value]) => sum + Math.max(0, value), 0);
  if (total <= 0) {
    const uniform = 1 / entries.length;
    return Object.fromEntries(entries.map(([key]) => [key, uniform]));
  }
  return Object.fromEntries(entries.map(([key, value]) => [key, Math.max(0, value) / total]));
}

export function argmaxBelief(beliefs) {
  let bestKey = "";
  let bestValue = -Infinity;
  for (const [key, value] of Object.entries(beliefs)) {
    if (value > bestValue) {
      bestValue = value;
      bestKey = key;
    }
  }
  return { type: bestKey, probability: bestValue };
}

export function mulberry32(seed) {
  let t = seed >>> 0;
  return function random() {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianPdf(x, mean, sigma) {
  const safeSigma = Math.max(sigma, 1e-6);
  const exponent = -((x - mean) ** 2) / (2 * safeSigma ** 2);
  return (1 / (safeSigma * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
}

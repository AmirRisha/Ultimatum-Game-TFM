import { clamp, sigmoid } from "./utils.mjs";

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += a[i] * b[i];
  }
  return total;
}

function tanh(x) {
  const e1 = Math.exp(x);
  const e2 = Math.exp(-x);
  return (e1 - e2) / (e1 + e2);
}

function ensureStd(value) {
  const candidate = Number(value);
  if (!Number.isFinite(candidate) || Math.abs(candidate) < 1e-8) {
    return 1;
  }
  return candidate;
}

export function buildNnFeatureVector(context) {
  const offerShare = clamp(Number(context.offerShare ?? 0), 0, 1);
  const logStake = Math.log(Math.max(Number(context.stake ?? 20), 1));
  const wealth = context.wealth === 1 ? 1 : 0;
  return [offerShare, logStake, wealth];
}

export function predictRejectProbability(nnModel, context) {
  if (!nnModel) {
    return null;
  }
  const input = buildNnFeatureVector(context);
  const means = nnModel.normalization?.mean ?? [0, 0, 0];
  const stds = nnModel.normalization?.std ?? [1, 1, 1];
  const normalized = input.map((value, index) => (value - Number(means[index] ?? 0)) / ensureStd(stds[index]));

  const w1 = nnModel.weights?.w1 ?? [];
  const b1 = nnModel.weights?.b1 ?? [];
  const w2 = nnModel.weights?.w2 ?? [];
  const b2 = nnModel.weights?.b2 ?? [0];
  if (w1.length === 0 || w2.length === 0) {
    return null;
  }

  const hidden = w1.map((row, index) => tanh(dot(row, normalized) + Number(b1[index] ?? 0)));
  const logits = w2.map((row, index) => dot(row, hidden) + Number(b2[index] ?? 0));
  const rejectProb = sigmoid(logits[0] ?? 0);
  return clamp(rejectProb, 0.001, 0.999);
}

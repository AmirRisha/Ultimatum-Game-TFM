import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { calibrationRows, parseCsvText, rejectModelFeatures } from "../src/data_loader.mjs";
import { PROPOSER_TYPES, RESPONDER_TYPES, proposerOfferLikelihood, responderAcceptProbability } from "../src/types.mjs";
import { roundTo, sigmoid } from "../src/utils.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {};
  for (const token of argv) {
    if (!token.startsWith("--")) {
      continue;
    }
    const [key, value] = token.slice(2).split("=");
    args[key] = value === undefined ? true : value;
  }
  return args;
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += a[i] * b[i];
  }
  return total;
}

function fitLogisticRegression(X, y, options = {}) {
  const iterations = Number(options.iterations ?? 12000);
  const learningRate = Number(options.learningRate ?? 0.08);
  const l2 = Number(options.l2 ?? 1e-4);
  const featureCount = X[0].length;
  const beta = Array(featureCount).fill(0);
  const gradSq = Array(featureCount).fill(1e-6);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const gradient = Array(featureCount).fill(0);
    for (let rowIndex = 0; rowIndex < X.length; rowIndex += 1) {
      const row = X[rowIndex];
      const prediction = sigmoid(dot(beta, row));
      const error = y[rowIndex] - prediction;
      for (let j = 0; j < featureCount; j += 1) {
        gradient[j] += error * row[j];
      }
    }

    let maxStep = 0;
    for (let j = 0; j < featureCount; j += 1) {
      const n = X.length;
      const regularizer = j === 0 ? 0 : l2 * beta[j];
      const g = gradient[j] / n - regularizer;
      gradSq[j] += g * g;
      const step = (learningRate / Math.sqrt(gradSq[j])) * g;
      beta[j] += step;
      maxStep = Math.max(maxStep, Math.abs(step));
    }

    if (maxStep < 1e-8) {
      break;
    }
  }

  return beta;
}

function modelMetrics(X, y, beta) {
  let llModel = 0;
  const pMean = y.reduce((sum, v) => sum + v, 0) / y.length;
  let llNull = 0;
  for (let i = 0; i < X.length; i += 1) {
    const p = Math.min(1 - 1e-9, Math.max(1e-9, sigmoid(dot(beta, X[i]))));
    llModel += y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);

    const pn = Math.min(1 - 1e-9, Math.max(1e-9, pMean));
    llNull += y[i] * Math.log(pn) + (1 - y[i]) * Math.log(1 - pn);
  }

  const pseudoR2 = 1 - llModel / llNull;
  return {
    logLikModel: llModel,
    logLikNull: llNull,
    mcfaddenR2: pseudoR2,
  };
}

function normalizeWeights(weights) {
  const entries = Object.entries(weights);
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  if (total <= 0) {
    const uniform = 1 / entries.length;
    return Object.fromEntries(entries.map(([key]) => [key, uniform]));
  }
  return Object.fromEntries(entries.map(([key, value]) => [key, value / total]));
}

function inferResponderPriors(rows, fittedParams) {
  const weights = Object.fromEntries(RESPONDER_TYPES.map((type) => [type, 1e-6]));
  for (const row of rows) {
    const context = {
      offerShare: row.offerShare,
      stake: row.stake,
      wealth: row.wealth,
      roundIndex: 1,
    };
    for (const type of RESPONDER_TYPES) {
      const acceptProb = responderAcceptProbability(type, context, fittedParams);
      const likelihood = row.accept === 1 ? acceptProb : 1 - acceptProb;
      weights[type] += Math.max(likelihood, 1e-6);
    }
  }
  return normalizeWeights(weights);
}

function inferProposerPriors(rows) {
  const weights = Object.fromEntries(PROPOSER_TYPES.map((type) => [type, 1e-6]));
  for (const row of rows) {
    const context = {
      offerShare: row.offerShare,
      stake: row.stake,
      wealth: row.wealth,
      roundIndex: 1,
    };
    for (const type of PROPOSER_TYPES) {
      weights[type] += proposerOfferLikelihood(type, context);
    }
  }
  return normalizeWeights(weights);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dataPath = path.resolve(projectRoot, args.data ?? "data/20100982_DATA.csv");
  const outPath = path.resolve(projectRoot, args.out ?? "data/fitted_params.json");

  const csvText = await readFile(dataPath, "utf8");
  const rawRows = parseCsvText(csvText);
  const rows = calibrationRows(rawRows);
  if (rows.length < 20) {
    throw new Error(
      `Insufficient calibration rows (${rows.length}). Need more valid rows with wealth, stake dummies and offer share.`
    );
  }

  const X = rows.map((row) => rejectModelFeatures(row));
  const y = rows.map((row) => row.reject);
  const beta = fitLogisticRegression(X, y);
  const metrics = modelMetrics(X, y, beta);

  const fittedParams = {
    beta0: roundTo(beta[0], 6),
    beta1: roundTo(beta[1], 6),
    beta2: roundTo(beta[2], 6),
    beta3: roundTo(beta[3], 6),
    beta4: roundTo(beta[4], 6),
    beta5: roundTo(beta[5], 6),
  };
  const priors = {
    responder: inferResponderPriors(rows, fittedParams),
    proposer: inferProposerPriors(rows),
  };

  const output = {
    source_data: path.relative(projectRoot, dataPath),
    model: "logit_reject ~ wealth + stake200 + stake2000 + stake20000 + offer_share",
    sample_size: rows.length,
    fitted_at: new Date().toISOString(),
    coefficients: fittedParams,
    priors,
    fit_metrics: {
      log_likelihood_model: roundTo(metrics.logLikModel, 4),
      log_likelihood_null: roundTo(metrics.logLikNull, 4),
      mcfadden_r2: roundTo(metrics.mcfaddenR2, 4),
    },
  };
  await writeFile(outPath, JSON.stringify(output, null, 2));

  process.stdout.write(`Fitted logit on ${rows.length} rows.\n`);
  process.stdout.write(`Saved coefficients to ${path.relative(projectRoot, outPath)}\n`);
  process.stdout.write(`beta = [${beta.map((x) => roundTo(x, 4)).join(", ")}]\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});

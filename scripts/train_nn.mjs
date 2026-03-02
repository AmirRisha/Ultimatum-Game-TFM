import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { nnTrainingRows, parseCsvText } from "../src/data_loader.mjs";
import { clamp, mulberry32, roundTo, sigmoid } from "../src/utils.mjs";

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

function gaussian(rng = Math.random) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeMatrix(rows, cols, initializer) {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => initializer(r, c))
  );
}

function zerosLikeMatrix(matrix) {
  return matrix.map((row) => row.map(() => 0));
}

function zerosLikeVector(vector) {
  return vector.map(() => 0);
}

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

function tanhDerivative(tanhValue) {
  return 1 - tanhValue ** 2;
}

function computeNormalization(X) {
  const featureCount = X[0].length;
  const mean = Array(featureCount).fill(0);
  const std = Array(featureCount).fill(0);
  for (const row of X) {
    for (let j = 0; j < featureCount; j += 1) {
      mean[j] += row[j];
    }
  }
  for (let j = 0; j < featureCount; j += 1) {
    mean[j] /= X.length;
  }
  for (const row of X) {
    for (let j = 0; j < featureCount; j += 1) {
      std[j] += (row[j] - mean[j]) ** 2;
    }
  }
  for (let j = 0; j < featureCount; j += 1) {
    std[j] = Math.sqrt(std[j] / X.length);
    if (std[j] < 1e-8) {
      std[j] = 1;
    }
  }
  return { mean, std };
}

function normalizeX(X, normalization) {
  return X.map((row) =>
    row.map((value, j) => (value - normalization.mean[j]) / normalization.std[j])
  );
}

function forwardSample(model, x) {
  const { w1, b1, w2, b2 } = model;
  const hiddenPre = w1.map((row, h) => dot(row, x) + b1[h]);
  const hidden = hiddenPre.map((value) => tanh(value));
  const logit = dot(w2[0], hidden) + b2[0];
  const pReject = clamp(sigmoid(logit), 1e-6, 1 - 1e-6);
  return { hidden, logit, pReject };
}

function evaluate(X, y, model) {
  let loss = 0;
  let correct = 0;
  for (let i = 0; i < X.length; i += 1) {
    const pred = forwardSample(model, X[i]).pReject;
    loss += -(y[i] * Math.log(pred) + (1 - y[i]) * Math.log(1 - pred));
    const predictedLabel = pred >= 0.5 ? 1 : 0;
    if (predictedLabel === y[i]) {
      correct += 1;
    }
  }
  return {
    loss: loss / X.length,
    accuracy: correct / X.length,
  };
}

function trainMlp(X, y, options = {}) {
  const rng = mulberry32(Number(options.seed ?? 20260218));
  const inputSize = X[0].length;
  const hiddenSize = Number(options.hidden ?? 8);
  const epochs = Number(options.epochs ?? 5000);
  const lr = Number(options.lr ?? 0.02);
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const l2 = Number(options.l2 ?? 1e-4);

  const model = {
    w1: makeMatrix(hiddenSize, inputSize, () => gaussian(rng) * 0.12),
    b1: Array(hiddenSize).fill(0),
    w2: makeMatrix(1, hiddenSize, () => gaussian(rng) * 0.12),
    b2: [0],
  };

  const m = {
    w1: zerosLikeMatrix(model.w1),
    b1: zerosLikeVector(model.b1),
    w2: zerosLikeMatrix(model.w2),
    b2: zerosLikeVector(model.b2),
  };
  const v = {
    w1: zerosLikeMatrix(model.w1),
    b1: zerosLikeVector(model.b1),
    w2: zerosLikeMatrix(model.w2),
    b2: zerosLikeVector(model.b2),
  };

  const updateAdam = (param, grad, mBuf, vBuf, step) => {
    if (Array.isArray(param[0])) {
      for (let i = 0; i < param.length; i += 1) {
        for (let j = 0; j < param[i].length; j += 1) {
          mBuf[i][j] = beta1 * mBuf[i][j] + (1 - beta1) * grad[i][j];
          vBuf[i][j] = beta2 * vBuf[i][j] + (1 - beta2) * grad[i][j] ** 2;
          const mHat = mBuf[i][j] / (1 - beta1 ** step);
          const vHat = vBuf[i][j] / (1 - beta2 ** step);
          param[i][j] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      }
      return;
    }

    for (let i = 0; i < param.length; i += 1) {
      mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * grad[i];
      vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * grad[i] ** 2;
      const mHat = mBuf[i] / (1 - beta1 ** step);
      const vHat = vBuf[i] / (1 - beta2 ** step);
      param[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  };

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const grad = {
      w1: zerosLikeMatrix(model.w1),
      b1: zerosLikeVector(model.b1),
      w2: zerosLikeMatrix(model.w2),
      b2: zerosLikeVector(model.b2),
    };

    for (let i = 0; i < X.length; i += 1) {
      const x = X[i];
      const target = y[i];
      const { hidden, pReject } = forwardSample(model, x);
      const dLogit = pReject - target;

      for (let h = 0; h < model.w2[0].length; h += 1) {
        grad.w2[0][h] += dLogit * hidden[h];
      }
      grad.b2[0] += dLogit;

      for (let h = 0; h < model.w1.length; h += 1) {
        const dHidden = model.w2[0][h] * dLogit;
        const dPre = dHidden * tanhDerivative(hidden[h]);
        for (let j = 0; j < model.w1[h].length; j += 1) {
          grad.w1[h][j] += dPre * x[j];
        }
        grad.b1[h] += dPre;
      }
    }

    const n = X.length;
    for (let h = 0; h < model.w1.length; h += 1) {
      for (let j = 0; j < model.w1[h].length; j += 1) {
        grad.w1[h][j] = grad.w1[h][j] / n + l2 * model.w1[h][j];
      }
      grad.b1[h] /= n;
    }
    for (let h = 0; h < model.w2[0].length; h += 1) {
      grad.w2[0][h] = grad.w2[0][h] / n + l2 * model.w2[0][h];
    }
    grad.b2[0] /= n;

    updateAdam(model.w1, grad.w1, m.w1, v.w1, epoch);
    updateAdam(model.b1, grad.b1, m.b1, v.b1, epoch);
    updateAdam(model.w2, grad.w2, m.w2, v.w2, epoch);
    updateAdam(model.b2, grad.b2, m.b2, v.b2, epoch);
  }

  return model;
}

function buildFeatures(rows) {
  return rows.map((row) => [row.offerShare, Math.log(row.stake), row.wealth]);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dataPath = path.resolve(projectRoot, args.data ?? "data/20100982_DATA.csv");
  const outPath = path.resolve(projectRoot, args.out ?? "data/nn_model.json");
  const hidden = Number(args.hidden ?? 8);
  const epochs = Number(args.epochs ?? 5000);
  const lr = Number(args.lr ?? 0.02);
  const l2 = Number(args.l2 ?? 0.0001);
  const seed = Number(args.seed ?? 20260218);

  const csvText = await readFile(dataPath, "utf8");
  const rawRows = parseCsvText(csvText);
  const rows = nnTrainingRows(rawRows);
  if (rows.length < 30) {
    throw new Error(`Insufficient rows for NN training (${rows.length}).`);
  }

  const X = buildFeatures(rows);
  const y = rows.map((row) => row.reject);
  const normalization = computeNormalization(X);
  const Xn = normalizeX(X, normalization);
  const model = trainMlp(Xn, y, { hidden, epochs, lr, l2, seed });
  const metrics = evaluate(Xn, y, model);

  const out = {
    source_data: path.relative(projectRoot, dataPath),
    input_features: ["offer_share", "log_stake", "wealth"],
    architecture: {
      input: 3,
      hidden,
      output: 1,
      activation: "tanh",
      output_activation: "sigmoid",
    },
    normalization: {
      mean: normalization.mean.map((value) => roundTo(value, 8)),
      std: normalization.std.map((value) => roundTo(value, 8)),
    },
    weights: {
      w1: model.w1.map((row) => row.map((value) => roundTo(value, 10))),
      b1: model.b1.map((value) => roundTo(value, 10)),
      w2: model.w2.map((row) => row.map((value) => roundTo(value, 10))),
      b2: model.b2.map((value) => roundTo(value, 10)),
    },
    training: {
      rows: rows.length,
      epochs,
      learning_rate: lr,
      l2,
      seed,
      loss: roundTo(metrics.loss, 6),
      accuracy: roundTo(metrics.accuracy, 6),
    },
    fitted_at: new Date().toISOString(),
  };

  await writeFile(outPath, JSON.stringify(out, null, 2));
  process.stdout.write(`Trained NN on ${rows.length} rows.\n`);
  process.stdout.write(`Saved model to ${path.relative(projectRoot, outPath)}\n`);
  process.stdout.write(
    `Train loss=${roundTo(metrics.loss, 4)} accuracy=${roundTo(metrics.accuracy, 4)}\n`
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});

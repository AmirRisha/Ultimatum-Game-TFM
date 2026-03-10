import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { MODES } from "../src/payoff_engine.mjs";
import { RepeatedUltimatumSession } from "../src/session_engine.mjs";
import { humanProposerOffer, humanResponderDecision, sampleHumanProposerType, sampleHumanResponderType } from "../src/sim_agents.mjs";
import { TARGET_STAKES } from "../src/types.mjs";
import { mulberry32, roundTo } from "../src/utils.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const PROPOSER_SELECTION_MODES = ["softmax", "proportional_ev", "normal_around_best"];

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

function csvEscape(value) {
  const stringValue = value === null || value === undefined ? "" : String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

function toCsv(rows, headers) {
  const lines = rows.map((row) => headers.map((header) => csvEscape(row[header])).join(","));
  return [headers.join(","), ...lines].join("\n");
}

function aggregateBy(rows, keys, metrics) {
  const groups = new Map();
  for (const row of rows) {
    const key = keys.map((k) => row[k]).join("|");
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(row);
  }

  const output = [];
  for (const [groupKey, groupedRows] of groups.entries()) {
    const sample = groupedRows[0];
    const summary = Object.fromEntries(keys.map((k) => [k, sample[k]]));
    summary.group_key = groupKey;
    summary.sessions = groupedRows.length;
    for (const metric of metrics) {
      const values = groupedRows.map((row) => Number(row[metric] ?? 0));
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
      summary[`mean_${metric}`] = roundTo(mean, 4);
      summary[`sd_${metric}`] = roundTo(Math.sqrt(variance), 4);
    }
    output.push(summary);
  }
  return output;
}

function parseWealthArg(wealthArg) {
  if (wealthArg === "0") {
    return [0];
  }
  if (wealthArg === "1") {
    return [1];
  }
  return [0, 1];
}

function parseStakesArg(stakesArg) {
  if (!stakesArg) {
    return TARGET_STAKES;
  }
  const parsed = stakesArg
    .split(",")
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && TARGET_STAKES.includes(v));
  if (parsed.length === 0) {
    return TARGET_STAKES;
  }
  return [...new Set(parsed)];
}

function assertFiniteMetric(value, label, context = {}) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || Number.isNaN(numeric)) {
    throw new Error(`Invalid numeric value for ${label}: ${value}. Context: ${JSON.stringify(context)}`);
  }
  return numeric;
}

function checkSelectionProbabilitySum(candidates = []) {
  const probabilities = candidates
    .map((candidate) => Number(candidate?.selectionProb))
    .filter((value) => Number.isFinite(value));
  if (probabilities.length === 0) {
    return { status: "not_logged", sum: null };
  }
  const sum = probabilities.reduce((acc, value) => acc + value, 0);
  const withinTolerance = Math.abs(sum - 1) <= 1e-4;
  if (!withinTolerance) {
    throw new Error(
      `Selection probabilities do not sum to 1 (sum=${sum}).`
    );
  }
  return { status: "ok", sum };
}

function validateProposerSelectionModes({
  fittedParams,
  priors,
  nnModel,
  policyMode,
  baseSeed,
}) {
  const rows = [];
  let validationCounter = 0;

  for (const stake of TARGET_STAKES) {
    for (const proposerSelectionMode of PROPOSER_SELECTION_MODES) {
      validationCounter += 1;
      const rounds = 5;
      const wealth = 1;
      const rng = mulberry32(baseSeed + 100000 + validationCounter * 31);
      const session = new RepeatedUltimatumSession(
        {
          mode: MODES.HUMAN_RESPONDER,
          policyMode,
          rounds,
          stake,
          wealth,
          proposerSelectionMode,
          proposerNormalSigmaSteps: 1.5,
        },
        {
          ...fittedParams.coefficients,
          priors,
          nnModel,
        },
        rng
      );

      let selectionProbStatus = "not_logged";
      let selectionProbSum = null;

      while (!session.isComplete()) {
        const pending = session.startRoundForHumanResponderMode();
        assertFiniteMetric(pending.expectedAcceptProb, "pending.expectedAcceptProb", {
          stake,
          proposerSelectionMode,
          round: pending.round,
        });
        assertFiniteMetric(pending.expectedValue, "pending.expectedValue", {
          stake,
          proposerSelectionMode,
          round: pending.round,
        });

        const debugState = session.getDebugState();
        const proposerGrid = Array.isArray(debugState?.proposerGrid)
          ? debugState.proposerGrid
          : Array.isArray(pending?.proposerGrid)
            ? pending.proposerGrid
            : [];
        for (const candidate of proposerGrid) {
          const acceptProbValue =
            candidate.expectedAcceptProb ?? candidate.acceptProb;
          assertFiniteMetric(acceptProbValue, "candidate.expectedAcceptProb", {
            stake,
            proposerSelectionMode,
            round: pending.round,
            offerShare: candidate.offerShare,
          });
          assertFiniteMetric(candidate.expectedValue, "candidate.expectedValue", {
            stake,
            proposerSelectionMode,
            round: pending.round,
            offerShare: candidate.offerShare,
          });
        }
        const probCheck = checkSelectionProbabilitySum(proposerGrid);
        if (probCheck.status === "ok") {
          selectionProbStatus = "ok";
          selectionProbSum = probCheck.sum;
        }

        const accepted = Number(pending.offerShare) >= 0.2;
        session.submitHumanResponse(accepted);
        const lastLog = session.getLogRecords().at(-1);
        assertFiniteMetric(lastLog?.expected_accept_prob, "log.expected_accept_prob", {
          stake,
          proposerSelectionMode,
          round: pending.round,
        });
        assertFiniteMetric(lastLog?.expected_value, "log.expected_value", {
          stake,
          proposerSelectionMode,
          round: pending.round,
        });
      }

      const logs = session.getLogRecords();
      if (logs.length !== rounds || !session.isComplete()) {
        throw new Error(
          `Validation session did not complete as expected for stake=${stake}, mode=${proposerSelectionMode}.`
        );
      }

      rows.push({
        stake,
        proposer_selection_mode: proposerSelectionMode,
        rounds,
        rounds_logged: logs.length,
        selection_prob_check: selectionProbStatus,
        selection_prob_sum:
          selectionProbSum === null ? "" : roundTo(selectionProbSum, 6),
      });
    }
  }

  return rows;
}

async function runStressTest() {
  const args = parseArgs(process.argv.slice(2));
  const sessionsPerCell = Number(args.sessions ?? 1000);
  const rounds = Number(args.rounds ?? 10);
  const baseSeed = Number(args.seed ?? 20260218);
  const wealthLevels = parseWealthArg(args.wealth ?? "both");
  const stakes = parseStakesArg(args.stakes ?? "");
  const outDir = path.resolve(projectRoot, args.outDir ?? "outputs");
  const fittedPath = path.resolve(projectRoot, args.params ?? "data/fitted_params.json");
  const requestedPolicyMode = args.policyMode === "nn" ? "nn" : "belief";
  const nnPath = path.resolve(projectRoot, args.nnModel ?? "data/nn_model.json");

  const fittedParams = JSON.parse(await readFile(fittedPath, "utf8"));
  const priors = fittedParams.priors ?? {};
  let policyMode = requestedPolicyMode;
  let nnModel = null;
  if (requestedPolicyMode === "nn") {
    try {
      nnModel = JSON.parse(await readFile(nnPath, "utf8"));
    } catch {
      policyMode = "belief";
      process.stdout.write(
        `Requested policyMode=nn but ${path.relative(projectRoot, nnPath)} not found. Falling back to belief.\n`
      );
    }
  }
  await mkdir(outDir, { recursive: true });

  const modeValues = [MODES.HUMAN_RESPONDER, MODES.HUMAN_PROPOSER];
  const sessionRows = [];
  let sessionCounter = 0;

  for (const mode of modeValues) {
    for (const wealth of wealthLevels) {
      for (const stake of stakes) {
        for (let run = 0; run < sessionsPerCell; run += 1) {
          sessionCounter += 1;
          const rng = mulberry32(baseSeed + sessionCounter * 17);
          const session = new RepeatedUltimatumSession(
            {
              mode,
              policyMode,
              rounds,
              stake,
              wealth,
            },
            {
              ...fittedParams.coefficients,
              priors,
              nnModel,
            },
            rng
          );

          let hiddenHumanType = "";
          if (mode === MODES.HUMAN_RESPONDER) {
            hiddenHumanType = sampleHumanResponderType(
              priors.responder ?? {
                money_maximizer: 0.25,
                fairness_sensitive: 0.25,
                stake_sensitive: 0.25,
                noisy: 0.25,
              },
              rng
            );
            while (!session.isComplete()) {
              const pending = session.startRoundForHumanResponderMode();
              const outcome = humanResponderDecision({
                responderType: hiddenHumanType,
                context: {
                  offerShare: pending.offerShare,
                  stake,
                  wealth,
                  roundIndex: pending.round,
                },
                fittedParams: {
                  ...fittedParams.coefficients,
                  priors,
                },
                rng,
              });
              session.submitHumanResponse(outcome.accepted);
            }
          } else {
            hiddenHumanType = sampleHumanProposerType(
              priors.proposer ?? {
                money_maximizer: 0.25,
                fairness_sensitive: 0.25,
                stake_sensitive: 0.25,
                noisy: 0.25,
              },
              rng
            );
            while (!session.isComplete()) {
              const context = {
                stake,
                wealth,
                roundIndex: session.getProgress().roundIndex,
              };
              const offerShare = humanProposerOffer({
                proposerType: hiddenHumanType,
                context,
                rng,
              });
              const offerAmount = Math.round(offerShare * stake);
              session.submitHumanOffer({ offerAmount, offerInputMode: "amount" });
            }
          }

          const logs = session.getLogRecords();
          const accepts = logs.filter((log) => log.accept_reject === "accept").length;
          const avgOfferShare =
            logs.reduce((sum, log) => sum + Number(log.offer_share), 0) / Math.max(logs.length, 1);

          sessionRows.push({
            session_id: sessionCounter,
            mode,
            policy_mode: policyMode,
            wealth,
            stake,
            rounds,
            hidden_human_type: hiddenHumanType,
            accept_rate: roundTo(accepts / Math.max(logs.length, 1), 4),
            avg_offer_share: roundTo(avgOfferShare, 4),
            human_payoff: roundTo(session.getProgress().cumulativePayoffs.human, 4),
            bot_payoff: roundTo(session.getProgress().cumulativePayoffs.bot, 4),
          });
        }
      }
    }
  }

  const summaryRows = aggregateBy(
    sessionRows,
    ["mode", "policy_mode", "wealth", "stake"],
    ["accept_rate", "avg_offer_share", "human_payoff", "bot_payoff"]
  );

  const sessionsCsv = toCsv(sessionRows, [
    "session_id",
    "mode",
    "policy_mode",
    "wealth",
    "stake",
    "rounds",
    "hidden_human_type",
    "accept_rate",
    "avg_offer_share",
    "human_payoff",
    "bot_payoff",
  ]);
  const summaryCsv = toCsv(summaryRows, [
    "group_key",
    "mode",
    "policy_mode",
    "wealth",
    "stake",
    "sessions",
    "mean_accept_rate",
    "sd_accept_rate",
    "mean_avg_offer_share",
    "sd_avg_offer_share",
    "mean_human_payoff",
    "sd_human_payoff",
    "mean_bot_payoff",
    "sd_bot_payoff",
  ]);

  const sessionsPath = path.join(outDir, "stress_sessions.csv");
  const summaryPath = path.join(outDir, "stress_summary.csv");
  await writeFile(sessionsPath, sessionsCsv);
  await writeFile(summaryPath, summaryCsv);

  const validationRows = validateProposerSelectionModes({
    fittedParams,
    priors,
    nnModel,
    policyMode,
    baseSeed,
  });
  const validationCsv = toCsv(validationRows, [
    "stake",
    "proposer_selection_mode",
    "rounds",
    "rounds_logged",
    "selection_prob_check",
    "selection_prob_sum",
  ]);
  const validationPath = path.join(outDir, "selection_mode_validation.csv");
  await writeFile(validationPath, validationCsv);

  process.stdout.write(`Stress test complete.\n`);
  process.stdout.write(`Sessions: ${sessionRows.length}\n`);
  process.stdout.write(`Wrote ${path.relative(projectRoot, sessionsPath)}\n`);
  process.stdout.write(`Wrote ${path.relative(projectRoot, summaryPath)}\n`);
  process.stdout.write(`Wrote ${path.relative(projectRoot, validationPath)}\n`);
}

runStressTest().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});

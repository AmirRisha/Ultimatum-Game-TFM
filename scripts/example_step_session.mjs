import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { MODES } from "../src/payoff_engine.mjs";
import { createInitialState, stateLogsToCsv, step } from "../src/step_api.mjs";
import { roundTo } from "../src/utils.mjs";

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

function pickHumanActionForMode1(botAction) {
  const share = Number(botAction?.offer_share ?? 0);
  const threshold = 0.18;
  return { decision: share >= threshold ? "accept" : "reject" };
}

function pickHumanActionForMode2(state) {
  const round = Number(state.round_index);
  const cycle = [0.2, 0.25, 0.3, 0.35];
  const share = cycle[(round - 1) % cycle.length];
  return { offer_share: share };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const modeArg = args.mode === "mode2" ? "mode2" : "mode1";
  const mode =
    modeArg === "mode2" ? MODES.HUMAN_PROPOSER : MODES.HUMAN_RESPONDER;
  const policyMode = args.policyMode === "nn" ? "nn" : "belief";
  const rounds = Number(args.rounds ?? 10);
  const stake = Number(args.stake ?? 200);
  const wealth = args.wealth === "0" ? 0 : 1;
  const seed = Number(args.seed ?? 20260218);
  const outDir = path.resolve(projectRoot, args.outDir ?? "outputs");
  const outPath = path.resolve(outDir, args.out ?? `example_step_${modeArg}_${policyMode}.csv`);

  const fittedPath = path.resolve(projectRoot, "data/fitted_params.json");
  const fittedPayload = JSON.parse(await readFile(fittedPath, "utf8"));
  let nnModel = null;
  if (policyMode === "nn") {
    try {
      nnModel = JSON.parse(
        await readFile(path.resolve(projectRoot, "data/nn_model.json"), "utf8")
      );
    } catch {
      nnModel = null;
    }
  }

  let state = createInitialState(
    {
      mode,
      policyMode,
      rounds,
      stake,
      wealth,
      seed,
    },
    {
      ...fittedPayload.coefficients,
      priors: fittedPayload.priors,
      nnModel,
    }
  );

  while (!state.complete) {
    if (mode === MODES.HUMAN_RESPONDER) {
      const offerStep = step(state, null);
      state = offerStep.new_state;
      const humanAction = pickHumanActionForMode1(offerStep.bot_action);
      const resolveStep = step(state, humanAction);
      state = resolveStep.new_state;
    } else {
      const humanAction = pickHumanActionForMode2(state);
      const resolveStep = step(state, humanAction);
      state = resolveStep.new_state;
    }
  }

  await mkdir(outDir, { recursive: true });
  await writeFile(outPath, stateLogsToCsv(state));

  process.stdout.write(`Example session complete.\n`);
  process.stdout.write(`Mode: ${mode}\n`);
  process.stdout.write(`Policy mode: ${state.config.policyMode}\n`);
  process.stdout.write(
    `Final payoffs -> human: ${roundTo(
      state.cumulative_payoffs.human,
      2
    )}, bot: ${roundTo(state.cumulative_payoffs.bot, 2)}\n`
  );
  process.stdout.write(`Rounds logged: ${state.logs.length}\n`);
  process.stdout.write(`Saved log CSV: ${path.relative(projectRoot, outPath)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});

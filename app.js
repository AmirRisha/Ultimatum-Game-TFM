import { RepeatedUltimatumSession } from "./src/session_engine.mjs";
import { MODES } from "./src/payoff_engine.mjs";
import { DEFAULT_FITTED_PARAMS } from "./src/types.mjs";
import { roundTo } from "./src/utils.mjs";

const TYPE_LABELS = {
  money_maximizer: "Money Maximizer",
  fairness_sensitive: "Fairness Sensitive",
  stake_sensitive: "Stake Sensitive",
  noisy: "Noisy Responder",
};
const TYPE_COLORS = {
  money_maximizer: "#2f5d8a",
  fairness_sensitive: "#0f766e",
  stake_sensitive: "#c26d1d",
  noisy: "#7a8a80",
};

const elements = {
  modeSelect: document.querySelector("#modeSelect"),
  roundsInput: document.querySelector("#roundsInput"),
  stakeSelect: document.querySelector("#stakeSelect"),
  wealthSelect: document.querySelector("#wealthSelect"),
  offerUnitSelect: document.querySelector("#offerUnitSelect"),
  policyModeSelect: document.querySelector("#policyModeSelect"),
  priorsSelect: document.querySelector("#priorsSelect"),
  researchModeToggle: document.querySelector("#researchModeToggle"),
  startBtn: document.querySelector("#startBtn"),
  modelStatus: document.querySelector("#modelStatus"),

  decisionPanel: document.querySelector("#decisionPanel"),
  roundStatus: document.querySelector("#roundStatus"),
  mode1Block: document.querySelector("#mode1Block"),
  mode2Block: document.querySelector("#mode2Block"),
  botOfferText: document.querySelector("#botOfferText"),
  acceptBtn: document.querySelector("#acceptBtn"),
  rejectBtn: document.querySelector("#rejectBtn"),
  offerInputLabel: document.querySelector("#offerInputLabel"),
  offerInput: document.querySelector("#offerInput"),
  submitOfferBtn: document.querySelector("#submitOfferBtn"),
  decisionFeedback: document.querySelector("#decisionFeedback"),
  decisionLogicCard: document.querySelector("#decisionLogicCard"),
  logicHighestType: document.querySelector("#logicHighestType"),
  logicMixtureAccept: document.querySelector("#logicMixtureAccept"),
  logicChosenOffer: document.querySelector("#logicChosenOffer"),
  logicExpectedUtility: document.querySelector("#logicExpectedUtility"),
  logicBestEu: document.querySelector("#logicBestEu"),

  debugPanel: document.querySelector("#debugPanel"),
  snapshotPriors: document.querySelector("#snapshotPriors"),
  snapshotPolicy: document.querySelector("#snapshotPolicy"),
  snapshotTopType: document.querySelector("#snapshotTopType"),
  snapshotBotType: document.querySelector("#snapshotBotType"),
  snapshotBetaUsed: document.querySelector("#snapshotBetaUsed"),
  snapshotMixturePAccept: document.querySelector("#snapshotMixturePAccept"),
  debugBeliefsBody: document.querySelector("#debugBeliefsBody"),
  debugGridWrap: document.querySelector("#debugGridWrap"),
  debugGridBody: document.querySelector("#debugGridBody"),
  advPriorsSource: document.querySelector("#advPriorsSource"),
  advPolicy: document.querySelector("#advPolicy"),
  advBotType: document.querySelector("#advBotType"),
  advBetaUsed: document.querySelector("#advBetaUsed"),
  advCurrentPAccept: document.querySelector("#advCurrentPAccept"),

  resultsPanel: document.querySelector("#resultsPanel"),
  payoffStatus: document.querySelector("#payoffStatus"),
  resultsBody: document.querySelector("#resultsBody"),
  downloadCsvBtn: document.querySelector("#downloadCsvBtn"),
  resetBtn: document.querySelector("#resetBtn"),
};

let fittedParams = { ...DEFAULT_FITTED_PARAMS };
let nnModel = null;
let session = null;
let calibrationMessage = "Calibration unavailable.";
let nnMessage = "NN unavailable.";
let selectedOfferShareForDebug = null;

function show(element, visible) {
  if (!element) {
    return;
  }
  element.classList.toggle("hidden", !visible);
}

function getTypeLabel(type) {
  return TYPE_LABELS[type] ?? type ?? "n/a";
}

function getBotTypeLabel(type) {
  if (type === "maximizer") {
    return "Maximizer";
  }
  if (type === "fairness_sensitive") {
    return "Fairness Sensitive";
  }
  return getTypeLabel(type);
}

function formatPriorsSourceLabel(source) {
  return source === "calibration" ? "Calibration priors" : "Literature (Prior 2)";
}

function formatPolicyLabel(mode) {
  return mode === "nn" ? "Neural-assisted" : "Belief-based";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function toSessionParams(rawPayload) {
  if (!rawPayload) {
    return { ...DEFAULT_FITTED_PARAMS };
  }
  if (rawPayload.coefficients) {
    return {
      ...DEFAULT_FITTED_PARAMS,
      ...rawPayload.coefficients,
      priors: rawPayload.priors ?? DEFAULT_FITTED_PARAMS.priors,
    };
  }
  return {
    ...DEFAULT_FITTED_PARAMS,
    ...rawPayload,
    priors: rawPayload.priors ?? DEFAULT_FITTED_PARAMS.priors,
  };
}

async function loadFittedParams() {
  try {
    const response = await fetch("./data/fitted_params.json");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    fittedParams = toSessionParams(payload);
    calibrationMessage = `Calibration loaded (${payload.sample_size ?? "n/a"} rows).`;
  } catch (error) {
    fittedParams = { ...DEFAULT_FITTED_PARAMS };
    calibrationMessage = `Calibration file not found; using fallback defaults. (${error.message})`;
  }
}

async function loadNnModel() {
  try {
    const response = await fetch("./data/nn_model.json");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    nnModel = await response.json();
    nnMessage = `NN loaded (${nnModel.training?.rows ?? "n/a"} rows, loss ${
      nnModel.training?.loss ?? "n/a"
    }).`;
  } catch (error) {
    nnModel = null;
    nnMessage = `NN file not found; nn policy will fallback to belief. (${error.message})`;
  }
}

function renderModelStatus() {
  elements.modelStatus.textContent = `${calibrationMessage} ${nnMessage}`;
}

function currentMode() {
  return elements.modeSelect.value;
}

function currentPriorsSource() {
  return elements.priorsSelect?.value === "calibration" ? "calibration" : "literature";
}

function isResearchModeEnabled() {
  return Boolean(elements.researchModeToggle?.checked);
}

function updateOfferInputLabel() {
  const unit = elements.offerUnitSelect.value;
  const stake = Number(elements.stakeSelect.value);
  if (unit === "share") {
    elements.offerInputLabel.textContent = "Your offer (%)";
    elements.offerInput.min = "0";
    elements.offerInput.max = "100";
    elements.offerInput.step = "0.5";
    elements.offerInput.placeholder = "e.g. 25";
  } else {
    elements.offerInputLabel.textContent = "Your offer (rupees)";
    elements.offerInput.min = "0";
    elements.offerInput.max = String(stake);
    elements.offerInput.step = "1";
    elements.offerInput.placeholder = `0-${stake}`;
  }
}

function clearResultsTable() {
  elements.resultsBody.innerHTML = "";
}

function parseRationaleForSummary(rationaleText) {
  const text = String(rationaleText ?? "");
  const topMatch = text.match(/Highest posterior type:\s*([a-z_]+)\s*\(([-+]?\d*\.?\d+)\)/i);
  const pAcceptMatch = text.match(/Mixture-based P\(accept\)=([-+]?\d*\.?\d+)/i);
  const objectiveMatch = text.match(/yields [^=]+=([-+]?\d*\.?\d+)/i);
  const bestMatch = text.match(/\(best [^=]+=([-+]?\d*\.?\d+)\)/i);
  const posteriorMatch = text.match(/Observed human [^.]*;\s*posterior now [^.]*\./i);
  const epsilonMatch = text.match(/epsilon exploration[^.]*\./i);

  return {
    topType: topMatch?.[1] ?? null,
    topProb: topMatch?.[2] ?? null,
    pAccept: pAcceptMatch?.[1] ?? null,
    objective: objectiveMatch?.[1] ?? null,
    bestObjective: bestMatch?.[1] ?? null,
    posteriorMessage: posteriorMatch?.[0] ?? null,
    epsilonNote: epsilonMatch?.[0] ?? null,
  };
}

function formatFixed(value, decimals) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(decimals) : "n/a";
}

function appendLastRecord() {
  if (!session) {
    return;
  }
  const records = session.getLogRecords();
  const record = records[records.length - 1];
  if (!record) {
    return;
  }
  const parsed = parseRationaleForSummary(record.decision_rationale);
  const typeKey = parsed.topType ?? record.inferred_type;
  const typeLabel = getTypeLabel(typeKey);
  const topProbText = formatFixed(parsed.topProb, 2);
  const pAcceptText = formatFixed(parsed.pAccept ?? record.expected_accept_prob, 4);
  const offerSharePercent = formatFixed(Number(record.offer_share) * 100, 1);
  const expectedUtilityText = formatFixed(parsed.objective ?? record.expected_value, 2);
  const bestUtilityText = formatFixed(parsed.bestObjective ?? parsed.objective ?? record.expected_value, 2);
  const detailsId = `rationale-details-${record.round}-${records.length}`;
  const detailsPosterior = parsed.posteriorMessage ?? "";
  const detailsEpsilon = parsed.epsilonNote ?? "";

  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${record.round}</td>
    <td>${record.offer_amount} (${roundTo(Number(record.offer_share) * 100, 1)}%)</td>
    <td>${record.accept_reject}</td>
    <td>${getTypeLabel(record.inferred_type)}</td>
    <td>${record.expected_accept_prob}</td>
    <td class="rationale-cell">
      <div class="rationale-line">
        Top type: ${escapeHtml(typeLabel)} (${topProbText}) • P(accept)=${pAcceptText}
      </div>
      <div class="rationale-line">
        Offer ${offerSharePercent}% • EU=${expectedUtilityText} (best ${bestUtilityText})
      </div>
      <button type="button" class="details-toggle" data-target="${detailsId}">Details</button>
      <div id="${detailsId}" class="rationale-details hidden">
        <p><strong>Observed outcome:</strong> ${escapeHtml(record.accept_reject)}</p>
        ${detailsPosterior ? `<p>${escapeHtml(detailsPosterior)}</p>` : ""}
        ${detailsEpsilon ? `<p>${escapeHtml(detailsEpsilon)}</p>` : ""}
        <p class="muted">${escapeHtml(record.decision_rationale)}</p>
      </div>
    </td>
  `;
  const detailsToggle = row.querySelector(".details-toggle");
  if (detailsToggle) {
    detailsToggle.addEventListener("click", () => {
      const targetId = detailsToggle.getAttribute("data-target");
      if (!targetId) {
        return;
      }
      const detailsNode = row.querySelector(`#${targetId}`);
      if (!detailsNode) {
        return;
      }
      const willShow = detailsNode.classList.contains("hidden");
      detailsNode.classList.toggle("hidden", !willShow);
      detailsToggle.textContent = willShow ? "Hide" : "Details";
    });
  }
  elements.resultsBody.appendChild(row);
}

function updatePayoffStatus() {
  if (!session) {
    elements.payoffStatus.textContent = "";
    return;
  }
  const progress = session.getProgress();
  elements.payoffStatus.textContent = `Round ${Math.min(
    progress.roundIndex,
    progress.totalRounds
  )}/${progress.totalRounds} | Human payoff: ${roundTo(
    progress.cumulativePayoffs.human,
    2
  )} | Bot payoff: ${roundTo(progress.cumulativePayoffs.bot, 2)}`;
}

function renderModeBlocks() {
  const mode = currentMode();
  show(elements.mode1Block, mode === MODES.HUMAN_RESPONDER);
  show(elements.mode2Block, mode === MODES.HUMAN_PROPOSER);
}

function formatMaybeNumber(value, decimals) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? roundTo(numeric, decimals) : "";
}

function renderDecisionLogicCard(debug) {
  const hasGrid = Array.isArray(debug?.proposerGrid) && debug.proposerGrid.length > 0;
  const shouldShowCard =
    Boolean(session) &&
    isResearchModeEnabled() &&
    currentMode() === MODES.HUMAN_RESPONDER &&
    hasGrid;
  show(elements.decisionLogicCard, shouldShowCard);
  if (!shouldShowCard) {
    return;
  }

  const beliefs = debug.beliefs ?? {};
  const inferredType = debug.inferredType ?? "n/a";
  const inferredTypeLabel = getTypeLabel(inferredType);
  const inferredProb = Number(beliefs[inferredType]);
  const epsilon = 1e-8;
  const selectedRow =
    debug.proposerGrid.find((item) => {
      const share = Number(item.offerShare);
      return (
        Number.isFinite(selectedOfferShareForDebug) &&
        Number.isFinite(share) &&
        Math.abs(share - selectedOfferShareForDebug) <= epsilon
      );
    }) ?? null;
  const bestExpectedValue = debug.proposerGrid.reduce((currentMax, item) => {
    const numeric = Number(item.expectedValue);
    if (!Number.isFinite(numeric)) {
      return currentMax;
    }
    return Math.max(currentMax, numeric);
  }, Number.NEGATIVE_INFINITY);

  elements.logicHighestType.textContent = `${inferredTypeLabel} (${
    Number.isFinite(inferredProb) ? inferredProb.toFixed(2) : "n/a"
  })`;
  elements.logicMixtureAccept.textContent = Number.isFinite(Number(debug.expectedAcceptProb))
    ? Number(debug.expectedAcceptProb).toFixed(4)
    : "n/a";
  elements.logicChosenOffer.textContent = Number.isFinite(Number(selectedRow?.offerShare))
    ? `${(Number(selectedRow.offerShare) * 100).toFixed(1)}%`
    : "n/a";
  elements.logicExpectedUtility.textContent = Number.isFinite(Number(selectedRow?.expectedValue))
    ? Number(selectedRow.expectedValue).toFixed(2)
    : "n/a";
  elements.logicBestEu.textContent = Number.isFinite(bestExpectedValue)
    ? bestExpectedValue.toFixed(2)
    : "n/a";
}

function renderDebugPanel() {
  if (!session || !isResearchModeEnabled()) {
    show(elements.debugPanel, false);
    show(elements.decisionLogicCard, false);
    return;
  }
  const debug = session.getDebugState();
  show(elements.debugPanel, true);

  renderDecisionLogicCard(debug);

  const botProfile = debug.bot_profile ?? debug.botProfile ?? {};
  const betaNumeric = Number(botProfile.beta);
  const betaUsed = Number.isFinite(betaNumeric) ? roundTo(betaNumeric, 4) : "n/a";
  const priorsSource = session?.config?.priorsSource ?? currentPriorsSource();
  const topTypeKey = debug.inferredType ?? "n/a";
  const topTypeProb = Number(debug.beliefs?.[topTypeKey]);
  elements.snapshotPriors.textContent = formatPriorsSourceLabel(priorsSource);
  elements.snapshotPolicy.textContent = formatPolicyLabel(debug.policyMode ?? "belief");
  elements.snapshotTopType.textContent = `${getTypeLabel(topTypeKey)} (${
    Number.isFinite(topTypeProb) ? topTypeProb.toFixed(2) : "n/a"
  })`;
  elements.snapshotBotType.textContent = getBotTypeLabel(botProfile.automatonType ?? "n/a");
  elements.snapshotBetaUsed.textContent = String(betaUsed);
  elements.snapshotMixturePAccept.textContent = Number.isFinite(Number(debug.expectedAcceptProb))
    ? Number(debug.expectedAcceptProb).toFixed(4)
    : "n/a";

  elements.advPriorsSource.textContent = priorsSource;
  elements.advPolicy.textContent = debug.policyMode ?? "belief";
  elements.advBotType.textContent = getTypeLabel(botProfile.automatonType ?? "n/a");
  elements.advBetaUsed.textContent = String(betaUsed);
  elements.advCurrentPAccept.textContent = Number.isFinite(Number(debug.expectedAcceptProb))
    ? Number(debug.expectedAcceptProb).toFixed(4)
    : "n/a";

  const beliefs = debug.beliefs ?? {};
  const highestBeliefType = Object.entries(beliefs).reduce(
    (best, [type, probability]) =>
      Number(probability) > best.value ? { type, value: Number(probability) } : best,
    { type: "", value: Number.NEGATIVE_INFINITY }
  ).type;
  const beliefRows = Object.entries(beliefs)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([type, probability]) => {
      const numericProbability = Number(probability);
      const percentage = Number.isFinite(numericProbability)
        ? Math.max(0, Math.min(100, numericProbability * 100))
        : 0;
      const probabilityText = Number.isFinite(numericProbability)
        ? numericProbability.toFixed(2)
        : "n/a";
      const barColor = TYPE_COLORS[type] ?? "#0f766e";
      const topClass = type === highestBeliefType ? " class=\"belief-top\"" : "";
      return `
        <tr${topClass}>
          <td>${getTypeLabel(type)}</td>
          <td>${probabilityText}</td>
          <td>
            <div class="belief-bar-track">
              <div class="belief-bar-fill" style="width: ${percentage}%; background-color: ${barColor};"></div>
            </div>
          </td>
        </tr>
      `;
    });
  elements.debugBeliefsBody.innerHTML =
    beliefRows.length > 0
      ? beliefRows.join("")
      : '<tr><td colspan="3" class="muted">No belief data available.</td></tr>';

  const hasGrid = Array.isArray(debug.proposerGrid) && debug.proposerGrid.length > 0;
  show(elements.debugGridWrap, hasGrid);
  if (!hasGrid) {
    elements.debugGridBody.innerHTML = "";
    return;
  }
  const epsilon = 1e-8;
  const maxExpectedValue = debug.proposerGrid.reduce((currentMax, item) => {
    const numeric = Number(item.expectedValue);
    if (!Number.isFinite(numeric)) {
      return currentMax;
    }
    return Math.max(currentMax, numeric);
  }, Number.NEGATIVE_INFINITY);
  const hasExpectedValueRanking = Number.isFinite(maxExpectedValue);
  elements.debugGridBody.innerHTML = debug.proposerGrid
    .map((item) => {
      const itemShare = Number(item.offerShare);
      const itemExpectedValue = Number(item.expectedValue);
      const matchesSelectedShare =
        Number.isFinite(selectedOfferShareForDebug) &&
        Number.isFinite(itemShare) &&
        Math.abs(itemShare - selectedOfferShareForDebug) <= epsilon;
      const isTopExpectedValue =
        hasExpectedValueRanking &&
        Number.isFinite(itemExpectedValue) &&
        Math.abs(itemExpectedValue - maxExpectedValue) <= epsilon;
      const rowHighlightStyle =
        matchesSelectedShare && isTopExpectedValue ? ' style="background-color: #eaf7ee;"' : "";
      return `
        <tr${rowHighlightStyle}>
          <td>${item.offerAmount}</td>
          <td>${roundTo(Number(item.offerShare) * 100, 1)}%</td>
          <td>${formatMaybeNumber(item.acceptProb ?? item.expectedAcceptProb, 4)}</td>
          <td>${formatMaybeNumber(item.ev_max, 2)}</td>
          <td>${formatMaybeNumber(item.eu_fs_current, 2)}</td>
          <td>${formatMaybeNumber(item.eu_fs_beta025, 2)}</td>
          <td>${formatMaybeNumber(item.eu_fs_beta060, 2)}</td>
        </tr>
      `;
    })
    .join("");
}

function finishSession() {
  if (!session) {
    return;
  }
  show(elements.decisionPanel, false);
  updatePayoffStatus();
  elements.decisionFeedback.textContent = "Session complete. Download CSV from Results.";
  renderDebugPanel();
}

function nextRoundPrompt() {
  if (!session) {
    return;
  }
  if (session.isComplete()) {
    finishSession();
    return;
  }

  const progress = session.getProgress();
  elements.roundStatus.textContent = `Round ${progress.roundIndex} of ${progress.totalRounds}`;
  if (currentMode() === MODES.HUMAN_RESPONDER) {
    const pending = session.startRoundForHumanResponderMode();
    selectedOfferShareForDebug = Number(pending.offerShare);
    elements.botOfferText.textContent = `Bot proposes ${pending.offerAmount} rupees (${roundTo(
      pending.offerShare * 100,
      1
    )}%).`;
    elements.decisionFeedback.textContent = isResearchModeEnabled()
      ? "Review the decision logic card for model details."
      : "Choose Accept or Reject.";
  } else {
    selectedOfferShareForDebug = null;
    elements.decisionFeedback.textContent = "Enter your offer and submit.";
  }
  renderDebugPanel();
}

function startSession() {
  const config = {
    mode: currentMode(),
    policyMode: elements.policyModeSelect.value,
    priorsSource: currentPriorsSource(),
    rounds: Number(elements.roundsInput.value),
    stake: Number(elements.stakeSelect.value),
    wealth: Number(elements.wealthSelect.value),
  };
  session = new RepeatedUltimatumSession(
    config,
    {
      ...fittedParams,
      nnModel,
    },
    Math.random
  );
  selectedOfferShareForDebug = null;

  show(elements.decisionPanel, true);
  show(elements.resultsPanel, true);
  renderModeBlocks();
  clearResultsTable();
  updateOfferInputLabel();
  updatePayoffStatus();
  renderDebugPanel();
  if (config.policyMode === "nn" && !nnModel) {
    elements.decisionFeedback.textContent = "NN mode requested but no model loaded; using belief mode.";
  }
  nextRoundPrompt();
}

function handleHumanResponse(accepted) {
  if (!session || session.isComplete()) {
    return;
  }
  session.submitHumanResponse(accepted);
  elements.decisionFeedback.textContent = `You ${accepted ? "accepted" : "rejected"} the offer.`;
  appendLastRecord();
  updatePayoffStatus();
  renderDebugPanel();
  nextRoundPrompt();
}

function parseOfferInput() {
  const rawValue = Number(elements.offerInput.value);
  if (!Number.isFinite(rawValue)) {
    throw new Error("Offer input must be numeric.");
  }
  const unit = elements.offerUnitSelect.value;
  if (unit === "share") {
    const percent = Math.max(0, Math.min(100, rawValue));
    return {
      offerInputMode: "share",
      offerShare: percent / 100,
    };
  }
  const stake = Number(elements.stakeSelect.value);
  const amount = Math.max(0, Math.min(stake, Math.round(rawValue)));
  return {
    offerInputMode: "amount",
    offerAmount: amount,
  };
}

function handleSubmitOffer() {
  if (!session || session.isComplete()) {
    return;
  }
  try {
    const parsed = parseOfferInput();
    const result = session.submitHumanOffer(parsed);
    elements.decisionFeedback.textContent = `Bot ${result.accepted ? "accepted" : "rejected"} your offer.`;
    appendLastRecord();
    updatePayoffStatus();
    elements.offerInput.value = "";
    renderDebugPanel();
    nextRoundPrompt();
  } catch (error) {
    elements.decisionFeedback.textContent = error.message;
  }
}

function downloadCsv() {
  if (!session) {
    return;
  }
  const modeShort = currentMode() === MODES.HUMAN_RESPONDER ? "mode1" : "mode2";
  const blob = new Blob([session.toCsv()], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `ultimatum_${modeShort}_stake${elements.stakeSelect.value}_wealth${elements.wealthSelect.value}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindEvents() {
  elements.startBtn.addEventListener("click", startSession);
  elements.acceptBtn.addEventListener("click", () => handleHumanResponse(true));
  elements.rejectBtn.addEventListener("click", () => handleHumanResponse(false));
  elements.submitOfferBtn.addEventListener("click", handleSubmitOffer);
  elements.downloadCsvBtn.addEventListener("click", downloadCsv);
  elements.resetBtn.addEventListener("click", () => {
    session = null;
    selectedOfferShareForDebug = null;
    show(elements.decisionPanel, false);
    show(elements.debugPanel, false);
    show(elements.resultsPanel, false);
    elements.decisionFeedback.textContent = "";
    clearResultsTable();
    updatePayoffStatus();
  });
  elements.offerUnitSelect.addEventListener("change", updateOfferInputLabel);
  elements.modeSelect.addEventListener("change", renderModeBlocks);
  elements.stakeSelect.addEventListener("change", updateOfferInputLabel);
  elements.researchModeToggle.addEventListener("change", () => {
    if (session && !session.isComplete()) {
      nextRoundPrompt();
      return;
    }
    renderDebugPanel();
  });
}

async function init() {
  bindEvents();
  renderModeBlocks();
  updateOfferInputLabel();
  show(elements.decisionPanel, false);
  show(elements.debugPanel, false);
  show(elements.resultsPanel, false);
  await Promise.all([loadFittedParams(), loadNnModel()]);
  renderModelStatus();
}

init();

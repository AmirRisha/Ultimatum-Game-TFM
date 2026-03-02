import { TARGET_STAKES } from "./types.mjs";

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      const next = line[index + 1];
      if (inQuotes && next === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function toNumber(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const trimmed = String(value).trim();
  if (trimmed.length === 0) {
    return null;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseCsvText(text) {
  if (!text) {
    return [];
  }
  const lines = text.replace(/\r/g, "").split("\n").filter((line) => line.trim().length > 0);
  if (lines.length === 0) {
    return [];
  }
  const header = splitCsvLine(lines[0]).map((column, index) =>
    index === 0 ? column.replace(/^\uFEFF/, "") : column
  );
  const rows = [];
  for (const line of lines.slice(1)) {
    const rawValues = splitCsvLine(line);
    const row = {};
    header.forEach((column, index) => {
      row[column] = rawValues[index] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

export function normalizeUltimatumRows(rawRows) {
  return rawRows
    .map((row) => {
      const stake = toNumber(row.stakes);
      const offerAmount = toNumber(row.offer ?? row.moneyoffer);
      const offerShare = toNumber(row.percent_offer ?? row.SR);
      const wealth = toNumber(row.wealth);
      const accept = toNumber(row.accept);
      const year = toNumber(row.year);
      return {
        stake,
        offerAmount,
        offerShare,
        wealth,
        accept,
        reject: accept === null ? null : 1 - accept,
        year,
      };
    })
    .filter((row) => row.stake !== null && row.offerShare !== null && row.accept !== null);
}

export function calibrationRows(rawRows) {
  return normalizeUltimatumRows(rawRows).filter(
    (row) =>
      TARGET_STAKES.includes(row.stake) &&
      (row.wealth === 0 || row.wealth === 1) &&
      row.offerShare >= 0 &&
      row.offerShare <= 1
  );
}

export function nnTrainingRows(rawRows) {
  return normalizeUltimatumRows(rawRows).filter(
    (row) =>
      row.stake > 0 &&
      (row.wealth === 0 || row.wealth === 1) &&
      row.offerShare >= 0 &&
      row.offerShare <= 1
  );
}

export function rejectModelFeatures(row) {
  return [
    1,
    row.wealth,
    row.stake === 200 ? 1 : 0,
    row.stake === 2000 ? 1 : 0,
    row.stake === 20000 ? 1 : 0,
    row.offerShare,
  ];
}

export function inferShareFromAmount(offerAmount, stake) {
  if (!Number.isFinite(offerAmount) || !Number.isFinite(stake) || stake <= 0) {
    return null;
  }
  return Math.max(0, Math.min(1, offerAmount / stake));
}

export function inferAmountFromShare(offerShare, stake) {
  if (!Number.isFinite(offerShare) || !Number.isFinite(stake) || stake <= 0) {
    return null;
  }
  return Math.round(Math.max(0, Math.min(1, offerShare)) * stake);
}

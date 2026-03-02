function csvEscape(value) {
  const stringValue = value === null || value === undefined ? "" : String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

export class RoundLogger {
  constructor() {
    this.records = [];
  }

  logRound(entry) {
    const normalized = {
      round: entry.round,
      mode: entry.mode,
      stake: entry.stake,
      wealth: entry.wealth,
      offer_amount: entry.offer_amount,
      offer_share: entry.offer_share,
      accept_reject: entry.accept_reject,
      bot_action: entry.bot_action,
      bot_beliefs: entry.bot_beliefs,
      inferred_type: entry.inferred_type,
      decision_rationale: entry.decision_rationale,
      expected_accept_prob: entry.expected_accept_prob,
      expected_value: entry.expected_value,
      cumulative_payoffs: entry.cumulative_payoffs,
    };
    this.records.push(normalized);
  }

  toCsv() {
    const headers = [
      "round",
      "mode",
      "stake",
      "wealth",
      "offer_amount",
      "offer_share",
      "accept_reject",
      "bot_action",
      "bot_beliefs",
      "inferred_type",
      "decision_rationale",
      "expected_accept_prob",
      "expected_value",
      "cumulative_payoffs",
    ];
    const rows = this.records.map((record) =>
      headers.map((header) => csvEscape(record[header])).join(",")
    );
    return [headers.join(","), ...rows].join("\n");
  }
}

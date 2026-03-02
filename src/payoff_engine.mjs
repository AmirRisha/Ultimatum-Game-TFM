export const MODES = {
  HUMAN_RESPONDER: "human_responder_vs_bot_proposer",
  HUMAN_PROPOSER: "human_proposer_vs_bot_responder",
};

export function resolvePayoffs({
  mode,
  stake,
  offerAmount,
  accepted,
  cumulativePayoffs,
}) {
  let humanRoundPayoff = 0;
  let botRoundPayoff = 0;

  if (accepted) {
    if (mode === MODES.HUMAN_RESPONDER) {
      humanRoundPayoff = offerAmount;
      botRoundPayoff = stake - offerAmount;
    } else {
      humanRoundPayoff = stake - offerAmount;
      botRoundPayoff = offerAmount;
    }
  }

  const human = (cumulativePayoffs?.human ?? 0) + humanRoundPayoff;
  const bot = (cumulativePayoffs?.bot ?? 0) + botRoundPayoff;
  return {
    humanRoundPayoff,
    botRoundPayoff,
    cumulative: {
      human,
      bot,
    },
  };
}

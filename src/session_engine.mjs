import { updateProposerBeliefs, updateResponderBeliefs, createInitialBeliefs } from "./belief_model.mjs";
import { chooseBotOffer, buildOfferGrid } from "./bot_proposer_policy.mjs";
import { decideBotResponse } from "./bot_responder_policy.mjs";
import { inferAmountFromShare, inferShareFromAmount } from "./data_loader.mjs";
import { RoundLogger } from "./logger.mjs";
import {
  AUTOMATON_TYPES,
  DEFAULT_AUTOMATON_MIX,
  DEFAULT_BETA_MODE,
  DEFAULT_FIXED_BETA,
  sampleAutomatonType,
  sampleFsBetaOnce,
} from "./automaton_profile.mjs";
import { MODES, resolvePayoffs } from "./payoff_engine.mjs";
import {
  DEFAULT_FITTED_PARAMS,
  LITERATURE_PRIOR_2_RESPONDER,
  PROPOSER_TYPES,
  RESPONDER_TYPES,
} from "./types.mjs";
import { argmaxBelief, roundTo } from "./utils.mjs";

export class RepeatedUltimatumSession {
  constructor(config = {}, fittedParams = DEFAULT_FITTED_PARAMS, rng = Math.random) {
    this.config = {
      mode: config.mode ?? MODES.HUMAN_RESPONDER,
      policyMode: config.policyMode === "nn" ? "nn" : "belief",
      priorsSource: config.priorsSource === "calibration" ? "calibration" : "literature",
      rounds: Math.max(1, Number(config.rounds ?? 10)),
      stake: Number(config.stake ?? 200),
      wealth: config.wealth === 0 ? 0 : 1,
      proposerEpsilon: Number(config.proposerEpsilon ?? 0.08),
      proposerTemperature: Number(config.proposerTemperature ?? 0.1),
      proposerTrembleProb: Number(config.proposerTrembleProb ?? 0.03),
      responderTemperature: Number(config.responderTemperature ?? 0.08),
      responderTrembleProb: Number(config.responderTrembleProb ?? 0.03),
      beliefLearningRate: Number(config.beliefLearningRate ?? 0.85),
      beliefPullback: Number(config.beliefPullback ?? 0.02),
      automatonMix: {
        ...DEFAULT_AUTOMATON_MIX,
        ...(config.automatonMix ?? {}),
      },
      betaMode: config.betaMode ?? DEFAULT_BETA_MODE,
      fixedBeta: Number(config.fixedBeta ?? DEFAULT_FIXED_BETA),
      offerGrid: config.offerGrid ?? buildOfferGrid({ stepShare: Number(config.offerStepShare ?? 0.025) }),
    };
    this.fittedParams = { ...DEFAULT_FITTED_PARAMS, ...fittedParams };
    if (this.config.policyMode === "nn" && !this.fittedParams.nnModel) {
      this.config.policyMode = "belief";
    }
    this.rng = rng;
    const automatonType = sampleAutomatonType(this.config.automatonMix, this.rng);
    const betaMode = this.config.betaMode;
    const beta =
      automatonType === AUTOMATON_TYPES.MAXIMIZER
        ? 0
        : sampleFsBetaOnce(betaMode, this.config.fixedBeta, this.rng);
    this.botProfile = {
      automatonType,
      betaMode,
      beta,
    };
    this.roundIndex = 1;
    this.cumulativePayoffs = { human: 0, bot: 0 };
    this.logger = new RoundLogger();
    this.pendingRound = null;
    const responderPriorConfig =
      this.config.priorsSource === "calibration"
        ? this.fittedParams.priors?.responder
        : LITERATURE_PRIOR_2_RESPONDER;

    this.anchorHumanResponderBeliefs = createInitialBeliefs(
      RESPONDER_TYPES,
      responderPriorConfig
    );
    this.anchorHumanProposerBeliefs = createInitialBeliefs(
      PROPOSER_TYPES,
      this.fittedParams.priors?.proposer
    );
    this.humanResponderBeliefs = { ...this.anchorHumanResponderBeliefs };
    this.humanProposerBeliefs = { ...this.anchorHumanProposerBeliefs };
    this.botResponderBeliefs = createInitialBeliefs(
      RESPONDER_TYPES,
      responderPriorConfig
    );
    this.debugState = {
      mode: this.config.mode,
      round: this.roundIndex,
      beliefs:
        this.config.mode === MODES.HUMAN_RESPONDER
          ? { ...this.humanResponderBeliefs }
          : { ...this.humanProposerBeliefs },
      inferredType:
        this.config.mode === MODES.HUMAN_RESPONDER
          ? argmaxBelief(this.humanResponderBeliefs).type
          : argmaxBelief(this.humanProposerBeliefs).type,
      policyMode: this.config.policyMode,
      botProfile: { ...this.botProfile },
      bot_profile: { ...this.botProfile },
      proposerGrid: [],
      expectedAcceptProb: null,
    };
  }

  isComplete() {
    return this.roundIndex > this.config.rounds;
  }

  getProgress() {
    return {
      roundIndex: this.roundIndex,
      totalRounds: this.config.rounds,
      complete: this.isComplete(),
      cumulativePayoffs: { ...this.cumulativePayoffs },
    };
  }

  getLogRecords() {
    return [...this.logger.records];
  }

  getDebugState() {
    const debugBotProfile = {
      ...(this.debugState?.bot_profile ?? this.debugState?.botProfile ?? this.botProfile),
    };
    return {
      ...this.debugState,
      beliefs: { ...(this.debugState?.beliefs ?? {}) },
      proposerGrid: [...(this.debugState?.proposerGrid ?? [])],
      botProfile: { ...debugBotProfile },
      bot_profile: { ...debugBotProfile },
    };
  }

  toCsv() {
    return this.logger.toCsv();
  }

  startRoundForHumanResponderMode() {
    if (this.config.mode !== MODES.HUMAN_RESPONDER) {
      throw new Error("startRoundForHumanResponderMode called in proposer mode.");
    }
    if (this.isComplete()) {
      throw new Error("Session is already complete.");
    }
    if (this.pendingRound) {
      return this.pendingRound;
    }

    const offerDecision = chooseBotOffer({
      stake: this.config.stake,
      wealth: this.config.wealth,
      roundIndex: this.roundIndex,
      responderBeliefs: this.humanResponderBeliefs,
      fittedParams: this.fittedParams,
      offerGrid: this.config.offerGrid,
      epsilon: this.config.proposerEpsilon,
      temperature: this.config.proposerTemperature,
      trembleProb: this.config.proposerTrembleProb,
      policyMode: this.config.policyMode,
      automatonType: this.botProfile.automatonType,
      betaUsed: this.botProfile.beta,
      rng: this.rng,
    });

    this.pendingRound = {
      round: this.roundIndex,
      offerAmount: offerDecision.offerAmount,
      offerShare: offerDecision.offerShare,
      expectedAcceptProb: offerDecision.expectedAcceptProb,
      expectedValue: offerDecision.expectedValue,
      inferredType: offerDecision.inferredType,
      rationale: offerDecision.rationale,
      beliefsBefore: { ...this.humanResponderBeliefs },
      proposerGrid: offerDecision.debug.candidateEvaluations.map((item) => ({
        offerAmount: item.offerAmount,
        offerShare: item.offerShare,
        acceptProb: item.acceptProb,
        expectedAcceptProb: item.expectedAcceptProb ?? item.acceptProb,
        expectedValue: item.expectedValue,
        ev_max: item.ev_max,
        eu_fs_current: item.eu_fs_current,
        eu_fs_beta025: item.eu_fs_beta025,
        eu_fs_beta060: item.eu_fs_beta060,
        utility_raw_fs_current: item.utility_raw_fs_current,
        betaUsed: item.betaUsed,
      })),
    };
    this.debugState = {
      mode: this.config.mode,
      round: this.roundIndex,
      beliefs: { ...this.humanResponderBeliefs },
      inferredType: offerDecision.inferredType,
      policyMode: this.config.policyMode,
      botProfile: { ...this.botProfile },
      bot_profile: { ...this.botProfile },
      proposerGrid: [...this.pendingRound.proposerGrid],
      expectedAcceptProb: offerDecision.expectedAcceptProb,
    };
    return this.pendingRound;
  }

  submitHumanResponse(accepted) {
    if (!this.pendingRound) {
      throw new Error("No pending round. Call startRoundForHumanResponderMode first.");
    }
    const actionAccepted = Boolean(accepted);
    const context = {
      offerShare: this.pendingRound.offerShare,
      stake: this.config.stake,
      wealth: this.config.wealth,
      roundIndex: this.roundIndex,
    };
    const beliefUpdate = updateResponderBeliefs({
      priorBeliefs: this.humanResponderBeliefs,
      accepted: actionAccepted,
      context,
      fittedParams: this.fittedParams,
      policyMode: this.config.policyMode,
      learningRate: this.config.beliefLearningRate,
      pullback: this.config.beliefPullback,
      pullbackPrior: this.anchorHumanResponderBeliefs,
    });
    this.humanResponderBeliefs = beliefUpdate.posterior;

    const payoffResult = resolvePayoffs({
      mode: this.config.mode,
      stake: this.config.stake,
      offerAmount: this.pendingRound.offerAmount,
      accepted: actionAccepted,
      cumulativePayoffs: this.cumulativePayoffs,
    });
    this.cumulativePayoffs = payoffResult.cumulative;

    const posteriorComment = `Observed human ${actionAccepted ? "accept" : "reject"}; posterior now ${
      beliefUpdate.inferred.type
    } (${roundTo(beliefUpdate.inferred.probability, 2)}).`;
    const priorTopType = this.pendingRound.inferredType;
    const priorTopProb = Number(this.pendingRound.beliefsBefore?.[priorTopType]);
    const mixtureAcceptProb = Number(this.pendingRound.expectedAcceptProb);
    const offerSharePercent = `${roundTo(this.pendingRound.offerShare * 100, 1)}%`;
    const isMaximizer = this.botProfile?.automatonType === "maximizer";
    const betaDisplay = roundTo(Number(this.botProfile?.beta ?? 0), 4);
    const selectedObjectiveValue = Number(this.pendingRound.expectedValue);
    const bestObjectiveValue = (this.pendingRound.proposerGrid ?? []).reduce(
      (maxValue, item) => {
        const numeric = Number(item?.expectedValue);
        if (!Number.isFinite(numeric)) {
          return maxValue;
        }
        return Math.max(maxValue, numeric);
      },
      Number.NEGATIVE_INFINITY
    );
    const safeBestObjectiveValue = Number.isFinite(bestObjectiveValue)
      ? bestObjectiveValue
      : selectedObjectiveValue;
    const objectiveLabel = isMaximizer ? "EV" : `EU (β=${betaDisplay})`;
    const bestLabel = isMaximizer ? "best EV" : "best EU";
    const preDecisionRationale = `Highest posterior type: ${priorTopType} (${Number.isFinite(priorTopProb) ? priorTopProb.toFixed(2) : "0.00"}). Mixture-based P(accept)=${Number.isFinite(mixtureAcceptProb) ? mixtureAcceptProb.toFixed(4) : "0.0000"}. Offer ${offerSharePercent} yields ${objectiveLabel}=${Number.isFinite(selectedObjectiveValue) ? selectedObjectiveValue.toFixed(2) : "0.00"} (${bestLabel}=${Number.isFinite(safeBestObjectiveValue) ? safeBestObjectiveValue.toFixed(2) : "0.00"}).`;
    const resultsRationale = `${preDecisionRationale} ${posteriorComment}`;
    const combinedRationale = `${this.pendingRound.rationale} ${posteriorComment}`;

    this.logger.logRound({
      round: this.roundIndex,
      mode: this.config.mode,
      stake: this.config.stake,
      wealth: this.config.wealth,
      offer_amount: this.pendingRound.offerAmount,
      offer_share: roundTo(this.pendingRound.offerShare, 4),
      accept_reject: actionAccepted ? "accept" : "reject",
      bot_action: `offer_${this.pendingRound.offerAmount}`,
      bot_beliefs: JSON.stringify(this.humanResponderBeliefs),
      inferred_type: beliefUpdate.inferred.type,
      decision_rationale: resultsRationale,
      expected_accept_prob: roundTo(this.pendingRound.expectedAcceptProb, 4),
      expected_value: roundTo(this.pendingRound.expectedValue, 4),
      cumulative_payoffs: JSON.stringify(this.cumulativePayoffs),
    });

    const result = {
      round: this.roundIndex,
      accepted: actionAccepted,
      offerAmount: this.pendingRound.offerAmount,
      offerShare: this.pendingRound.offerShare,
      payoff: payoffResult,
      beliefs: this.humanResponderBeliefs,
      inferredType: beliefUpdate.inferred.type,
      rationale: combinedRationale,
    };
    this.debugState = {
      mode: this.config.mode,
      round: this.roundIndex,
      beliefs: { ...this.humanResponderBeliefs },
      inferredType: beliefUpdate.inferred.type,
      policyMode: this.config.policyMode,
      botProfile: { ...this.botProfile },
      bot_profile: { ...this.botProfile },
      proposerGrid: [...(this.pendingRound.proposerGrid ?? [])],
      expectedAcceptProb: this.pendingRound.expectedAcceptProb,
    };

    this.pendingRound = null;
    this.roundIndex += 1;
    return result;
  }

  submitHumanOffer({
    offerAmount = null,
    offerShare = null,
    offerInputMode = "amount",
  }) {
    if (this.config.mode !== MODES.HUMAN_PROPOSER) {
      throw new Error("submitHumanOffer called in responder mode.");
    }
    if (this.isComplete()) {
      throw new Error("Session is already complete.");
    }

    let normalizedOfferAmount = offerAmount;
    let normalizedOfferShare = offerShare;
    if (offerInputMode === "share") {
      normalizedOfferAmount = inferAmountFromShare(Number(offerShare), this.config.stake);
      normalizedOfferShare = inferShareFromAmount(normalizedOfferAmount, this.config.stake);
    } else {
      normalizedOfferAmount = Math.max(0, Math.min(this.config.stake, Math.round(Number(offerAmount))));
      normalizedOfferShare = inferShareFromAmount(normalizedOfferAmount, this.config.stake);
    }

    if (normalizedOfferAmount === null || normalizedOfferShare === null) {
      throw new Error("Offer input is invalid.");
    }

    const proposerUpdate = updateProposerBeliefs({
      priorBeliefs: this.humanProposerBeliefs,
      observedOfferShare: normalizedOfferShare,
      context: {
        stake: this.config.stake,
        wealth: this.config.wealth,
        roundIndex: this.roundIndex,
      },
      learningRate: this.config.beliefLearningRate,
      pullback: this.config.beliefPullback,
      pullbackPrior: this.anchorHumanProposerBeliefs,
    });
    this.humanProposerBeliefs = proposerUpdate.posterior;

    const botDecision = decideBotResponse({
      offerShare: normalizedOfferShare,
      offerAmount: normalizedOfferAmount,
      stake: this.config.stake,
      wealth: this.config.wealth,
      roundIndex: this.roundIndex,
      proposerBeliefs: this.humanProposerBeliefs,
      responderBeliefs: this.botResponderBeliefs,
      fittedParams: this.fittedParams,
      policyMode: this.config.policyMode,
      temperature: this.config.responderTemperature,
      trembleProb: this.config.responderTrembleProb,
      rng: this.rng,
    });

    const payoffResult = resolvePayoffs({
      mode: this.config.mode,
      stake: this.config.stake,
      offerAmount: normalizedOfferAmount,
      accepted: botDecision.accepted,
      cumulativePayoffs: this.cumulativePayoffs,
    });
    this.cumulativePayoffs = payoffResult.cumulative;

    const proposerPosteriorComment = `Observed human offer ${roundTo(
      normalizedOfferShare * 100,
      1
    )}%; inferred proposer type ${proposerUpdate.inferred.type} (${roundTo(
      proposerUpdate.inferred.probability,
      2
    )}).`;
    const combinedRationale = `${proposerPosteriorComment} ${botDecision.rationale}`;

    this.logger.logRound({
      round: this.roundIndex,
      mode: this.config.mode,
      stake: this.config.stake,
      wealth: this.config.wealth,
      offer_amount: normalizedOfferAmount,
      offer_share: roundTo(normalizedOfferShare, 4),
      accept_reject: botDecision.accepted ? "accept" : "reject",
      bot_action: botDecision.botAction,
      bot_beliefs: JSON.stringify(this.humanProposerBeliefs),
      inferred_type: proposerUpdate.inferred.type,
      decision_rationale: combinedRationale,
      expected_accept_prob: roundTo(botDecision.expectedAcceptProb, 4),
      expected_value: roundTo(botDecision.expectedValue, 4),
      cumulative_payoffs: JSON.stringify(this.cumulativePayoffs),
    });

    const result = {
      round: this.roundIndex,
      offerAmount: normalizedOfferAmount,
      offerShare: normalizedOfferShare,
      accepted: botDecision.accepted,
      botDecision,
      payoff: payoffResult,
      beliefs: this.humanProposerBeliefs,
      inferredType: proposerUpdate.inferred.type,
      rationale: combinedRationale,
    };
    this.debugState = {
      mode: this.config.mode,
      round: this.roundIndex,
      beliefs: { ...this.humanProposerBeliefs },
      inferredType: proposerUpdate.inferred.type,
      policyMode: this.config.policyMode,
      botProfile: { ...this.botProfile },
      bot_profile: { ...this.botProfile },
      proposerGrid: [],
      expectedAcceptProb: botDecision.expectedAcceptProb,
    };

    this.roundIndex += 1;
    return result;
  }
}

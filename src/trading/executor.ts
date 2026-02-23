import { ExecutedTrade, PaperTradingReport, SetupSignal, TradeOutcome } from './types';

export interface ExecutorConfig {
	initialUnit: number;
	realizedSavingsPct: number;
	legAScalpRoi: number;
	legBRunnerRoi: number;
	stopLossRoi: number;
	maxSignalsPerRun: number;
}

const DEFAULT_CONFIG: ExecutorConfig = {
	initialUnit: 7000,
	realizedSavingsPct: 0.25,
	legAScalpRoi: 0.2,
	legBRunnerRoi: 0.45,
	stopLossRoi: -0.3,
	maxSignalsPerRun: 5
};

function resolveOutcome(signal: SetupSignal): TradeOutcome {
	if (signal.historicalOutcome) {
		return signal.historicalOutcome;
	}
	return signal.qualityScore >= 60 ? 'win' : 'loss';
}

function isWithinWindow(dateIso: string, startIso: string, endIso: string): boolean {
	return dateIso >= startIso && dateIso <= endIso;
}

export class TradeExecutor {
	private readonly config: ExecutorConfig;

	constructor(config?: Partial<ExecutorConfig>) {
		this.config = { ...DEFAULT_CONFIG, ...(config || {}) };
	}

	runWeekPaperTrading(signals: SetupSignal[], endDate: Date = new Date()): PaperTradingReport {
		const endIso = endDate.toISOString().slice(0, 10);
		const startDate = new Date(endDate);
		startDate.setUTCDate(startDate.getUTCDate() - 7);
		const startIso = startDate.toISOString().slice(0, 10);

		let candidates = signals.filter((signal) => isWithinWindow(signal.setupDate, startIso, endIso));
		if (candidates.length === 0) {
			candidates = [...signals].slice(0, this.config.maxSignalsPerRun);
		}

		const selected = candidates
			.sort((a, b) => {
				if (a.setupDate === b.setupDate) {
					return b.qualityScore - a.qualityScore;
				}
				return a.setupDate.localeCompare(b.setupDate);
			})
			.slice(0, this.config.maxSignalsPerRun);

		let currentUnit = this.config.initialUnit;
		let realizedSavings = 0;
		const trades: ExecutedTrade[] = [];

		for (const signal of selected) {
			const outcome = resolveOutcome(signal);
			const startingUnit = currentUnit;
			const legAllocation = startingUnit / 2;

			const legARoi = outcome === 'win' ? this.config.legAScalpRoi : this.config.stopLossRoi;
			const legBRoi = outcome === 'win' ? this.config.legBRunnerRoi : this.config.stopLossRoi;

			const legAProfit = legAllocation * legARoi;
			const legBProfit = legAllocation * legBRoi;
			const totalProfit = legAProfit + legBProfit;

			if (totalProfit > 0) {
				const toSavings = totalProfit * this.config.realizedSavingsPct;
				realizedSavings += toSavings;
				currentUnit += totalProfit - toSavings;
			} else {
				currentUnit += totalProfit;
			}

			currentUnit = Number(Math.max(0, currentUnit).toFixed(2));
			realizedSavings = Number(realizedSavings.toFixed(2));

			trades.push({
				ticker: signal.ticker,
				setupDate: signal.setupDate,
				direction: signal.direction,
				qualityScore: signal.qualityScore,
				result: outcome,
				startingUnit: Number(startingUnit.toFixed(2)),
				legA: {
					allocation: Number(legAllocation.toFixed(2)),
					roi: Number(legARoi.toFixed(4)),
					profit: Number(legAProfit.toFixed(2))
				},
				legB: {
					allocation: Number(legAllocation.toFixed(2)),
					roi: Number(legBRoi.toFixed(4)),
					profit: Number(legBProfit.toFixed(2))
				},
				totalProfit: Number(totalProfit.toFixed(2)),
				endingUnit: currentUnit,
				realizedSavingsBalance: realizedSavings
			});
		}

		const wins = trades.filter((trade) => trade.result === 'win').length;
		const losses = trades.length - wins;
		const winRate = trades.length === 0 ? 0 : Number((wins / trades.length).toFixed(4));

		return {
			startDate: startIso,
			endDate: endIso,
			startingUnit: this.config.initialUnit,
			endingUnit: currentUnit,
			realizedSavings,
			totalTrades: trades.length,
			wins,
			losses,
			winRate,
			trades
		};
	}
}

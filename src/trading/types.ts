export type SetupDirection = 'bull' | 'bear';
export type TradeOutcome = 'win' | 'loss';

export interface SetupCandidate {
	setupDate: string;
	ticker: string;
	entryPrice: number;
	newsCountWindow: number;
	avgSentimentWindow: number;
	newsHeadlinesWindow: string;
	historicalOutcome?: TradeOutcome;
	rawRow: Record<string, string>;
}

export interface SetupSignal extends SetupCandidate {
	direction: SetupDirection;
	qualityScore: number;
	confidence: number;
	recommendedHoldDays: number;
}

export interface TradeLegResult {
	allocation: number;
	roi: number;
	profit: number;
}

export interface ExecutedTrade {
	ticker: string;
	setupDate: string;
	direction: SetupDirection;
	qualityScore: number;
	result: TradeOutcome;
	startingUnit: number;
	legA: TradeLegResult;
	legB: TradeLegResult;
	totalProfit: number;
	endingUnit: number;
	realizedSavingsBalance: number;
}

export interface PaperTradingReport {
	startDate: string;
	endDate: string;
	startingUnit: number;
	endingUnit: number;
	realizedSavings: number;
	totalTrades: number;
	wins: number;
	losses: number;
	winRate: number;
	trades: ExecutedTrade[];
}

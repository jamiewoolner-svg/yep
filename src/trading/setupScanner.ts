import path from 'path';
import { readCsv } from './csv';
import { SetupCandidate, SetupDirection, SetupSignal, TradeOutcome } from './types';

function toNumber(value: string): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : 0;
}

function parseOutcome(value: string): TradeOutcome | undefined {
	const lowered = value.trim().toLowerCase();
	if (lowered === 'win' || lowered === 'loss') {
		return lowered;
	}
	return undefined;
}

function normalizeDate(date: string): string {
	if (/^\d{4}-\d{2}-\d{2}$/.test(date)) {
		return date;
	}
	const parsed = new Date(date);
	if (Number.isNaN(parsed.getTime())) {
		return '';
	}
	return parsed.toISOString().slice(0, 10);
}

function dateDistanceInDays(fromIso: string, toIso: string): number {
	const from = new Date(`${fromIso}T00:00:00Z`).getTime();
	const to = new Date(`${toIso}T00:00:00Z`).getTime();
	return Math.round((to - from) / 86400000);
}

function scoreCandidate(candidate: SetupCandidate, todayIso: string): number {
	const newsScore = Math.min(candidate.newsCountWindow / 25, 1) * 40;
	const sentimentScore = Math.min(Math.abs(candidate.avgSentimentWindow) / 0.2, 1) * 25;
	const hasHeadlineScore = candidate.newsHeadlinesWindow.trim().length > 0 ? 10 : 0;
	const recencyDays = Math.max(0, dateDistanceInDays(candidate.setupDate, todayIso));
	const recencyScore = Math.max(0, 25 - recencyDays * 1.5);
	return Number((newsScore + sentimentScore + hasHeadlineScore + recencyScore).toFixed(2));
}

function recommendHoldDays(qualityScore: number): number {
	if (qualityScore >= 70) {
		return 10;
	}
	if (qualityScore >= 55) {
		return 7;
	}
	return 3;
}

function toDirection(avgSentiment: number): SetupDirection {
	return avgSentiment >= 0 ? 'bull' : 'bear';
}

export class SetupScanner {
	loadCandidatesFromCsv(filePath: string): SetupCandidate[] {
		const rows = readCsv(filePath);
		return rows
			.map((row) => {
				const setupDate = normalizeDate(row.setup_date || row.date || '');
				const ticker = String(row.ticker || row.symbol || '').toUpperCase();
				if (!setupDate || !ticker) {
					return null;
				}

				return {
					setupDate,
					ticker,
					entryPrice: toNumber(row.entry_price || row.price || '0'),
					newsCountWindow: toNumber(row.news_count_window || row.news_count_day0 || '0'),
					avgSentimentWindow: toNumber(row.avg_sentiment_window || row.avg_sentiment_day0 || '0'),
					newsHeadlinesWindow: String(row.news_headlines_window || row.news_headlines_day0 || ''),
					historicalOutcome: parseOutcome(String(row.outcome || '')),
					rawRow: row
				} as SetupCandidate;
			})
			.filter((candidate): candidate is SetupCandidate => candidate !== null);
	}

	findHighQualitySetups(candidates: SetupCandidate[], maxResults: number, minQuality = 40): SetupSignal[] {
		const todayIso = new Date().toISOString().slice(0, 10);
		return candidates
			.map((candidate) => {
				const qualityScore = scoreCandidate(candidate, todayIso);
				const confidence = Number(Math.min(0.95, Math.max(0.5, qualityScore / 100)).toFixed(2));
				return {
					...candidate,
					direction: toDirection(candidate.avgSentimentWindow),
					qualityScore,
					confidence,
					recommendedHoldDays: recommendHoldDays(qualityScore)
				} as SetupSignal;
			})
			.filter((signal) => signal.qualityScore >= minQuality)
			.sort((a, b) => b.qualityScore - a.qualityScore)
			.slice(0, maxResults);
	}
}

export function defaultSetupTablePath(): string {
	return path.join(process.cwd(), 'data/setup_news_table.csv');
}

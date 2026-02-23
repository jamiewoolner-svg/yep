import fs from 'fs';
import path from 'path';
import axios from 'axios';
import { readCsv, writeCsv, CsvRow } from './lib/csv';

const DATA_DIR = path.join(__dirname, '../data');
const V2_PATH = process.env.V2_FILE_PATH || path.join(DATA_DIR, 'v2_trades.csv');
const NEWS_PATH = process.env.NEWS_FILE_PATH || path.join(DATA_DIR, 'news_sentiment.csv');
const OUTPUT_PATH = process.env.SETUP_NEWS_OUT_PATH || path.join(DATA_DIR, 'setup_news_table.csv');
const WINDOW_DAYS = Number(process.env.NEWS_WINDOW_DAYS || 2);
const USE_GDELT_FALLBACK = String(process.env.USE_GDELT_FALLBACK || 'true').toLowerCase() !== 'false';

type SetupNewsItem = {
	date: string;
	headline: string;
	sentiment_score: string;
	source: string;
};

function normalizeDate(input: string): string {
	if (!input) {
		return '';
	}
	if (/^\d{4}-\d{2}-\d{2}$/.test(input)) {
		return input;
	}
	const parsed = new Date(input);
	if (Number.isNaN(parsed.getTime())) {
		return '';
	}
	return parsed.toISOString().slice(0, 10);
}

function dateDiffDays(a: string, b: string): number {
	const da = new Date(`${a}T00:00:00Z`).getTime();
	const db = new Date(`${b}T00:00:00Z`).getTime();
	return Math.round((da - db) / 86400000);
}

function detectColumn(row: CsvRow, options: string[]): string {
	for (const key of options) {
		if (key in row) {
			return key;
		}
	}
	throw new Error(`Missing required columns. Tried: ${options.join(', ')}`);
}

function parseNumber(value: string): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : 0;
}

function scoreHeadline(headline: string): number {
	const lower = headline.toLowerCase();
	const positiveWords = ['beat', 'surge', 'strong', 'growth', 'upgrade', 'record', 'gain'];
	const negativeWords = ['miss', 'fall', 'downgrade', 'lawsuit', 'weak', 'decline', 'cut'];
	let score = 0;
	for (const token of positiveWords) {
		if (lower.includes(token)) {
			score += 0.2;
		}
	}
	for (const token of negativeWords) {
		if (lower.includes(token)) {
			score -= 0.2;
		}
	}
	if (score > 1) {
		return 1;
	}
	if (score < -1) {
		return -1;
	}
	return Number(score.toFixed(2));
}

function formatGdeltDateTime(date: string, endOfDay: boolean): string {
	const base = date.replace(/-/g, '');
	return `${base}${endOfDay ? '235959' : '000000'}`;
}

function parseGdeltDate(input: string): string {
	const digits = String(input || '').replace(/[^0-9]/g, '');
	if (digits.length < 8) {
		return '';
	}
	const y = digits.slice(0, 4);
	const m = digits.slice(4, 6);
	const d = digits.slice(6, 8);
	return `${y}-${m}-${d}`;
}

async function delay(ms: number): Promise<void> {
	await new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchGdeltNewsForSetupWindow(ticker: string, startDate: string, endDate: string): Promise<SetupNewsItem[]> {
	const url = 'https://api.gdeltproject.org/api/v2/doc/doc';
    let response;
	let lastError: unknown;
	for (let attempt = 1; attempt <= 3; attempt++) {
		try {
			response = await axios.get(url, {
				timeout: 20000,
				params: {
					query: `"${ticker}"`,
					mode: 'ArtList',
					format: 'json',
					maxrecords: 120,
					sort: 'DateDesc',
					startdatetime: formatGdeltDateTime(startDate, false),
					enddatetime: formatGdeltDateTime(endDate, true)
				}
			});
			break;
		} catch (error) {
			lastError = error;
			if (attempt < 3) {
				await delay(500 * attempt);
			}
		}
	}

	if (!response) {
		throw (lastError instanceof Error ? lastError : new Error('GDELT request failed'));
	}

	const articles = Array.isArray(response.data?.articles) ? response.data.articles : [];
	const rows: SetupNewsItem[] = [];
	for (const article of articles) {
		const headline = String(article?.title || '').replace(/\s+/g, ' ').trim();
		const seenDate = parseGdeltDate(String(article?.seendate || article?.date || ''));
		if (!headline || !seenDate) {
			continue;
		}
		rows.push({
			date: seenDate,
			headline,
			sentiment_score: String(scoreHeadline(headline)),
			source: 'gdelt'
		});
	}

	const deduped = new Map<string, SetupNewsItem>();
	for (const row of rows) {
		deduped.set(`${row.date}|${row.headline}`, row);
	}
	return Array.from(deduped.values());
}

async function main(): Promise<void> {
	if (!fs.existsSync(V2_PATH)) {
		throw new Error(`Missing v2 setup file: ${V2_PATH}`);
	}
	if (!fs.existsSync(NEWS_PATH)) {
		throw new Error(`Missing news file: ${NEWS_PATH}`);
	}

	const v2Rows = readCsv(V2_PATH);
	const newsRows = readCsv(NEWS_PATH);

	if (v2Rows.length === 0) {
		throw new Error(`No rows found in ${V2_PATH}`);
	}

	const sample = v2Rows[0];
	const dateCol = detectColumn(sample, ['date', 'trade_date', 'day0_date', 'opened_at']);
	const tickerCol = detectColumn(sample, ['ticker', 'symbol']);

	const newsByTicker = new Map<string, CsvRow[]>();
	for (const row of newsRows) {
		const ticker = String(row.ticker || row.symbol || '').toUpperCase();
		const date = normalizeDate(String(row.date || row.news_date || ''));
		const headline = String(row.headline || '').trim();
		if (!ticker || !date || !headline) {
			continue;
		}
		const current = newsByTicker.get(ticker) || [];
		current.push({ ...row, date, ticker, headline });
		newsByTicker.set(ticker, current);
	}

	const outRows: CsvRow[] = [];
	for (const setup of v2Rows) {
		const setupDate = normalizeDate(String(setup[dateCol] || ''));
		const ticker = String(setup[tickerCol] || '').toUpperCase();
		const candidates = newsByTicker.get(ticker) || [];

		let around = candidates
			.map((news) => {
				const newsDate = normalizeDate(String(news.date || ''));
				const delta = dateDiffDays(newsDate, setupDate);
				return { news, delta };
			})
			.filter(({ delta }) => Math.abs(delta) <= WINDOW_DAYS)
			.sort((a, b) => a.delta - b.delta || String(a.news.date).localeCompare(String(b.news.date)));

		if (around.length === 0 && USE_GDELT_FALLBACK && setupDate) {
			const startDate = toDateWithOffset(setupDate, -WINDOW_DAYS);
			const endDate = toDateWithOffset(setupDate, WINDOW_DAYS);
			try {
				const gdeltRows = await fetchGdeltNewsForSetupWindow(ticker, startDate, endDate);
				around = gdeltRows
					.map((row) => {
						const delta = dateDiffDays(row.date, setupDate);
						return {
							news: {
								date: row.date,
								headline: row.headline,
								sentiment_score: row.sentiment_score,
								source: row.source
							},
							delta
						};
					})
					.filter(({ delta }) => Math.abs(delta) <= WINDOW_DAYS)
					.sort((a, b) => a.delta - b.delta || String(a.news.date).localeCompare(String(b.news.date)));
			} catch (error) {
				const message = error instanceof Error ? error.message : 'unknown error';
				console.warn(`GDELT fallback failed for ${ticker} @ ${setupDate}: ${message}`);
			}
		}

		const sentiments = around.map(({ news }) => parseNumber(String(news.sentiment_score || '0')));
		const avgSentiment = sentiments.length > 0
			? Number((sentiments.reduce((sum, value) => sum + value, 0) / sentiments.length).toFixed(4))
			: 0;
		const minSentiment = sentiments.length > 0 ? Math.min(...sentiments) : 0;
		const maxSentiment = sentiments.length > 0 ? Math.max(...sentiments) : 0;

		outRows.push({
			...setup,
			setup_date: setupDate,
			ticker,
			news_window_days: String(WINDOW_DAYS),
			news_count_window: String(around.length),
			avg_sentiment_window: String(avgSentiment),
			min_sentiment_window: String(Number(minSentiment.toFixed(4))),
			max_sentiment_window: String(Number(maxSentiment.toFixed(4))),
			news_headlines_window: around
				.map(({ delta, news }) => `[${delta >= 0 ? '+' : ''}${delta}d] ${String(news.date)} (${String(news.source || 'local')}): ${String(news.headline)}`)
				.join(' || ')
		});
	}

	const headers = Array.from(new Set(outRows.flatMap((row) => Object.keys(row))));
	writeCsv(OUTPUT_PATH, headers, outRows);
	console.log(`Saved ${outRows.length} setup rows to ${OUTPUT_PATH} using ±${WINDOW_DAYS} day news window.`);
}

function toDateWithOffset(date: string, offsetDays: number): string {
	const d = new Date(`${date}T00:00:00Z`);
	d.setUTCDate(d.getUTCDate() + offsetDays);
	return d.toISOString().slice(0, 10);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

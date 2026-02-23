import fs from 'fs';
import path from 'path';
import axios from 'axios';
import dotenv from 'dotenv';
import { readCsv, writeCsv } from './lib/csv';
import { getLastSixYearsRange, isInRange, toIsoDate } from './lib/dates';

dotenv.config();

type NewsRow = {
	date: string;
	ticker: string;
	headline: string;
	sentiment_score: string;
	source: string;
};

type PolygonNewsItem = {
	published_utc?: string;
	title?: string;
};

const OUTPUT_PATH = path.join(__dirname, '../data/news_sentiment.csv');
const DEFAULT_V2_PATH = path.join(__dirname, '../data/v2_trades.csv');

function getTickersFromFile(filePath: string): string[] {
	const content = fs.readFileSync(filePath, 'utf8');
	const rows = content
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean)
		.filter((line) => line.toLowerCase() !== 'ticker');
	return [...new Set(rows.map((value) => value.replace(/,.*$/, '').trim().toUpperCase()).filter(Boolean))];
}

function getTickers(): string[] {
	const tickersFilePath = process.env.TICKERS_FILE_PATH;
	if (tickersFilePath && fs.existsSync(tickersFilePath)) {
		const tickers = getTickersFromFile(tickersFilePath);
		if (tickers.length > 0) {
			return tickers;
		}
	}

	const explicitTickers = process.env.TICKERS;
	if (explicitTickers) {
		return explicitTickers
			.split(',')
			.map((ticker) => ticker.trim().toUpperCase())
			.filter(Boolean);
	}

	const v2Path = process.env.V2_FILE_PATH || DEFAULT_V2_PATH;
	if (fs.existsSync(v2Path)) {
		const rows = readCsv(v2Path);
		const symbolKey = rows.length > 0 && ('ticker' in rows[0] ? 'ticker' : 'symbol');
		if (!symbolKey) {
			throw new Error('v2 file exists but no ticker/symbol column found.');
		}
		return [...new Set(rows.map((row) => (row[symbolKey] || '').toUpperCase()).filter(Boolean))];
	}

	return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'];
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

async function fetchFinnhubNews(ticker: string, from: string, to: string): Promise<NewsRow[]> {
	const token = process.env.FINNHUB_API_KEY;
	if (!token) {
		return [];
	}

	const rows: NewsRow[] = [];
	let cursor = from;
	let rateLimited = false;
	while (cursor <= to) {
		const monthEnd = toIsoDate(
			new Date(
				Math.min(
					new Date(`${cursor}T00:00:00Z`).setUTCMonth(new Date(`${cursor}T00:00:00Z`).getUTCMonth() + 1),
					new Date(`${to}T00:00:00Z`).getTime()
				)
			)
		);
		const url = `https://finnhub.io/api/v1/company-news?symbol=${encodeURIComponent(ticker)}&from=${cursor}&to=${monthEnd}&token=${token}`;

		try {
			const response = await axios.get(url, { timeout: 15000 });
			const items = Array.isArray(response.data) ? response.data : [];

			for (const item of items) {
				const isoDate = item.datetime ? toIsoDate(new Date(item.datetime * 1000)) : '';
				if (!isoDate || !isInRange(isoDate, { start: from, end: to })) {
					continue;
				}
				const headline = String(item.headline || '').trim();
				if (!headline) {
					continue;
				}
				rows.push({
					date: isoDate,
					ticker,
					headline,
					sentiment_score: String(scoreHeadline(headline)),
					source: 'finnhub'
				});
			}
		} catch (error) {
			const status = typeof error === 'object' && error !== null && 'response' in error
				? Number((error as any).response?.status || 0)
				: 0;
			if (status === 429) {
				rateLimited = true;
				break;
			}
			const message = error instanceof Error ? error.message : 'unknown error';
			console.warn(`Finnhub fetch failed for ${ticker} (${cursor}..${monthEnd}): ${message}`);
		}

		const next = new Date(`${cursor}T00:00:00Z`);
		next.setUTCMonth(next.getUTCMonth() + 1);
		cursor = toIsoDate(next);
	}

	if (rateLimited) {
		console.warn(`Finnhub rate limit reached for ${ticker}; using Yahoo fallback for this ticker.`);
	}

	return rows;
}

async function fetchPolygonNews(ticker: string, start: string, end: string): Promise<NewsRow[]> {
	const apiKey = process.env.POLYGON_API_KEY || process.env.POLYGON_API_TOKEN;
	if (!apiKey) {
		return [];
	}

	const rows: NewsRow[] = [];
	let nextUrl: string | null = `https://api.polygon.io/v2/reference/news?ticker=${encodeURIComponent(
		ticker
	)}&published_utc.gte=${start}T00:00:00Z&published_utc.lte=${end}T23:59:59Z&order=asc&sort=published_utc&limit=1000&apiKey=${apiKey}`;
	let pages = 0;

	while (nextUrl && pages < 50) {
		pages += 1;
		try {
			const response: any = await axios.get(nextUrl, { timeout: 20000 });
			const items = Array.isArray(response.data?.results) ? (response.data.results as PolygonNewsItem[]) : [];

			for (const item of items) {
				const isoDate = item.published_utc ? toIsoDate(new Date(item.published_utc)) : '';
				if (!isoDate || !isInRange(isoDate, { start, end })) {
					continue;
				}
				const headline = String(item.title || '').replace(/\s+/g, ' ').trim();
				if (!headline) {
					continue;
				}
				rows.push({
					date: isoDate,
					ticker,
					headline,
					sentiment_score: String(scoreHeadline(headline)),
					source: 'polygon'
				});
			}

			const rawNextUrl: string = String(response.data?.next_url || '');
			nextUrl = rawNextUrl ? `${rawNextUrl}${rawNextUrl.includes('?') ? '&' : '?'}apiKey=${apiKey}` : null;
		} catch (error) {
			const status =
				typeof error === 'object' && error !== null && 'response' in error
					? Number((error as any).response?.status || 0)
					: 0;
			const message = error instanceof Error ? error.message : 'unknown error';
			if (status === 429) {
				console.warn(`Polygon rate limit reached for ${ticker}; falling through to Yahoo fallback.`);
				break;
			}
			console.warn(`Polygon fetch failed for ${ticker}: ${message}`);
			break;
		}
	}

	return rows;
}

async function fetchYahooNews(ticker: string, start: string, end: string): Promise<NewsRow[]> {
	const url = `https://query1.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(ticker)}&newsCount=100`;
	const response = await axios.get(url, { timeout: 15000 });
	const items = Array.isArray(response.data?.news) ? response.data.news : [];
	const rows: NewsRow[] = [];

	for (const item of items) {
		const isoDate = item.providerPublishTime ? toIsoDate(new Date(item.providerPublishTime * 1000)) : '';
		if (!isoDate || !isInRange(isoDate, { start, end })) {
			continue;
		}
		const headline = String(item.title || '').replace(/\s+/g, ' ').trim();
		if (!headline) {
			continue;
		}
		rows.push({
			date: isoDate,
			ticker,
			headline,
			sentiment_score: String(scoreHeadline(headline)),
			source: 'yahoo'
		});
	}

	return rows;
}

async function main(): Promise<void> {
	const range = getLastSixYearsRange();
	const tickers = getTickers();
	const allRows: NewsRow[] = [];

	for (const ticker of tickers) {
		let rows: NewsRow[] = [];
		try {
			rows = await fetchFinnhubNews(ticker, range.start, range.end);
		} catch (error) {
			const message = error instanceof Error ? error.message : 'unknown error';
			console.warn(`Finnhub unavailable for ${ticker}: ${message}`);
		}
		if (rows.length === 0) {
			try {
				rows = await fetchPolygonNews(ticker, range.start, range.end);
			} catch (error) {
				const message = error instanceof Error ? error.message : 'unknown error';
				console.warn(`Polygon unavailable for ${ticker}: ${message}`);
			}
		}
		if (rows.length === 0) {
			rows = await fetchYahooNews(ticker, range.start, range.end);
		}
		allRows.push(...rows);
	}

	const unique = new Map<string, NewsRow>();
	for (const row of allRows) {
		unique.set(`${row.ticker}|${row.date}|${row.headline}`, row);
	}

	writeCsv(
		OUTPUT_PATH,
		['date', 'ticker', 'headline', 'sentiment_score', 'source'],
		Array.from(unique.values())
			.sort((a, b) => (a.date === b.date ? a.ticker.localeCompare(b.ticker) : a.date.localeCompare(b.date)))
			.map((row) => ({
				date: row.date,
				ticker: row.ticker,
				headline: row.headline,
				sentiment_score: row.sentiment_score,
				source: row.source
			}))
	);

	console.log(`Saved ${unique.size} news rows to ${OUTPUT_PATH}.`);
	if (!process.env.FINNHUB_API_KEY && !process.env.POLYGON_API_KEY && !process.env.POLYGON_API_TOKEN) {
		console.log(
			'Note: no FINNHUB_API_KEY or POLYGON_API_KEY/POLYGON_API_TOKEN found (including .env); using Yahoo fallback, which is mostly recent news.'
		);
	}
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

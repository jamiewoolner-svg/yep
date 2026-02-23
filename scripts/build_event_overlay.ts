import fs from 'fs';
import path from 'path';
import { readCsv, writeCsv, CsvRow } from './lib/csv';

const DATA_DIR = path.join(__dirname, '../data');
const V2_PATH = process.env.V2_FILE_PATH || path.join(DATA_DIR, 'v2_trades.csv');
const EARNINGS_PATH = process.env.EARNINGS_FILE_PATH || path.join(DATA_DIR, 'earnings_calendar.csv');
const MACRO_PATH = process.env.MACRO_FILE_PATH || path.join(DATA_DIR, 'macro_events.csv');
const NEWS_PATH = process.env.NEWS_FILE_PATH || path.join(DATA_DIR, 'news_sentiment.csv');
const NEWS_DAILY_PATH = process.env.NEWS_DAILY_FILE_PATH || path.join(DATA_DIR, 'news_daily_6y.csv');
const OUTPUT_PATH = process.env.OVERLAY_OUT_PATH || path.join(DATA_DIR, 'event_overlay.csv');

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

function buildMacroSet(rows: CsvRow[]): Set<string> {
	const set = new Set<string>();
	for (const row of rows) {
		const date = normalizeDate(row.event_date || row.date || '');
		const name = (row.event_name || '').toUpperCase();
		if (date && name) {
			set.add(`${date}|${name}`);
		}
	}
	return set;
}

function buildEarningsIndex(rows: CsvRow[]): Map<string, string[]> {
	const index = new Map<string, string[]>();
	for (const row of rows) {
		const ticker = (row.ticker || row.symbol || '').toUpperCase();
		const date = normalizeDate(row.earnings_date || row.date || '');
		if (!ticker || !date) {
			continue;
		}
		const current = index.get(ticker) || [];
		current.push(date);
		index.set(ticker, current);
	}
	for (const [ticker, dates] of index) {
		index.set(ticker, [...new Set(dates)].sort());
	}
	return index;
}

function buildNewsIndex(rows: CsvRow[]): Map<string, CsvRow[]> {
	const index = new Map<string, CsvRow[]>();
	for (const row of rows) {
		const ticker = (row.ticker || row.symbol || '').toUpperCase();
		const date = normalizeDate(row.date || row.news_date || '');
		if (!ticker || !date) {
			continue;
		}
		const key = `${ticker}|${date}`;
		const current = index.get(key) || [];
		current.push(row);
		index.set(key, current);
	}
	return index;
}

function buildNewsDailyIndex(rows: CsvRow[]): Map<string, CsvRow> {
	const index = new Map<string, CsvRow>();
	for (const row of rows) {
		const ticker = (row.ticker || row.symbol || '').toUpperCase();
		const date = normalizeDate(row.date || '');
		if (!ticker || !date) {
			continue;
		}
		index.set(`${ticker}|${date}`, row);
	}
	return index;
}

function hasEarningsWithin(ticker: string, tradeDate: string, earningsByTicker: Map<string, string[]>, windowDays: number): boolean {
	const earningsDates = earningsByTicker.get(ticker) || [];
	for (const earningsDate of earningsDates) {
		const diff = Math.abs(dateDiffDays(earningsDate, tradeDate));
		if (diff <= windowDays) {
			return true;
		}
	}
	return false;
}

function avgSentiment(rows: CsvRow[]): number {
	if (rows.length === 0) {
		return 0;
	}
	const values = rows
		.map((row) => Number(row.sentiment_score || 0))
		.filter((value) => Number.isFinite(value));
	if (values.length === 0) {
		return 0;
	}
	const sum = values.reduce((a, b) => a + b, 0);
	return Number((sum / values.length).toFixed(4));
}

function main(): void {
	if (!fs.existsSync(V2_PATH)) {
		throw new Error(
			[
				`Missing v2 input file: ${V2_PATH}`,
				'Export your v2 table to CSV and place it at data/v2_trades.csv,',
				'or set V2_FILE_PATH=/absolute/path/to/v2.csv when running this script.'
			].join(' ')
		);
	}

	const v2Rows = readCsv(V2_PATH);
	const macroRows = readCsv(MACRO_PATH);
	const earningsRows = readCsv(EARNINGS_PATH);
	const newsRows = readCsv(NEWS_PATH);
	const newsDailyRows = fs.existsSync(NEWS_DAILY_PATH) ? readCsv(NEWS_DAILY_PATH) : [];

	if (v2Rows.length === 0) {
		throw new Error(`No rows found in ${V2_PATH}`);
	}

	const sample = v2Rows[0];
	const dateCol = detectColumn(sample, ['date', 'trade_date', 'day0_date', 'opened_at']);
	const tickerCol = detectColumn(sample, ['ticker', 'symbol']);

	const macroSet = buildMacroSet(macroRows);
	const earningsByTicker = buildEarningsIndex(earningsRows);
	const newsByDayTicker = buildNewsIndex(newsRows);
	const newsDailyByKey = buildNewsDailyIndex(newsDailyRows);

	const outRows: CsvRow[] = v2Rows.map((trade) => {
		const tradeDate = normalizeDate(trade[dateCol]);
		const ticker = String(trade[tickerCol] || '').toUpperCase();
		const newsKey = `${ticker}|${tradeDate}`;
		const sameDayNews = newsByDayTicker.get(newsKey) || [];
		const dailyNews = newsDailyByKey.get(newsKey);
		const macroFomc = macroSet.has(`${tradeDate}|FOMC`) ? '1' : '0';
		const macroCpi = macroSet.has(`${tradeDate}|CPI`) ? '1' : '0';
		const macroNfp = macroSet.has(`${tradeDate}|NFP`) ? '1' : '0';

		return {
			...trade,
			date: tradeDate,
			ticker,
			is_earnings_window_48h: hasEarningsWithin(ticker, tradeDate, earningsByTicker, 2) ? '1' : '0',
			is_fomc_day: macroFomc,
			is_cpi_day: macroCpi,
			is_nfp_day: macroNfp,
			news_count_day0: String(dailyNews?.news_count || sameDayNews.length),
			avg_sentiment_day0: String(dailyNews?.avg_sentiment || avgSentiment(sameDayNews)),
			news_min_sentiment_day0: String(dailyNews?.min_sentiment || 0),
			news_max_sentiment_day0: String(dailyNews?.max_sentiment || 0),
			news_positive_count_day0: String(dailyNews?.positive_count || 0),
			news_negative_count_day0: String(dailyNews?.negative_count || 0),
			news_neutral_count_day0: String(dailyNews?.neutral_count || 0),
			has_high_impact_news_day0: String(dailyNews?.has_high_impact_news || 0),
			is_high_impact_macro_day: macroFomc === '1' || macroCpi === '1' || macroNfp === '1' ? '1' : '0'
		};
	});

	const headers = Array.from(new Set(outRows.flatMap((row) => Object.keys(row))));
	writeCsv(OUTPUT_PATH, headers, outRows);
	console.log(`Saved overlay with ${outRows.length} rows to ${OUTPUT_PATH}.`);
}

main();

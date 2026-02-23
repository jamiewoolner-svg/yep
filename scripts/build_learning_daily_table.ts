import path from 'path';
import { readCsv, writeCsv, CsvRow } from './lib/csv';
import { getLastSixYearsRange, toIsoDate } from './lib/dates';

const DATA_DIR = path.join(__dirname, '../data');
const TICKERS_PATH = path.join(DATA_DIR, 'qqq_tickers.csv');
const NEWS_DAILY_PATH = path.join(DATA_DIR, 'news_daily_6y.csv');
const NEWS_EVENTS_PATH = path.join(DATA_DIR, 'news_events_6y.csv');
const EARNINGS_PATH = path.join(DATA_DIR, 'earnings_calendar.csv');
const MACRO_PATH = path.join(DATA_DIR, 'macro_events.csv');
const OUTPUT_PATH = path.join(DATA_DIR, 'learning_daily_qqq_6y.csv');

function getTickers(): string[] {
	const rows = readCsv(TICKERS_PATH);
	const tickers = rows.map((row) => (row.ticker || '').toUpperCase()).filter(Boolean);
	return [...new Set(tickers)].sort();
}

function dateDiffDays(a: string, b: string): number {
	const da = new Date(`${a}T00:00:00Z`).getTime();
	const db = new Date(`${b}T00:00:00Z`).getTime();
	return Math.round((da - db) / 86400000);
}

function buildDateSeries(start: string, end: string): string[] {
	const out: string[] = [];
	let cursor = new Date(`${start}T00:00:00Z`);
	const endDate = new Date(`${end}T00:00:00Z`);
	while (cursor <= endDate) {
		out.push(toIsoDate(cursor));
		cursor.setUTCDate(cursor.getUTCDate() + 1);
	}
	return out;
}

function buildMacroSet(rows: CsvRow[]): Set<string> {
	const set = new Set<string>();
	for (const row of rows) {
		const date = row.event_date || row.date;
		const eventName = (row.event_name || '').toUpperCase();
		if (date && eventName) {
			set.add(`${date}|${eventName}`);
		}
	}
	return set;
}

function buildEarningsIndex(rows: CsvRow[]): Map<string, string[]> {
	const map = new Map<string, string[]>();
	for (const row of rows) {
		const ticker = (row.ticker || row.symbol || '').toUpperCase();
		const date = row.earnings_date || row.date;
		if (!ticker || !date) {
			continue;
		}
		const current = map.get(ticker) || [];
		current.push(date);
		map.set(ticker, current);
	}
	for (const [ticker, dates] of map) {
		map.set(ticker, [...new Set(dates)].sort());
	}
	return map;
}

function hasEarningsWithin(ticker: string, day: string, earningsByTicker: Map<string, string[]>, windowDays: number): boolean {
	const dates = earningsByTicker.get(ticker) || [];
	for (const earningsDate of dates) {
		if (Math.abs(dateDiffDays(earningsDate, day)) <= windowDays) {
			return true;
		}
	}
	return false;
}

function buildNewsDailyIndex(rows: CsvRow[]): Map<string, CsvRow> {
	const map = new Map<string, CsvRow>();
	for (const row of rows) {
		const ticker = (row.ticker || row.symbol || '').toUpperCase();
		const date = row.date || '';
		if (!ticker || !date) {
			continue;
		}
		map.set(`${ticker}|${date}`, row);
	}
	return map;
}

function buildNewsHeadlinesIndex(rows: CsvRow[]): Map<string, string> {
	const map = new Map<string, string[]>();
	for (const row of rows) {
		const ticker = (row.ticker || row.symbol || '').toUpperCase();
		const date = row.date || '';
		const headline = String(row.headline || '').trim();
		if (!ticker || !date || !headline) {
			continue;
		}
		const key = `${ticker}|${date}`;
		const current = map.get(key) || [];
		current.push(headline.replace(/\s+/g, ' '));
		map.set(key, current);
	}

	const deduped = new Map<string, string>();
	for (const [key, headlines] of map) {
		deduped.set(key, [...new Set(headlines)].join(' || '));
	}

	return deduped;
}

function main(): void {
	const range = getLastSixYearsRange();
	const dates = buildDateSeries(range.start, range.end);
	const tickers = getTickers();
	const macroSet = buildMacroSet(readCsv(MACRO_PATH));
	const earningsByTicker = buildEarningsIndex(readCsv(EARNINGS_PATH));
	const newsByKey = buildNewsDailyIndex(readCsv(NEWS_DAILY_PATH));
	const newsHeadlinesByKey = buildNewsHeadlinesIndex(readCsv(NEWS_EVENTS_PATH));

	const rows: CsvRow[] = [];
	for (const day of dates) {
		const isFomc = macroSet.has(`${day}|FOMC`) ? '1' : '0';
		const isCpi = macroSet.has(`${day}|CPI`) ? '1' : '0';
		const isNfp = macroSet.has(`${day}|NFP`) ? '1' : '0';
		const isHighImpactMacro = isFomc === '1' || isCpi === '1' || isNfp === '1' ? '1' : '0';

		for (const ticker of tickers) {
			const news = newsByKey.get(`${ticker}|${day}`);
			const newsHeadlines = newsHeadlinesByKey.get(`${ticker}|${day}`) || '';
			rows.push({
				date: day,
				ticker,
				is_earnings_window_48h: hasEarningsWithin(ticker, day, earningsByTicker, 2) ? '1' : '0',
				is_fomc_day: isFomc,
				is_cpi_day: isCpi,
				is_nfp_day: isNfp,
				is_high_impact_macro_day: isHighImpactMacro,
				news_count_day0: String(news?.news_count || 0),
				avg_sentiment_day0: String(news?.avg_sentiment || 0),
				news_min_sentiment_day0: String(news?.min_sentiment || 0),
				news_max_sentiment_day0: String(news?.max_sentiment || 0),
				news_positive_count_day0: String(news?.positive_count || 0),
				news_negative_count_day0: String(news?.negative_count || 0),
				news_neutral_count_day0: String(news?.neutral_count || 0),
				has_high_impact_news_day0: String(news?.has_high_impact_news || 0),
				news_headlines_day0: newsHeadlines
			});
		}
	}

	writeCsv(
		OUTPUT_PATH,
		[
			'date',
			'ticker',
			'is_earnings_window_48h',
			'is_fomc_day',
			'is_cpi_day',
			'is_nfp_day',
			'is_high_impact_macro_day',
			'news_count_day0',
			'avg_sentiment_day0',
			'news_min_sentiment_day0',
			'news_max_sentiment_day0',
			'news_positive_count_day0',
			'news_negative_count_day0',
			'news_neutral_count_day0',
			'has_high_impact_news_day0',
			'news_headlines_day0'
		],
		rows
	);

	console.log(`Saved ${rows.length} rows to ${OUTPUT_PATH}.`);
}

main();

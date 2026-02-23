import fs from 'fs';
import path from 'path';
import axios from 'axios';
import { readCsv, writeCsv } from './lib/csv';
import { getLastSixYearsRange, isInRange, toIsoDate, toUnixSeconds } from './lib/dates';

type EarningsRow = {
	ticker: string;
	earnings_date: string;
	release_time: string;
};

const OUTPUT_PATH = path.join(__dirname, '../data/earnings_calendar.csv');
const DEFAULT_V2_PATH = path.join(__dirname, '../data/v2_trades.csv');
const SEC_TICKER_URL = 'https://www.sec.gov/include/ticker.txt';
const SEC_SUBMISSIONS_URL = 'https://data.sec.gov/submissions';
const SEC_HEADERS = {
	'User-Agent': `stock-scanner/1.0 (${process.env.CONTACT_EMAIL || 'research@example.com'})`,
	Accept: 'application/json'
};

function getTickers(): string[] {
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

async function fetchEarningsForTicker(ticker: string, period1: number, period2: number): Promise<EarningsRow[]> {
	const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?period1=${period1}&period2=${period2}&interval=1d&events=earnings`;
	try {
		const response = await axios.get(url, {
			timeout: 15000,
			headers: {
				'User-Agent':
					'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
				Accept: 'application/json,text/plain,*/*'
			}
		});
		const earnings = response.data?.chart?.result?.[0]?.events?.earnings ?? {};
		const rows: EarningsRow[] = [];
		for (const event of Object.values(earnings) as Array<{ date?: number; hour?: number }>) {
			if (!event?.date) {
				continue;
			}
			const earningsDate = toIsoDate(new Date(event.date * 1000));
			const releaseTime = event.hour && event.hour < 12 ? 'BMO' : event.hour ? 'AMC' : 'UNSPECIFIED';
			rows.push({ ticker, earnings_date: earningsDate, release_time: releaseTime });
		}
		return rows;
	} catch (error) {
		const message = error instanceof Error ? error.message : 'unknown error';
		console.error(`Failed earnings fetch for ${ticker}: ${message}`);
		return [];
	}
}

async function fetchTickerToCikMap(): Promise<Map<string, string>> {
	const response = await axios.get(SEC_TICKER_URL, { timeout: 20000, headers: SEC_HEADERS });
	const lines = String(response.data)
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean);
	const map = new Map<string, string>();
	for (const line of lines) {
		const [ticker, cikRaw] = line.split('\t');
		if (!ticker || !cikRaw) {
			continue;
		}
		map.set(ticker.toUpperCase(), cikRaw.padStart(10, '0'));
	}
	return map;
}

async function fetchSecEarningsFallback(ticker: string, rangeStart: string, rangeEnd: string, cikMap: Map<string, string>): Promise<EarningsRow[]> {
	const cik = cikMap.get(ticker);
	if (!cik) {
		return [];
	}
	const url = `${SEC_SUBMISSIONS_URL}/CIK${cik}.json`;
	try {
		const response = await axios.get(url, { timeout: 20000, headers: SEC_HEADERS });
		const recent = response.data?.filings?.recent;
		const forms: string[] = recent?.form || [];
		const filingDates: string[] = recent?.filingDate || [];
		const rows: EarningsRow[] = [];
		for (let i = 0; i < forms.length; i++) {
			const form = forms[i];
			const filingDate = filingDates[i];
			if (!filingDate || !['10-Q', '10-K', '20-F', '6-K'].includes(form)) {
				continue;
			}
			if (filingDate >= rangeStart && filingDate <= rangeEnd) {
				rows.push({ ticker, earnings_date: filingDate, release_time: 'UNSPECIFIED' });
			}
		}
		return rows;
	} catch (error) {
		const message = error instanceof Error ? error.message : 'unknown error';
		console.error(`SEC fallback failed for ${ticker}: ${message}`);
		return [];
	}
}

async function main(): Promise<void> {
	const range = getLastSixYearsRange();
	const tickers = getTickers();
	const period1 = toUnixSeconds(range.start);
	const period2 = toUnixSeconds(range.end) + 86400;
	const cikMap = await fetchTickerToCikMap();

	const allRows: EarningsRow[] = [];
	for (const ticker of tickers) {
		let rows = await fetchEarningsForTicker(ticker, period1, period2);
		if (rows.length === 0) {
			rows = await fetchSecEarningsFallback(ticker, range.start, range.end, cikMap);
		}
		allRows.push(...rows.filter((row) => isInRange(row.earnings_date, range)));
	}

	const unique = new Map<string, EarningsRow>();
	for (const row of allRows) {
		unique.set(`${row.ticker}|${row.earnings_date}`, row);
	}

	writeCsv(
		OUTPUT_PATH,
		['ticker', 'earnings_date', 'release_time'],
		Array.from(unique.values())
			.sort((a, b) => (a.ticker === b.ticker ? a.earnings_date.localeCompare(b.earnings_date) : a.ticker.localeCompare(b.ticker)))
			.map((row) => ({
				ticker: row.ticker,
				earnings_date: row.earnings_date,
				release_time: row.release_time
			}))
	);

	console.log(`Saved ${unique.size} earnings rows to ${OUTPUT_PATH} for ${tickers.length} tickers.`);
	console.log('Source preference: Yahoo Finance, with SEC filings fallback when Yahoo data is unavailable.');
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

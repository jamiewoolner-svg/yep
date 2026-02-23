import path from 'path';
import axios from 'axios';
import { writeCsv } from './lib/csv';
import { getLastSixYearsRange, isInRange } from './lib/dates';

type MacroRow = {
	event_date: string;
	event_name: 'FOMC' | 'CPI' | 'NFP';
	impact_level: 'High';
};

const OUTPUT_PATH = path.join(__dirname, '../data/macro_events.csv');
const FOMC_URL = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm';
const CPI_ARCHIVE_URL = 'https://www.bls.gov/bls/news-release/cpi.htm';
const NFP_ARCHIVE_URL = 'https://www.bls.gov/bls/news-release/empsit.htm';
const REQUEST_HEADERS = {
	'User-Agent':
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
	Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
	'Accept-Language': 'en-US,en;q=0.9',
	Referer: 'https://www.bls.gov/'
};

function unique(rows: MacroRow[]): MacroRow[] {
	const map = new Map<string, MacroRow>();
	for (const row of rows) {
		map.set(`${row.event_name}|${row.event_date}`, row);
	}
	return Array.from(map.values()).sort((a, b) => a.event_date.localeCompare(b.event_date));
}

function extractBlsDates(html: string, prefix: 'cpi' | 'empsit'): string[] {
	const dates = new Set<string>();
	const regex = new RegExp(`${prefix}_(\\d{8})\\.htm`, 'g');
	let match: RegExpExecArray | null;
	while ((match = regex.exec(html)) !== null) {
		const token = match[1];
		const month = token.slice(0, 2);
		const day = token.slice(2, 4);
		const year = token.slice(4, 8);
		dates.add(`${year}-${month}-${day}`);
	}
	return Array.from(dates);
}

function extractFomcDates(html: string): string[] {
	const dates = new Set<string>();
	const regex = /monetary(\d{8})a\.htm/g;
	let match: RegExpExecArray | null;
	while ((match = regex.exec(html)) !== null) {
		const token = match[1];
		dates.add(`${token.slice(0, 4)}-${token.slice(4, 6)}-${token.slice(6, 8)}`);
	}
	return Array.from(dates);
}

async function fetchMacroEvents(): Promise<void> {
	const range = getLastSixYearsRange();
	const [fomcRes, cpiRes, nfpRes] = await Promise.all([
		axios.get(FOMC_URL, { timeout: 15000, headers: REQUEST_HEADERS }),
		axios.get(CPI_ARCHIVE_URL, { timeout: 15000, headers: REQUEST_HEADERS }),
		axios.get(NFP_ARCHIVE_URL, { timeout: 15000, headers: REQUEST_HEADERS })
	]);

	const fomcRows: MacroRow[] = extractFomcDates(String(fomcRes.data))
		.filter((eventDate) => isInRange(eventDate, range))
		.map((eventDate) => ({ event_date: eventDate, event_name: 'FOMC', impact_level: 'High' }));

	const cpiRows: MacroRow[] = extractBlsDates(String(cpiRes.data), 'cpi')
		.filter((eventDate) => isInRange(eventDate, range))
		.map((eventDate) => ({ event_date: eventDate, event_name: 'CPI', impact_level: 'High' }));

	const nfpRows: MacroRow[] = extractBlsDates(String(nfpRes.data), 'empsit')
		.filter((eventDate) => isInRange(eventDate, range))
		.map((eventDate) => ({ event_date: eventDate, event_name: 'NFP', impact_level: 'High' }));

	const rows = unique([...fomcRows, ...cpiRows, ...nfpRows]);

	writeCsv(
		OUTPUT_PATH,
		['event_date', 'event_name', 'impact_level'],
		rows.map((row) => ({
			event_date: row.event_date,
			event_name: row.event_name,
			impact_level: row.impact_level
		}))
	);

	console.log(`Saved ${rows.length} macro events to ${OUTPUT_PATH}.`);
}

fetchMacroEvents().catch((error) => {
	console.error(error);
	process.exit(1);
});

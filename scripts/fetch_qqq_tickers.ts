import fs from 'fs';
import path from 'path';
import axios from 'axios';

const OUTPUT_PATH = path.join(__dirname, '../data/qqq_tickers.csv');
const WIKI_RAW_URL = 'https://en.wikipedia.org/w/index.php?title=Nasdaq-100&action=raw';

async function main(): Promise<void> {
	const response = await axios.get(WIKI_RAW_URL, {
		timeout: 20000,
		headers: {
			'User-Agent': 'stock-scanner/1.0 (research@example.com)'
		}
	});

	const raw = String(response.data);
	const tableStart = raw.indexOf('id="constituents"');
	if (tableStart < 0) {
		throw new Error('Unable to locate constituents table in source text.');
	}

	const tableBody = raw.slice(tableStart);
	const tableEnd = tableBody.indexOf('\n|}');
	if (tableEnd < 0) {
		throw new Error('Unable to find constituents table end marker.');
	}

	const table = tableBody.slice(0, tableEnd);
	const matches = [...table.matchAll(/^\|\s*([A-Z.]+)\s*\|\|/gm)];
	const tickers = [...new Set(matches.map((match) => match[1].trim().toUpperCase()))];

	if (tickers.length < 95) {
		throw new Error(`Parsed only ${tickers.length} tickers; expected around 100.`);
	}

	const lines = ['ticker', ...tickers.sort()];
	fs.writeFileSync(OUTPUT_PATH, `${lines.join('\n')}\n`);
	console.log(`Saved ${tickers.length} Nasdaq-100 tickers to ${OUTPUT_PATH}.`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

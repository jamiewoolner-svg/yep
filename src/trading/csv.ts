import fs from 'fs';

export type CsvRow = Record<string, string>;

function parseCsvLine(line: string): string[] {
	const values: string[] = [];
	let current = '';
	let inQuotes = false;

	for (let i = 0; i < line.length; i++) {
		const char = line[i];
		if (char === '"') {
			if (inQuotes && line[i + 1] === '"') {
				current += '"';
				i++;
			} else {
				inQuotes = !inQuotes;
			}
		} else if (char === ',' && !inQuotes) {
			values.push(current);
			current = '';
		} else {
			current += char;
		}
	}

	values.push(current);
	return values;
}

export function readCsv(filePath: string): CsvRow[] {
	const content = fs.readFileSync(filePath, 'utf8');
	const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
	if (lines.length === 0) {
		return [];
	}

	const headers = parseCsvLine(lines[0]);
	const rows: CsvRow[] = [];

	for (let i = 1; i < lines.length; i++) {
		const values = parseCsvLine(lines[i]);
		const row: CsvRow = {};
		for (let h = 0; h < headers.length; h++) {
			row[headers[h]] = values[h] ?? '';
		}
		rows.push(row);
	}

	return rows;
}

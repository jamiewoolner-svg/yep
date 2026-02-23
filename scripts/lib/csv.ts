import fs from 'fs';

export type CsvRow = Record<string, string>;

function escapeCsv(value: string): string {
	if (value.includes('"')) {
		value = value.replace(/"/g, '""');
	}
	if (/[",\n]/.test(value)) {
		return `"${value}"`;
	}
	return value;
}

export function writeCsv(filePath: string, headers: string[], rows: CsvRow[]): void {
	const lines: string[] = [headers.join(',')];
	for (const row of rows) {
		lines.push(headers.map((header) => escapeCsv(String(row[header] ?? ''))).join(','));
	}
	fs.writeFileSync(filePath, lines.join('\n'));
}

export function parseCsv(content: string): CsvRow[] {
	const rows: CsvRow[] = [];
	const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
	if (lines.length === 0) {
		return rows;
	}

	const headers = parseCsvLine(lines[0]);
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

export function readCsv(filePath: string): CsvRow[] {
	const content = fs.readFileSync(filePath, 'utf8');
	return parseCsv(content);
}

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

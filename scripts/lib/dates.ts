export interface DateRange {
	start: string;
	end: string;
}

export function toIsoDate(input: Date): string {
	return input.toISOString().slice(0, 10);
}

export function getLastSixYearsRange(referenceDate: Date = new Date()): DateRange {
	const end = toIsoDate(referenceDate);
	const startDate = new Date(referenceDate);
	startDate.setUTCFullYear(startDate.getUTCFullYear() - 6);
	const start = toIsoDate(startDate);
	return { start, end };
}

export function toUnixSeconds(isoDate: string): number {
	return Math.floor(new Date(`${isoDate}T00:00:00Z`).getTime() / 1000);
}

export function isInRange(date: string, range: DateRange): boolean {
	return date >= range.start && date <= range.end;
}

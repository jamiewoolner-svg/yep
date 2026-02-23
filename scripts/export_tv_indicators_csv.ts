import fs from 'fs';
import path from 'path';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

type Candle = {
	date: string;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
};

type OutRow = {
	date: string;
	open: string;
	high: string;
	low: string;
	close: string;
	volume: string;
	bb_upper_21_2: string;
	bb_mid_21: string;
	bb_lower_21_2: string;
	sma_200: string;
	sma_89: string;
	sma_50: string;
	sma_3: string;
	sma_2: string;
	macd_line_8_21: string;
	macd_signal_5: string;
	macd_hist: string;
	rsi_13: string;
	stoch_rsi_raw_21: string;
	stoch_rsi_k_3: string;
	stoch_rsi_d_5: string;
	plus_di_5: string;
	minus_di_5: string;
	adx_13: string;
};

const TICKER = process.env.TICKER || 'AAPL';
const FROM = process.env.FROM_DATE || '2025-11-14';
const TO = process.env.TO_DATE || '2026-02-25';
const OUTPUT_PATH = process.env.OUT_PATH || path.join(__dirname, '../data/aapl_tv_indicators_2025-11-14_to_2026-02-25.csv');
const WARMUP_CALENDAR_DAYS = Number(process.env.WARMUP_CALENDAR_DAYS || 450);

function shiftIsoDateByDays(isoDate: string, days: number): string {
	const d = new Date(`${isoDate}T00:00:00Z`);
	d.setUTCDate(d.getUTCDate() + days);
	return d.toISOString().slice(0, 10);
}

function toIsoDate(ms: number): string {
	return new Date(ms).toISOString().slice(0, 10);
}

function round(value: number, digits = 6): string {
	if (!Number.isFinite(value)) {
		return '';
	}
	return value.toFixed(digits);
}

function sma(values: number[], index: number, length: number): number {
	if (index + 1 < length) {
		return NaN;
	}
	let sum = 0;
	for (let i = index - length + 1; i <= index; i++) {
		sum += values[i];
	}
	return sum / length;
}

function stddev(values: number[], index: number, length: number): number {
	if (index + 1 < length) {
		return NaN;
	}
	const mean = sma(values, index, length);
	let acc = 0;
	for (let i = index - length + 1; i <= index; i++) {
		const delta = values[i] - mean;
		acc += delta * delta;
	}
	return Math.sqrt(acc / length);
}

function emaSeries(values: number[], length: number): number[] {
	const out: number[] = new Array(values.length).fill(NaN);
	if (values.length === 0) {
		return out;
	}
	const alpha = 2 / (length + 1);
	let seeded = false;
	let prev = NaN;

	for (let i = 0; i < values.length; i++) {
		const value = values[i];
		if (!seeded) {
			const base = sma(values, i, length);
			if (Number.isFinite(base)) {
				prev = base;
				out[i] = base;
				seeded = true;
			}
			continue;
		}
		prev = alpha * value + (1 - alpha) * prev;
		out[i] = prev;
	}

	return out;
}

function rsiSeries(values: number[], length: number): number[] {
	const out: number[] = new Array(values.length).fill(NaN);
	if (values.length <= length) {
		return out;
	}

	let gainSum = 0;
	let lossSum = 0;
	for (let i = 1; i <= length; i++) {
		const change = values[i] - values[i - 1];
		gainSum += Math.max(change, 0);
		lossSum += Math.max(-change, 0);
	}

	let avgGain = gainSum / length;
	let avgLoss = lossSum / length;
	if (avgLoss === 0) {
		out[length] = 100;
	} else {
		const rs = avgGain / avgLoss;
		out[length] = 100 - 100 / (1 + rs);
	}

	for (let i = length + 1; i < values.length; i++) {
		const change = values[i] - values[i - 1];
		const gain = Math.max(change, 0);
		const loss = Math.max(-change, 0);
		avgGain = (avgGain * (length - 1) + gain) / length;
		avgLoss = (avgLoss * (length - 1) + loss) / length;
		if (avgLoss === 0) {
			out[i] = 100;
		} else {
			const rs = avgGain / avgLoss;
			out[i] = 100 - 100 / (1 + rs);
		}
	}

	return out;
}

function taDmi(high: number[], low: number[], close: number[], diLength: number, adxSmoothing: number): { plusDi: number[]; minusDi: number[]; adx: number[] } {
	const n = close.length;
	const tr: number[] = new Array(n).fill(NaN);
	const plusDm: number[] = new Array(n).fill(0);
	const minusDm: number[] = new Array(n).fill(0);

	for (let i = 1; i < n; i++) {
		const upMove = high[i] - high[i - 1];
		const downMove = low[i - 1] - low[i];
		plusDm[i] = upMove > downMove && upMove > 0 ? upMove : 0;
		minusDm[i] = downMove > upMove && downMove > 0 ? downMove : 0;

		const range1 = high[i] - low[i];
		const range2 = Math.abs(high[i] - close[i - 1]);
		const range3 = Math.abs(low[i] - close[i - 1]);
		tr[i] = Math.max(range1, range2, range3);
	}

	const smTr = wilderSmooth(tr, diLength);
	const smPlusDm = wilderSmooth(plusDm, diLength);
	const smMinusDm = wilderSmooth(minusDm, diLength);

	const plusDi: number[] = new Array(n).fill(NaN);
	const minusDi: number[] = new Array(n).fill(NaN);
	const dx: number[] = new Array(n).fill(NaN);

	for (let i = 0; i < n; i++) {
		if (!Number.isFinite(smTr[i]) || smTr[i] === 0) {
			continue;
		}
		plusDi[i] = (100 * smPlusDm[i]) / smTr[i];
		minusDi[i] = (100 * smMinusDm[i]) / smTr[i];
		const sum = plusDi[i] + minusDi[i];
		dx[i] = sum === 0 ? 0 : (100 * Math.abs(plusDi[i] - minusDi[i])) / sum;
	}

	const adx = wilderSmooth(dx, adxSmoothing);
	return { plusDi, minusDi, adx };
}

function wilderSmooth(values: number[], length: number): number[] {
	const out: number[] = new Array(values.length).fill(NaN);
	if (values.length <= length) {
		return out;
	}

	let sum = 0;
	for (let i = 1; i <= length; i++) {
		sum += Number.isFinite(values[i]) ? values[i] : 0;
	}
	out[length] = sum / length;

	for (let i = length + 1; i < values.length; i++) {
		const current = Number.isFinite(values[i]) ? values[i] : 0;
		out[i] = (out[i - 1] * (length - 1) + current) / length;
	}

	return out;
}

function minWindow(values: number[], index: number, length: number): number {
	if (index + 1 < length) {
		return NaN;
	}
	let m = Number.POSITIVE_INFINITY;
	for (let i = index - length + 1; i <= index; i++) {
		if (values[i] < m) {
			m = values[i];
		}
	}
	return m;
}

function maxWindow(values: number[], index: number, length: number): number {
	if (index + 1 < length) {
		return NaN;
	}
	let m = Number.NEGATIVE_INFINITY;
	for (let i = index - length + 1; i <= index; i++) {
		if (values[i] > m) {
			m = values[i];
		}
	}
	return m;
}

function writeCsv(filePath: string, headers: (keyof OutRow)[], rows: OutRow[]): void {
	const lines: string[] = [headers.join(',')];
	for (const row of rows) {
		lines.push(headers.map((header) => row[header]).join(','));
	}
	fs.writeFileSync(filePath, `${lines.join('\n')}\n`);
}

async function fetchPolygonDailyCandles(ticker: string, from: string, to: string): Promise<Candle[]> {
	const apiKey = process.env.POLYGON_API_KEY || process.env.POLYGON_API_TOKEN;
	if (!apiKey) {
		throw new Error('Missing POLYGON_API_KEY/POLYGON_API_TOKEN in environment.');
	}

	const url = `https://api.polygon.io/v2/aggs/ticker/${encodeURIComponent(ticker)}/range/1/day/${from}/${to}`;
	const response: any = await axios.get(url, {
		timeout: 20000,
		params: {
			adjusted: 'true',
			sort: 'asc',
			limit: 50000,
			apiKey
		}
	});

	const results = Array.isArray(response.data?.results) ? response.data.results : [];
	return results.map((bar: any) => ({
		date: toIsoDate(Number(bar.t)),
		open: Number(bar.o),
		high: Number(bar.h),
		low: Number(bar.l),
		close: Number(bar.c),
		volume: Number(bar.v)
	}));
}

async function main(): Promise<void> {
	const fetchFrom = shiftIsoDateByDays(FROM, -WARMUP_CALENDAR_DAYS);
	const candles = await fetchPolygonDailyCandles(TICKER, fetchFrom, TO);
	if (candles.length === 0) {
		throw new Error('No candles returned from Polygon for requested range.');
	}

	const close = candles.map((c) => c.close);
	const high = candles.map((c) => c.high);
	const low = candles.map((c) => c.low);

	const macdFast = emaSeries(close, 8);
	const macdSlow = emaSeries(close, 21);
	const macdLine: number[] = close.map((_, i) => (Number.isFinite(macdFast[i]) && Number.isFinite(macdSlow[i]) ? macdFast[i] - macdSlow[i] : NaN));
	const macdSignal = emaSeries(macdLine.map((v) => (Number.isFinite(v) ? v : 0)), 5);
	const macdHist = macdLine.map((v, i) => (Number.isFinite(v) && Number.isFinite(macdSignal[i]) ? v - macdSignal[i] : NaN));

	const rsi13 = rsiSeries(close, 13);
	const stochRaw: number[] = new Array(close.length).fill(NaN);
	for (let i = 0; i < close.length; i++) {
		const minRsi = minWindow(rsi13, i, 21);
		const maxRsi = maxWindow(rsi13, i, 21);
		if (!Number.isFinite(minRsi) || !Number.isFinite(maxRsi)) {
			continue;
		}
		const denom = maxRsi - minRsi;
		stochRaw[i] = denom === 0 ? 50 : (100 * (rsi13[i] - minRsi)) / denom;
	}
	const stochK = new Array(close.length).fill(NaN).map((_, i) => sma(stochRaw, i, 3));
	const stochD = new Array(close.length).fill(NaN).map((_, i) => sma(stochK, i, 5));

	const dmi = taDmi(high, low, close, 5, 13);

	const allRows: OutRow[] = candles.map((candle, i) => {
		const bbMid = sma(close, i, 21);
		const bbStd = stddev(close, i, 21);
		const bbUpper = Number.isFinite(bbMid) && Number.isFinite(bbStd) ? bbMid + 2 * bbStd : NaN;
		const bbLower = Number.isFinite(bbMid) && Number.isFinite(bbStd) ? bbMid - 2 * bbStd : NaN;

		return {
			date: candle.date,
			open: round(candle.open, 4),
			high: round(candle.high, 4),
			low: round(candle.low, 4),
			close: round(candle.close, 4),
			volume: String(Math.round(candle.volume)),
			bb_upper_21_2: round(bbUpper),
			bb_mid_21: round(bbMid),
			bb_lower_21_2: round(bbLower),
			sma_200: round(sma(close, i, 200)),
			sma_89: round(sma(close, i, 89)),
			sma_50: round(sma(close, i, 50)),
			sma_3: round(sma(close, i, 3)),
			sma_2: round(sma(close, i, 2)),
			macd_line_8_21: round(macdLine[i]),
			macd_signal_5: round(macdSignal[i]),
			macd_hist: round(macdHist[i]),
			rsi_13: round(rsi13[i]),
			stoch_rsi_raw_21: round(stochRaw[i]),
			stoch_rsi_k_3: round(stochK[i]),
			stoch_rsi_d_5: round(stochD[i]),
			plus_di_5: round(dmi.plusDi[i]),
			minus_di_5: round(dmi.minusDi[i]),
			adx_13: round(dmi.adx[i])
		};
	});

	const outRows = allRows.filter((row) => row.date >= FROM && row.date <= TO);

	const headers: (keyof OutRow)[] = [
		'date',
		'open',
		'high',
		'low',
		'close',
		'volume',
		'bb_upper_21_2',
		'bb_mid_21',
		'bb_lower_21_2',
		'sma_200',
		'sma_89',
		'sma_50',
		'sma_3',
		'sma_2',
		'macd_line_8_21',
		'macd_signal_5',
		'macd_hist',
		'rsi_13',
		'stoch_rsi_raw_21',
		'stoch_rsi_k_3',
		'stoch_rsi_d_5',
		'plus_di_5',
		'minus_di_5',
		'adx_13'
	];

	writeCsv(OUTPUT_PATH, headers, outRows);
	console.log(`Saved ${outRows.length} rows to ${OUTPUT_PATH}`);
	console.log(`Ticker=${TICKER} range=${FROM}..${TO} (warmup from ${fetchFrom})`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

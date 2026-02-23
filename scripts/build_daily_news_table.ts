import path from 'path';
import { readCsv, writeCsv, CsvRow } from './lib/csv';

type DailyBucket = {
	date: string;
	ticker: string;
	news_count: number;
	avg_sentiment: number;
	min_sentiment: number;
	max_sentiment: number;
	positive_count: number;
	negative_count: number;
	neutral_count: number;
	has_high_impact_news: number;
};

const SOURCE_PATH = process.env.NEWS_FILE_PATH || path.join(__dirname, '../data/news_events_6y.csv');
const OUTPUT_PATH = process.env.NEWS_DAILY_OUT_PATH || path.join(__dirname, '../data/news_daily_6y.csv');

function toNumber(value: string): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : 0;
}

function isHighImpact(headline: string): boolean {
	const lower = headline.toLowerCase();
	const triggers = [
		'earnings',
		'fomc',
		'cpi',
		'nfp',
		'guidance',
		'downgrade',
		'upgrade',
		'lawsuit',
		'acquisition',
		'merger'
	];
	return triggers.some((token) => lower.includes(token));
}

function main(): void {
	const rows = readCsv(SOURCE_PATH);
	if (rows.length === 0) {
		throw new Error(`No rows found in ${SOURCE_PATH}`);
	}

	const buckets = new Map<string, DailyBucket>();

	for (const row of rows) {
		const date = String(row.date || '').slice(0, 10);
		const ticker = String(row.ticker || row.symbol || '').toUpperCase();
		const headline = String(row.headline || '');
		if (!date || !ticker) {
			continue;
		}

		const key = `${date}|${ticker}`;
		if (!buckets.has(key)) {
			buckets.set(key, {
				date,
				ticker,
				news_count: 0,
				avg_sentiment: 0,
				min_sentiment: 999,
				max_sentiment: -999,
				positive_count: 0,
				negative_count: 0,
				neutral_count: 0,
				has_high_impact_news: 0
			});
		}

		const bucket = buckets.get(key)!;
		const sentiment = toNumber(String(row.sentiment_score || '0'));

		bucket.news_count += 1;
		bucket.avg_sentiment += sentiment;
		bucket.min_sentiment = Math.min(bucket.min_sentiment, sentiment);
		bucket.max_sentiment = Math.max(bucket.max_sentiment, sentiment);

		if (sentiment > 0.1) {
			bucket.positive_count += 1;
		} else if (sentiment < -0.1) {
			bucket.negative_count += 1;
		} else {
			bucket.neutral_count += 1;
		}

		if (isHighImpact(headline)) {
			bucket.has_high_impact_news = 1;
		}
	}

	const outRows: CsvRow[] = Array.from(buckets.values())
		.map((bucket) => {
			const avg = bucket.news_count === 0 ? 0 : bucket.avg_sentiment / bucket.news_count;
			return {
				date: bucket.date,
				ticker: bucket.ticker,
				news_count: String(bucket.news_count),
				avg_sentiment: String(Number(avg.toFixed(4))),
				min_sentiment: String(Number((bucket.min_sentiment === 999 ? 0 : bucket.min_sentiment).toFixed(4))),
				max_sentiment: String(Number((bucket.max_sentiment === -999 ? 0 : bucket.max_sentiment).toFixed(4))),
				positive_count: String(bucket.positive_count),
				negative_count: String(bucket.negative_count),
				neutral_count: String(bucket.neutral_count),
				has_high_impact_news: String(bucket.has_high_impact_news)
			};
		})
		.sort((a, b) => (a.date === b.date ? a.ticker.localeCompare(b.ticker) : a.date.localeCompare(b.date)));

	writeCsv(
		OUTPUT_PATH,
		[
			'date',
			'ticker',
			'news_count',
			'avg_sentiment',
			'min_sentiment',
			'max_sentiment',
			'positive_count',
			'negative_count',
			'neutral_count',
			'has_high_impact_news'
		],
		outRows
	);

	console.log(`Saved ${outRows.length} daily rows to ${OUTPUT_PATH}.`);
}

main();

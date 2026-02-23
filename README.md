# Stock Scanner Tool

This project is a stock scanner tool designed to identify stocks that meet specific criteria based on defined rules. It fetches real-time and historical stock data, evaluates stocks against the rules, and notifies users when stocks meet the criteria.

## Features

- Real-time stock data fetching from various sources.
- Historical stock data retrieval for performance analysis.
- Customizable scanning rules to identify stocks based on user-defined criteria.
- Notification system to alert users when stocks meet specific conditions.
- Data persistence for saving and retrieving stock information.

## Project Structure

```
stock-scanner
├── src
│   ├── index.ts               # Entry point of the application
│   ├── scanner
│   │   ├── engine.ts          # Scanning engine logic
│   │   ├── rules.ts           # Stock scanning rules
│   │   └── index.ts           # Exports for the scanner module
│   ├── feeds
│   │   ├── provider.ts        # Real-time data provider
│   │   └── historical.ts      # Historical data provider
│   ├── models
│   │   └── stock.ts           # Stock entity representation
│   ├── services
│   │   ├── notifier.ts        # Notification service
│   │   └── persistence.ts     # Data persistence service
│   ├── config
│   │   └── index.ts           # Configuration settings
│   ├── utils
│   │   └── helpers.ts         # Utility functions
│   └── types
│       └── index.ts           # TypeScript interfaces and types
├── tests
│   ├── scanner.test.ts        # Unit tests for scanner module
│   └── feeds.test.ts          # Unit tests for feeds module
├── scripts
│   └── fetch-data.sh          # Script to fetch stock data
├── .env.example                # Example environment variables
├── .gitignore                  # Files to ignore in version control
├── package.json                # npm configuration file
├── tsconfig.json              # TypeScript configuration file
├── jest.config.js             # Jest configuration for tests
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-scanner
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Set up environment variables by copying `.env.example` to `.env` and filling in the required values.

## Usage

To start the stock scanner, run:
```
npm start
```

You can customize the scanning rules in the `src/scanner/rules.ts` file to fit your specific needs.

## Running Tests

To run the tests, use:
```
npm test
```

## Event Overlay Data Pipeline (6 Years)

This repo includes scripts to build a 6-year event overlay dataset for your learning engine.

### Output Files

- `data/earnings_calendar.csv`
- `data/macro_events.csv`
- `data/news_sentiment.csv`
- `data/event_overlay.csv` (final joined training table)

### Required Input

Export your v2 trades table to CSV and place it at:

- `data/v2_trades.csv`

Minimum required columns in v2 CSV:

- date column: one of `date`, `trade_date`, `day0_date`, `opened_at`
- ticker column: one of `ticker`, `symbol`

### Run End-to-End

```bash
npm run prepare:event-data
```

### Individual Steps

```bash
npm run fetch:earnings
npm run fetch:macro
npm run fetch:news
npm run build:event-overlay
```

### Optional Environment Variables

- `V2_FILE_PATH=/absolute/path/to/v2.csv`
- `TICKERS=AAPL,MSFT,NVDA` (if no v2 file is available for fetch scripts)
- `FINNHUB_API_KEY=...` (improves historical news depth)
- `POLYGON_API_KEY=...` (alternative historical news source)
- `CONTACT_EMAIL=you@example.com` (used for SEC data user-agent)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.# yep1

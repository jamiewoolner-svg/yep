import { SetupScanner, defaultSetupTablePath } from './trading/setupScanner';
import { TradeExecutor } from './trading/executor';

function toNumber(value: string | undefined, fallback: number): number {
    if (value === undefined) {
        return fallback;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

async function main(): Promise<void> {
    const scanner = new SetupScanner();
    const executor = new TradeExecutor({
        initialUnit: toNumber(process.env.INITIAL_CAPITAL, 7000),
        realizedSavingsPct: 0.25,
        legAScalpRoi: toNumber(process.env.LEG_A_ROI, 0.2),
        legBRunnerRoi: toNumber(process.env.LEG_B_ROI, 0.45),
        stopLossRoi: toNumber(process.env.STOP_LOSS_ROI, -0.3),
        maxSignalsPerRun: toNumber(process.env.MAX_SETUPS, 5)
    });

    const setupTablePath = process.env.SETUP_TABLE_PATH || defaultSetupTablePath();
    const allCandidates = scanner.loadCandidatesFromCsv(setupTablePath);
    const topSignals = scanner.findHighQualitySetups(allCandidates, toNumber(process.env.MAX_SETUPS, 5), 35);

    if (topSignals.length === 0) {
        console.log('No high-quality setups found. Check setup_news_table.csv and window settings.');
        return;
    }

    console.log('Top setup signals:');
    for (const signal of topSignals) {
        console.log(
            `${signal.setupDate} ${signal.ticker} ${signal.direction.toUpperCase()} score=${signal.qualityScore} confidence=${signal.confidence} news=${signal.newsCountWindow}`
        );
    }

    const report = executor.runWeekPaperTrading(topSignals, new Date());

    console.log('');
    console.log('Paper trading report (last 7 days):');
    console.log(`Trades: ${report.totalTrades} | Wins: ${report.wins} | Losses: ${report.losses} | WinRate: ${report.winRate}`);
    console.log(`Starting Unit: ${report.startingUnit.toFixed(2)} | Ending Unit: ${report.endingUnit.toFixed(2)}`);
    console.log(`Realized Savings: ${report.realizedSavings.toFixed(2)}`);

    for (const trade of report.trades) {
        console.log(
            `${trade.setupDate} ${trade.ticker} ${trade.result.toUpperCase()} pnl=${trade.totalProfit.toFixed(2)} unit=${trade.endingUnit.toFixed(2)}`
        );
    }
}

main().catch((error) => {
    console.error('Error running scanner/executor:', error);
    process.exit(1);
});
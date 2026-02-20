import { DataProvider } from '../src/feeds/provider';
import { HistoricalDataProvider } from '../src/feeds/historical';
import { Stock } from '../src/models/stock';

describe('DataProvider', () => {
    let dataProvider: DataProvider;

    beforeEach(() => {
        dataProvider = new DataProvider();
    });

    test('fetchData should return an array of stocks', async () => {
        const stocks: Stock[] = await dataProvider.fetchData();
        expect(Array.isArray(stocks)).toBe(true);
        expect(stocks.length).toBeGreaterThan(0);
        stocks.forEach(stock => {
            expect(stock).toHaveProperty('symbol');
            expect(stock).toHaveProperty('price');
            expect(stock).toHaveProperty('volume');
        });
    });
});

describe('HistoricalDataProvider', () => {
    let historicalDataProvider: HistoricalDataProvider;

    beforeEach(() => {
        historicalDataProvider = new HistoricalDataProvider();
    });

    test('fetchHistoricalData should return historical data for a given stock symbol', async () => {
        const stockSymbol = 'AAPL';
        const historicalData = await historicalDataProvider.fetchHistoricalData(stockSymbol);
        expect(Array.isArray(historicalData)).toBe(true);
        expect(historicalData.length).toBeGreaterThan(0);
        historicalData.forEach(data => {
            expect(data).toHaveProperty('date');
            expect(data).toHaveProperty('price');
        });
    });
});
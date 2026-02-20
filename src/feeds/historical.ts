export class HistoricalDataProvider {
    async fetchHistoricalData(stockSymbol: string): Promise<HistoricalData[]> {
        return [
            {
                symbol: stockSymbol,
                date: '2025-01-02',
                price: 150
            }
        ];
    }
}

interface HistoricalData {
    symbol: string;
    date: string;
    price: number;
}

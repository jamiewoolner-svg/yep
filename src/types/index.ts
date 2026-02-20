export interface Stock {
    symbol: string;
    price: number;
    volume: number;
}

export interface HistoricalData {
    date: string;
    open: number;
    close: number;
    high: number;
    low: number;
    volume: number;
}
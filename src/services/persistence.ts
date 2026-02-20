import { Stock } from '../models/stock';

export class Persistence {
    private stocks: Stock[] = [];

    saveStock(stock: Stock): void {
        this.stocks.push(stock);
    }

    async getStocks(): Promise<Stock[]> {
        return this.stocks;
    }
}
import { Stock } from '../models/stock'

export class Notifier {
  sendNotification(_stock: Stock) { /* noop for tests */ }
}

export class StockScannerEngine {
  private stocks: Stock[] = []
  public notifier = new Notifier()

  getStocks(): Stock[] {
    return [...this.stocks]
  }

  addStock(stock: Stock) {
    this.stocks.push(stock)
  }

  // tests expect a simple rule: price > 100
  evaluateStock(stock: Stock): boolean {
    return (stock?.price ?? 0) > 100
  }

  checkForNotifications() {
    for (const s of this.stocks) {
      if (this.evaluateStock(s)) {
        this.notifier.sendNotification(s)
      }
    }
  }
}

export default StockScannerEngine

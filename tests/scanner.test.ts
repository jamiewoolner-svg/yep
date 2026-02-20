import { StockScannerEngine } from '../src/scanner/engine';
import { Stock } from '../src/models/stock';

describe('StockScannerEngine', () => {
    let scanner: StockScannerEngine;

    beforeEach(() => {
        scanner = new StockScannerEngine();
    });

    it('should initialize with no stocks', () => {
        expect(scanner.getStocks()).toHaveLength(0);
    });

    it('should add a stock', () => {
        const stock = new Stock('AAPL', 150, 1000);
        scanner.addStock(stock);
        expect(scanner.getStocks()).toContain(stock);
    });

    it('should evaluate stocks based on rules', () => {
        const stock = new Stock('AAPL', 150, 1000);
        scanner.addStock(stock);
        // Assuming there is a rule that checks if the price is above 100
        const result = scanner.evaluateStock(stock);
        expect(result).toBe(true);
    });

    it('should notify when a stock meets criteria', () => {
        const stock = new Stock('AAPL', 150, 1000);
        scanner.addStock(stock);
        // Mock the notifier service
        const notifySpy = jest.spyOn(scanner.notifier, 'sendNotification');
        scanner.checkForNotifications();
        expect(notifySpy).toHaveBeenCalledWith(stock);
    });
});
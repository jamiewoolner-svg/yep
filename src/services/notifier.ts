import { Stock } from '../models/stock';

export class Notifier {
    sendNotification(stock: Stock): void {
        // Logic to send notification for the stock
        console.log(`Notification: Stock ${stock.symbol} meets the criteria!`);
    }
}
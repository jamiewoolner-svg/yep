export class Stock {
    symbol: string;
    price: number;
    volume: number;

    constructor(symbol: string, price: number, volume: number) {
        this.symbol = symbol;
        this.price = price;
        this.volume = volume;
    }

    updatePrice(newPrice: number): void {
        this.price = newPrice;
    }

    updateVolume(newVolume: number): void {
        this.volume = newVolume;
    }
}
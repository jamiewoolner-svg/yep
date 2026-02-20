export class Rule {
    evaluate(stock: Stock): boolean {
        // Implement rule evaluation logic here
        return false; // Placeholder return value
    }
}

export class PriceAboveRule extends Rule {
    private threshold: number;

    constructor(threshold: number) {
        super();
        this.threshold = threshold;
    }

    evaluate(stock: Stock): boolean {
        return stock.price > this.threshold;
    }
}

export class VolumeAboveRule extends Rule {
    private threshold: number;

    constructor(threshold: number) {
        super();
        this.threshold = threshold;
    }

    evaluate(stock: Stock): boolean {
        return stock.volume > this.threshold;
    }
}

// Add more rules as needed

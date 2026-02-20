export const formatStockData = (stock) => {
    return {
        symbol: stock.symbol.toUpperCase(),
        price: parseFloat(stock.price).toFixed(2),
        volume: stock.volume,
    };
};

export const logMessage = (message) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${message}`);
};
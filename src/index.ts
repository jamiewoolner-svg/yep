import { StockScannerEngine } from './scanner/engine';
import { loadConfig } from './config';
import { DataProvider } from './feeds/provider';

async function main() {
    // Load configuration settings
    const config = loadConfig();

    // Initialize data provider
    const dataProvider = new DataProvider(config.apiKey);

    // Create an instance of the stock scanner engine
    const scannerEngine = new StockScannerEngine(dataProvider);

    // Start the scanning process
    await scannerEngine.startScanning();
}

// Execute the main function
main().catch(error => {
    console.error('Error starting the stock scanner:', error);
});
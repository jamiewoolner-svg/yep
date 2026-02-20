export const config = {
    apiKey: process.env.API_KEY || '',
    apiUrl: process.env.API_URL || 'https://api.example.com',
    scanInterval: parseInt(process.env.SCAN_INTERVAL, 10) || 60000, // default to 60 seconds
};
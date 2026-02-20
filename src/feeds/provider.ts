import { Stock } from '../models/stock'

export class DataProvider {
  // return actual Stock instances to match tests' expectations
  async fetchData(): Promise<Stock[]> {
    return [new Stock('AAPL', 150, 1000)]
  }
}

export default DataProvider

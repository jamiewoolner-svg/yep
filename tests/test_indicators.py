import math
import unittest

from stock_scanner import adx, find_crossings, macd, rsi, sma, stoch_rsi


class IndicatorTests(unittest.TestCase):
    def test_sma(self):
        self.assertEqual(sma([1, 2, 3, 4, 5], 5), 3)
        self.assertTrue(math.isnan(sma([1, 2], 5)))

    def test_rsi_uptrend(self):
        vals = list(range(1, 30))
        self.assertGreaterEqual(rsi(vals, 14), 90)

    def test_rsi_downtrend(self):
        vals = list(range(30, 1, -1))
        self.assertLessEqual(rsi(vals, 14), 10)

    def test_macd_numeric(self):
        vals = [100 + i * 0.5 for i in range(80)]
        macd_line, signal = macd(vals)
        self.assertFalse(math.isnan(macd_line))
        self.assertFalse(math.isnan(signal))
        self.assertGreater(macd_line, 0)

    def test_stoch_rsi_bounds(self):
        vals = [100 + (i % 7) for i in range(120)]
        k, d = stoch_rsi(vals)
        self.assertGreaterEqual(k, 0)
        self.assertLessEqual(k, 100)
        self.assertGreaterEqual(d, 0)
        self.assertLessEqual(d, 100)

    def test_adx_numeric(self):
        highs = [100 + i * 0.6 for i in range(80)]
        lows = [98 + i * 0.6 for i in range(80)]
        closes = [99 + i * 0.6 for i in range(80)]
        adx_val, plus_di, minus_di = adx(highs, lows, closes, 14)
        self.assertFalse(math.isnan(adx_val))
        self.assertFalse(math.isnan(plus_di))
        self.assertFalse(math.isnan(minus_di))
        self.assertGreater(adx_val, 0)

    def test_find_crossings(self):
        lhs = [1, 2, 3, 2, 1, 2]
        rhs = [2, 2, 2, 2, 2, 2]
        up, down = find_crossings(lhs, rhs)
        self.assertEqual(up, [2])
        self.assertEqual(down, [4])


if __name__ == "__main__":
    unittest.main()

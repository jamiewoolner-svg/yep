# POWS Transcript Rule Map

This file captures recurring setup language found in:
- `docs/pows/transcripts/01_POWS Class 1 Introduction.txt`
- `docs/pows/transcripts/02_POWS Class 1 Pg 1 thru 7.txt`
- `docs/pows/transcripts/03_POWS Class 1 Pg 8 thru 17.txt`
- `docs/pows/transcripts/04_POWS Class 1 Pg 18 thru 41.txt`
- `docs/pows/transcripts/05_POWS Class 1 Pg 42 thru 56.txt`
- `docs/pows/transcripts/06_POWS Class 2 Pg 57 thru 74.txt`
- `docs/pows/transcripts/07_POWS Class 2 Pg 75 thru 90.txt`
- `docs/pows/transcripts/08_POWS Class 2 Pg 91 thru 134.txt`
- `docs/pows/transcripts/09_POWS Class 3 Pg 135 thru 146.txt`
- `docs/pows/transcripts/10_POWS Class 3 Pg 147 thru 158.txt`
- `docs/pows/transcripts/11_POWS Class 3 Pg 159 thru 163.txt`
- `docs/pows/transcripts/12_POWS Class 3 Pg 164 thru 184.txt`
- `docs/pows/transcripts/13_POWS Class 4 Pg 185 thru 190.txt`
- `docs/pows/transcripts/14_POWS Class 4 Pg 191 thru 197.txt`
- `docs/pows/transcripts/15_POWS Class 4 Pg 197 thru 200.txt`
- `docs/pows/transcripts/16_POWS Class 4 Pg 201 thru 206.txt`
- `docs/pows/transcripts/17_POWS Class 5 Pg 207 thru 213.txt`
- `docs/pows/transcripts/18_POWS Class 5 Pg 214 thru 222.txt`
- `docs/pows/transcripts/19_POWS Class 5 Pg 223 thru 232.txt`

## Repeated Rules

1. "3x" setup: price + Stoch + MACD crossing together.
2. Use weekly as trend context when decision chart is daily/233.
3. Entry is not the absolute top/bottom; wait for confirmation.
4. "Hit band, spread band, come off band" for liftoff/rejection context.
5. Pre/post earnings can be catalysts, but still requires technical setup.
6. "Proper entry, intelligent exit, according to plan" as primary discipline rule.

## Current Code Mapping

Implemented in scanner logic:
- POWS mode now requires recent price cross (2/3 MA) plus MACD/Stoch alignment.
- POWS mode enforces tight multi-indicator cross timing (`triple_cross_gap <= 2` bars).
- POWS mode keeps Bollinger liftoff/rejection behavior as a setup qualifier.

Already present before this map:
- Dual-timeframe confirmation options (`daily` + `233m`).
- Band touch/lookback and band expansion filters.
- MACD/Stoch cross lookback filters.

## Next Iteration Candidates

1. Add explicit weekly-context filter for daily/233 decisions.
2. Add optional earnings-window include/exclude filter.
3. Add separate setup profiles (trend continuation vs reversal) from transcript examples.

# Implied Volatility Surface Builder

A Spyder-friendly Python project that downloads listed option chains, cleans
the quotes, computes Black-Scholes implied volatility, and visualizes the
resulting volatility surface.

## What it demonstrates

- Option-chain collection across multiple target maturities
- Liquidity, spread, and no-arbitrage quality filters
- Forward moneyness and OTM option selection
- Numerical inversion of Black-Scholes with Brent's method
- Smile interpolation on a common log-moneyness grid
- 3D surface, skew, and ATM term-structure charts

## Run in Spyder

1. Open `implied_volatility_surface.py`.
2. Install the packages in `requirements.txt` in the same environment used by
   Spyder.
3. Run the file. The default ticker is `SPY`; charts remain in Spyder's Plots
   pane.

Optional command-line parameters include `--ticker`, `--rate`,
`--target-days`, and `--moneyness-band`.

## Methodological notes

The implementation uses a constant user-supplied risk-free rate and does not
model dividends explicitly. It is an educational research project, not a
production option-pricing system or investment advice.

Data source: Yahoo Finance through `yfinance`.

# Discounted Cash Flow (DCF) Valuation Model

A 5-year DCF model to estimate the intrinsic value of publicly traded companies. Built in Python with automated financial data retrieval, WACC calculation via CAPM, sensitivity analysis, and visualization. Pls try it ;)

## Features

|Feature|Description|
|-|-|
|**5-Year FCF Projections**|Revenue, EBIT, EBIAT, D\&A, CapEx, and NWC projected using weighted historical margins and analyst consensus estimates|
|**WACC via CAPM**|Automated calculation of Cost of Equity (CAPM), Cost of Debt, and capital structure weights|
|**Terminal Value**|Perpetual Growth (Gordon Growth) and EV/EBITDA Exit Multiple methods, with sector-aware default multiples|
|**Monte Carlo Simulation**|10,000-run probabilistic valuation randomizing growth, margins, WACC, and TGR — outputs a price distribution with percentiles|
|**Scenario Analysis**|Bull / Base / Bear cases with different growth and margin assumptions|
|**Sensitivity Analysis**|Matrix of implied prices across WACC and Terminal Growth Rate scenarios|
|**Comparable Company Analysis**|Auto-fetches P/E, EV/EBITDA, EV/Revenue, and P/S for peer companies|
|**Implied Growth Rate**|Bisection algorithm to reverse-engineer the growth rate priced into the current stock price|
|**Multi-Stock Comparison**|Run the model across multiple tickers and compare valuations side by side|
|**Visualization**|6-chart report: FCF bars, sensitivity heatmap, valuation bridge, Monte Carlo histogram, scenario analysis, and key metrics|
|**JSON Export**|Export model outputs for integration with other tools|

## Quick Start

```bash
# Clone the repo
git clone https://github.com/EmanueleSturzo/DCF-Valuation-Model.git
cd DCF-Valuation-Model

# Install dependencies
pip install -r requirements.txt

# Run the model
python dcf\_model.py --ticker AAPL
```

## Usage

### Single Stock Valuation

```bash
python dcf\_model.py --ticker AAPL
```

**Output:**

```
============================================================
  DCF VALUATION SUMMARY — Apple Inc. (AAPL)
============================================================
  Sector:            Technology
  Industry:          Consumer Electronics
  Current Price:     $217.90
  Shares Outstanding: 15.115B
============================================================

  KEY ASSUMPTIONS
  ────────────────────────────────────────
  Revenue Growth (Yr1): 7.32%
  EBIT Margin:          31.52%
  Effective Tax Rate:   16.02%
  ...

  VALUATION OUTPUT
  ────────────────────────────────────────
  Implied Share Price:  $243.18
  Current Market Price: $217.90
  Margin of Safety:     11.60% (UNDERVALUED ↑)
  Implied Growth Rate:  5.14%
```

### Custom Parameters

```bash
python dcf\_model.py --ticker NVDA --tgr 0.04 --riskfree ^FVX --market VT
```

|Argument|Description|Default|
|-|-|-|
|`--ticker`|Stock ticker symbol (required)|—|
|`--tgr`|Terminal Growth Rate|`0.03`|
|`--riskfree`|Risk-free rate proxy ticker|`^TNX` (US 10Y)|
|`--market`|Market return proxy ticker|`VTI`|
|`--json`|Export results to JSON file|`None`|

### Compare Multiple Stocks

```bash
python compare.py --tickers AAPL MSFT GOOGL META AMZN
```

Outputs a side-by-side comparison table with implied prices, margins of safety, and buy/hold/sell signals.

### Generate Visual Report

```bash
python visualize.py --ticker AAPL --save
```

Produces a full-page report with 4 charts saved to `output/AAPL\_dcf\_report.png`.

### Use as a Python Module

```python
from dcf\_model import DCFModel

model = DCFModel("MSFT")

# Core valuation
print(f"Implied Price: ${model.implied\_share\_price:.2f}")
print(f"WACC: {model.wacc:.2%}")
print(f"Margin of Safety: {model.margin\_of\_safety:.2%}")

# Monte Carlo simulation
mc = model.monte\_carlo()
print(f"Monte Carlo Median: ${mc\['median']:.2f}")
print(f"P10–P90 Range: ${mc\['p10']:.0f}–${mc\['p90']:.0f}")
print(f"Probability Undervalued: {mc\['prob\_undervalued']:.1f}%")

# Scenario analysis
print(model.scenario\_analysis())

# Exit multiple valuation
em = model.exit\_multiple\_valuation()
print(f"Exit Multiple Price: ${em\['implied\_price']:.2f} ({em\['multiple']:.0f}x EV/EBITDA)")

# Sensitivity table
print(model.sensitivity\_analysis())

# Comparable company analysis (requires internet)
print(model.comparable\_analysis(\["GOOGL", "META", "AMZN"]))

# Export
model.to\_json("output/msft\_valuation.json")
```

### Comparable Company Analysis

```bash
# Auto-detect peers by sector
python dcf\_model.py --ticker AAPL --comps

# Specify custom peers
python dcf\_model.py --ticker AAPL --comps MSFT GOOGL AMZN META
```

\---

## Methodology

### 1\. Free Cash Flow (FCF)

Free Cash Flow represents the cash available to all capital providers after operating expenses and reinvestment:

$$\\text{FCF} = \\text{EBIT} \\times (1 - \\text{Tax Rate}) + \\text{D\&A} - \\text{CapEx} - \\Delta\\text{NWC}$$

Where:

* **EBIT** = Earnings Before Interest and Taxes
* **D\&A** = Depreciation \& Amortization (non-cash add-back)
* **CapEx** = Capital Expenditures (reinvestment in fixed assets)
* **ΔNWC** = Change in Net Working Capital

Each component is projected as a **weighted-average margin of revenue**, with more recent years weighted more heavily (40/30/20/10).

### 2\. Weighted Average Cost of Capital (WACC)

WACC is the blended cost of financing used as the discount rate:

$$\\text{WACC} = W\_D \\cdot R\_D \\cdot (1 - T) + W\_E \\cdot R\_E$$

**Cost of Equity** is derived from the Capital Asset Pricing Model (CAPM):

$$R\_E = R\_f + \\beta \\cdot (R\_m - R\_f)$$

|Variable|Meaning|Source|
|-|-|-|
|$R\_f$|Risk-Free Rate|US 10-Year Treasury Yield (`^TNX`)|
|$\\beta$|Systematic risk relative to market|Yahoo Finance|
|$R\_m$|Expected Market Return|VTI 3-year average return|
|$R\_D$|Cost of Debt = Interest Expense / Total Debt|Income Statement|
|$W\_D, W\_E$|Debt and Equity weights in capital structure|Balance Sheet + Market Cap|

### 3\. Terminal Value — Perpetual Growth Method

Assumes the company generates free cash flow at a constant growth rate in perpetuity after the projection period:

$$\\text{TV} = \\frac{\\text{FCF}\_n \\times (1 + g)}{(\\text{WACC} - g)}$$

Where $g$ is the terminal growth rate (default: 3%, approximating long-run nominal GDP growth).

### 4\. Discounted Cash Flow (Enterprise Value)

$$\\text{EV} = \\sum\_{t=1}^{n} \\frac{\\text{FCF}\_t}{(1 + \\text{WACC})^{t - 0.5}} + \\frac{\\text{TV}}{(1 + \\text{WACC})^n}$$

Uses **mid-year convention** ($t - 0.5$) to reflect that cash flows are received throughout the year, not only at year-end.

### 5\. Equity Value → Implied Share Price

$$\\text{Equity Value} = \\text{EV} - \\text{Debt} + \\text{Cash}$$

$$\\text{Implied Price} = \\frac{\\text{Equity Value}}{\\text{Shares Outstanding}}$$

### 6\. Implied Growth Rate (Bisection Method)

Reverse-engineers the revenue growth rate the market is currently pricing in by finding $g^\*$ such that:

$$\\text{ImpliedPrice}(g^\*) = \\text{CurrentPrice}$$

Solved iteratively using the bisection algorithm with convergence tolerance of $0.01.

### 7\. Sensitivity Analysis

Generates a matrix of implied share prices across a range of WACC and Terminal Growth Rate assumptions, enabling the analyst to assess how sensitive the valuation is to key inputs.

### 8\. Monte Carlo Simulation

Randomizes key model inputs across distributions to produce a probability distribution of implied prices:

|Input|Distribution|
|-|-|
|Revenue Growth|Normal (μ = base estimate, σ = 3%)|
|EBIT Margin|Normal (μ = historical avg, σ = 2%)|
|WACC|Normal (μ = calculated, σ = 1.5%)|
|Terminal Growth Rate|Uniform (1.5% – 4.0%)|

10,000 simulations produce percentiles (P10, P25, P50, P75, P90) and the probability that the stock is undervalued.

### 9\. Scenario Analysis

Three cases with distinct assumptions:

|Scenario|Growth Adj|Margin Adj|
|-|-|-|
|**Bull**|Base + 2%|Base + 1%|
|**Base**|Model default|Model default|
|**Bear**|Base − 3%|Base − 2%|

### 10\. Exit Multiple Terminal Value

An alternative to the Gordon Growth method:

$$\\text{TV}\_{exit} = \\text{EBITDA}\_n \\times \\text{EV/EBITDA Multiple}$$

The model auto-selects a sector-appropriate multiple (e.g., 20x for Technology, 6x for Energy) or accepts a custom input.

### 11\. Comparable Company Analysis

Automatically retrieves valuation multiples (P/E, EV/EBITDA, EV/Revenue, P/S) for peer companies and displays a comparison table. Peers are either specified manually or auto-selected by sector.

\---

## Project Structure

```
DCF-Valuation-Model/
├── dcf\_model.py          # Core DCF model (CLI + importable class)
├── compare.py            # Multi-stock comparison tool
├── visualize.py          # Chart generation and visual reports
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
├── .gitignore
├── output/               # Generated reports (gitignored)
└── README.md
```

## Data Sources

|Data|Source|
|-|-|
|Financial Statements|[Yahoo Finance](https://finance.yahoo.com) via `yfinance`|
|Analyst Revenue Estimates|Yahoo Finance consensus|
|Risk-Free Rate|US Treasury Yields (`^TNX`, `^FVX`)|
|Market Return|Vanguard Total Stock Market ETF (`VTI`)|
|Beta|Yahoo Finance|

## Limitations

* **Analyst estimates**: Revenue projections rely on Yahoo Finance consensus; these may be stale or unavailable for smaller companies. The model falls back to historical CAGR when estimates are not available.
* **Margin stability**: The model assumes historical margins (EBIT, D\&A, CapEx) are representative of the future. This may not hold for companies undergoing transformation.
* **Terminal Value dominance**: In most DCFs, the Terminal Value accounts for 60–80% of the Enterprise Value. Small changes in WACC or TGR can dramatically change the output — hence the sensitivity table.
* **Single-scenario**: This is a deterministic model. A Monte Carlo extension would provide a probability distribution of outcomes.

## Potential Extensions

* \[ ] Revenue build-up by business segment (e.g., iPhone, Services, Mac)
* \[ ] Regression-based beta calculation from historical returns
* \[ ] ROIC and reinvestment rate consistency check
* \[ ] Streamlit web dashboard for interactive analysis
* \[ ] Historical DCF accuracy backtesting
* \[ ] LBO model add-on

## References

* Damodaran, A. — [NYU Stern Valuation Resources](https://pages.stern.nyu.edu/~adamodar/)
* [Investopedia — DCF Analysis](https://www.investopedia.com/terms/d/dcf.asp)
* [Wall Street Oasis — Terminal Value](https://www.wallstreetoasis.com/resources/skills/valuation/terminal-value)
* Financial data: [Yahoo Finance](https://finance.yahoo.com)

## License

[MIT](LICENSE)


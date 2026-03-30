"""
Demo: Run the DCF model with real Apple (AAPL) financial data.
This uses hardcoded data so it works without internet access.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  REAL APPLE FINANCIAL DATA (FY2022–FY2025, in millions USD)
# ═══════════════════════════════════════════════════════════════

# Revenue: FY2025, FY2024, FY2023, FY2022
revenue = pd.Series([416_161, 391_035, 383_285, 394_328]) * 1e6

# EBIT (Operating Income)
ebit = pd.Series([133_050, 123_216, 114_301, 119_437]) * 1e6

# Tax Provision
tax = pd.Series([20_719, 29_749, 16_741, 19_300]) * 1e6

# Depreciation & Amortization (from cash flow statement)
da = pd.Series([11_445, 8_534, 8_866, 11_104]) * 1e6

# Capital Expenditures
capex = pd.Series([12_715, 9_447, 10_959, 10_708]) * 1e6

# Change in Working Capital
nwc_change = pd.Series([-3_651, -5_259, -6_577, 1_200]) * 1e6

# ═══════════════════════════════════════════════════════════════
#  MARKET & COMPANY DATA (as of March 2026)
# ═══════════════════════════════════════════════════════════════

current_price = 217.90
shares_outstanding = 14.95e9
market_cap = current_price * shares_outstanding  # ~$3.26T
total_debt = 99.887e9
total_cash = 29.943e9
beta = 1.24
interest_expense = 0  # Apple has net interest income
risk_free_rate = 0.0425   # US 10Y Treasury
market_return = 0.1040     # VTI 3-year avg return

# Analyst consensus revenue estimates
rev_estimate_current_year = 445.0e9   # FY2026E
rev_growth_next_year = 0.072          # +7.2% FY2027E

# ═══════════════════════════════════════════════════════════════
#  MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════

terminal_growth_rate = 0.03
projection_years = 5
weights = np.array([0.4, 0.3, 0.2, 0.1])
tax_weights = np.array([0.5, 0.3, 0.2])

# ═══════════════════════════════════════════════════════════════
#  STEP 1: PROJECT REVENUE
# ═══════════════════════════════════════════════════════════════

rev_growth = 1 + rev_growth_next_year
projected_revenue = pd.Series([rev_estimate_current_year * (rev_growth ** i) for i in range(projection_years)])

print("\n" + "=" * 65)
print("  DCF VALUATION — Apple Inc. (AAPL)")
print("=" * 65)
print(f"  Sector:             Technology")
print(f"  Industry:           Consumer Electronics")
print(f"  Current Price:      ${current_price:.2f}")
print(f"  Shares Outstanding: {shares_outstanding/1e9:.3f}B")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
#  STEP 2: CALCULATE MARGINS (weighted historical average)
# ═══════════════════════════════════════════════════════════════

ebit_margin = np.average(ebit / revenue, weights=weights)
tax_rate = np.clip(np.average((tax / ebit)[:3], weights=tax_weights), 0, 0.50)
da_margin = np.average(da / revenue, weights=weights)
capex_margin = np.average(capex / revenue, weights=weights)
nwc_margin = np.average(nwc_change / revenue, weights=weights)

print(f"\n  KEY ASSUMPTIONS")
print(f"  {'─'*45}")
print(f"  Revenue Growth (Yr1):   {rev_growth_next_year:.2%}")
print(f"  EBIT Margin:            {ebit_margin:.2%}")
print(f"  Effective Tax Rate:     {tax_rate:.2%}")
print(f"  D&A / Revenue:          {da_margin:.2%}")
print(f"  CapEx / Revenue:        {capex_margin:.2%}")
print(f"  ΔNWC / Revenue:         {nwc_margin:.2%}")

# ═══════════════════════════════════════════════════════════════
#  STEP 3: PROJECT CASH FLOWS
# ═══════════════════════════════════════════════════════════════

projected_ebit = projected_revenue * ebit_margin
projected_tax = projected_ebit * tax_rate
projected_ebiat = projected_ebit - projected_tax
projected_da = projected_revenue * da_margin
projected_capex = projected_revenue * capex_margin
projected_nwc = projected_revenue * nwc_margin

# FCF = EBIAT + D&A - CapEx - ΔNWC
projected_fcf = projected_ebiat + projected_da - projected_capex - projected_nwc

# ═══════════════════════════════════════════════════════════════
#  STEP 4: CALCULATE WACC
# ═══════════════════════════════════════════════════════════════

total_capital = total_debt + market_cap
weight_debt = total_debt / total_capital
weight_equity = market_cap / total_capital
cost_of_debt = max(interest_expense / max(total_debt, 1), 0.035)  # Floor at 3.5% for Apple
cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
wacc = weight_debt * cost_of_debt * (1 - tax_rate) + weight_equity * cost_of_equity

print(f"\n  WACC COMPONENTS")
print(f"  {'─'*45}")
print(f"  Risk-Free Rate (Rf):    {risk_free_rate:.2%}")
print(f"  Market Return (Rm):     {market_return:.2%}")
print(f"  Equity Risk Premium:    {market_return - risk_free_rate:.2%}")
print(f"  Beta (β):               {beta:.2f}")
print(f"  Cost of Equity (Re):    {cost_of_equity:.2%}")
print(f"  Cost of Debt (Rd):      {cost_of_debt:.2%}")
print(f"  Weight Debt:            {weight_debt:.2%}")
print(f"  Weight Equity:          {weight_equity:.2%}")
print(f"  ══► WACC:               {wacc:.2%}")

# ═══════════════════════════════════════════════════════════════
#  STEP 5: TERMINAL VALUE (Gordon Growth Model)
# ═══════════════════════════════════════════════════════════════

terminal_value = projected_fcf.iloc[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

print(f"  Terminal Growth Rate:   {terminal_growth_rate:.2%}")

# ═══════════════════════════════════════════════════════════════
#  STEP 6: DISCOUNT CASH FLOWS → ENTERPRISE VALUE
# ═══════════════════════════════════════════════════════════════

discounted_fcf = sum(
    fcf / ((1 + wacc) ** (0.5 + i))
    for i, fcf in enumerate(projected_fcf)
)
discounted_tv = terminal_value / ((1 + wacc) ** projection_years)
enterprise_value = discounted_fcf + discounted_tv

# ═══════════════════════════════════════════════════════════════
#  STEP 7: EQUITY VALUE → IMPLIED SHARE PRICE
# ═══════════════════════════════════════════════════════════════

equity_value = enterprise_value - total_debt + total_cash
implied_price = equity_value / shares_outstanding
margin_of_safety = (implied_price - current_price) / current_price

# ═══════════════════════════════════════════════════════════════
#  PROJECTED FREE CASH FLOWS TABLE
# ═══════════════════════════════════════════════════════════════

print(f"\n  PROJECTED FREE CASH FLOWS")
print(f"  {'─'*45}")
years = [f"Year {i+1}" for i in range(projection_years)]
table_data = {
    "Revenue":          projected_revenue.values,
    "EBIT":             projected_ebit.values,
    "EBIAT":            projected_ebiat.values,
    "D&A":              projected_da.values,
    "CapEx":            projected_capex.values,
    "ΔNWC":             projected_nwc.values,
    "Free Cash Flow":   projected_fcf.values,
}
df = pd.DataFrame(table_data, index=years).T
print(df.map(lambda x: f"${x/1e9:.2f}B").to_string())

# ═══════════════════════════════════════════════════════════════
#  VALUATION OUTPUT
# ═══════════════════════════════════════════════════════════════

direction = "UNDERVALUED ↑" if margin_of_safety > 0 else "OVERVALUED ↓"

print(f"\n  VALUATION OUTPUT")
print(f"  {'─'*45}")
print(f"  PV of FCFs (Yr 1–5):   ${discounted_fcf/1e9:.2f}B")
print(f"  Terminal Value:         ${terminal_value/1e9:.2f}B")
print(f"  PV of Terminal Value:   ${discounted_tv/1e9:.2f}B")
print(f"  TV as % of EV:          {discounted_tv/enterprise_value:.1%}")
print(f"  Enterprise Value:       ${enterprise_value/1e9:.2f}B")
print(f"  (−) Total Debt:         ${total_debt/1e9:.2f}B")
print(f"  (+) Total Cash:         ${total_cash/1e9:.2f}B")
print(f"  Equity Value:           ${equity_value/1e9:.2f}B")
print(f"  ═══════════════════════════════════════════")
print(f"  ══► Implied Share Price: ${implied_price:.2f}")
print(f"  ══► Current Price:       ${current_price:.2f}")
print(f"  ══► Margin of Safety:    {margin_of_safety:.2%} ({direction})")

# ═══════════════════════════════════════════════════════════════
#  IMPLIED GROWTH RATE (Bisection Method)
# ═══════════════════════════════════════════════════════════════

base_rev = revenue.iloc[0]
lo, hi = -0.50, 1.00
for _ in range(100):
    mid = (lo + hi) / 2
    rev = pd.Series([base_rev * ((1 + mid) ** i) for i in range(projection_years)])
    e = rev * ebit_margin
    t = e * tax_rate
    eb = e - t
    d = rev * da_margin
    cx = rev * capex_margin
    nw = rev * nwc_margin
    fcf = eb + d - cx - nw
    disc = sum(f / ((1 + wacc) ** (0.5 + i)) for i, f in enumerate(fcf))
    tv = fcf.iloc[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    dtv = tv / ((1 + wacc) ** projection_years)
    ev = disc + dtv
    eq = ev - total_debt + total_cash
    p = eq / shares_outstanding
    if abs(p - current_price) < 0.01:
        break
    if p > current_price:
        hi = mid
    else:
        lo = mid

print(f"  ══► Implied Growth Rate: {mid:.2%}")
print(f"      (the growth rate the market is pricing in)")

# ═══════════════════════════════════════════════════════════════
#  SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n  SENSITIVITY ANALYSIS — Implied Share Price")
print(f"  Rows = WACC | Columns = Terminal Growth Rate")
print(f"  {'─'*45}")

wacc_range = np.arange(max(wacc - 0.02, 0.04), wacc + 0.025, 0.005)
tgr_range = np.arange(0.015, 0.045, 0.005)

sens = {}
for tgr in tgr_range:
    col = {}
    for w in wacc_range:
        if w <= tgr:
            col[f"{w:.1%}"] = "N/A"
            continue
        d = sum(f / ((1 + w) ** (0.5 + i)) for i, f in enumerate(projected_fcf))
        tv_s = projected_fcf.iloc[-1] * (1 + tgr) / (w - tgr)
        dtv_s = tv_s / ((1 + w) ** projection_years)
        ev_s = d + dtv_s
        eq_s = ev_s - total_debt + total_cash
        p_s = eq_s / shares_outstanding
        col[f"{w:.1%}"] = f"${p_s:.0f}"
    sens[f"{tgr:.1%}"] = col

sens_df = pd.DataFrame(sens)
sens_df.index.name = "WACC \\ TGR"
print(sens_df.to_string())

print(f"\n  Current price ${current_price:.2f} shown for reference.")

# ═══════════════════════════════════════════════════════════════
#  SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n  SCENARIO ANALYSIS")
print(f"  {'─'*45}")

scenarios = {
    "Bull": {"growth_adj": 0.02, "margin_adj": 0.01},
    "Base": {"growth_adj": 0.00, "margin_adj": 0.00},
    "Bear": {"growth_adj": -0.03, "margin_adj": -0.02},
}
base_rev = revenue.iloc[0]

for name, adj in scenarios.items():
    g = rev_growth_next_year + adj["growth_adj"]
    m = ebit_margin + adj["margin_adj"]
    rev_s = pd.Series([base_rev * ((1 + g) ** i) for i in range(projection_years)])
    ebit_s = rev_s * m
    ebiat_s = ebit_s * (1 - tax_rate)
    da_s = rev_s * da_margin
    capex_s = rev_s * capex_margin
    nwc_s = rev_s * nwc_margin
    fcf_s = ebiat_s + da_s - capex_s - nwc_s
    disc_s = sum(f / ((1 + wacc) ** (0.5 + i)) for i, f in enumerate(fcf_s))
    tv_s = fcf_s.iloc[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    dtv_s = tv_s / ((1 + wacc) ** projection_years)
    ev_s = disc_s + dtv_s
    eq_s = ev_s - total_debt + total_cash
    p_s = eq_s / shares_outstanding
    vs = (p_s - current_price) / current_price
    print(f"  {name:<6} Growth: {g:.1%}  Margin: {m:.1%}  →  ${p_s:.2f}  ({vs:+.1%})")

# ═══════════════════════════════════════════════════════════════
#  EXIT MULTIPLE VALUATION
# ═══════════════════════════════════════════════════════════════

print(f"\n  EXIT MULTIPLE VALUATION")
print(f"  {'─'*45}")
ev_ebitda_multiple = 20.0  # Technology sector
final_ebitda = projected_ebit.iloc[-1] + projected_da.iloc[-1]
tv_exit = final_ebitda * ev_ebitda_multiple
dtv_exit = tv_exit / ((1 + wacc) ** projection_years)
ev_exit = discounted_fcf + dtv_exit
eq_exit = ev_exit - total_debt + total_cash
price_exit = eq_exit / shares_outstanding
print(f"  EV/EBITDA Multiple:  {ev_ebitda_multiple:.0f}x (Technology sector)")
print(f"  Terminal Value:      ${tv_exit/1e9:.2f}B")
print(f"  Implied Price:       ${price_exit:.2f}")

# ═══════════════════════════════════════════════════════════════
#  MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════

print(f"\n  MONTE CARLO SIMULATION (10,000 runs)")
print(f"  {'─'*45}")

rng = np.random.default_rng(42)
n_sims = 10000
growth_samples = rng.normal(rev_growth_next_year, 0.03, n_sims)
margin_samples = rng.normal(ebit_margin, 0.02, n_sims)
wacc_samples = rng.normal(wacc, 0.015, n_sims)
tgr_samples = rng.uniform(0.015, 0.04, n_sims)

mc_prices = np.zeros(n_sims)
for s in range(n_sims):
    g = growth_samples[s]
    m = np.clip(margin_samples[s], 0.05, 0.60)
    w = np.clip(wacc_samples[s], 0.04, 0.25)
    tgr = tgr_samples[s]
    if w <= tgr:
        tgr = w - 0.01

    rev_mc = np.array([base_rev * ((1 + g) ** i) for i in range(projection_years)])
    ebit_mc = rev_mc * m
    ebiat_mc = ebit_mc * (1 - tax_rate)
    da_mc = rev_mc * da_margin
    cx_mc = rev_mc * capex_margin
    nwc_mc = rev_mc * nwc_margin
    fcf_mc = ebiat_mc + da_mc - cx_mc - nwc_mc

    disc_mc = sum(fcf_mc[i] / ((1 + w) ** (0.5 + i)) for i in range(projection_years))
    tv_mc = fcf_mc[-1] * (1 + tgr) / (w - tgr)
    dtv_mc = tv_mc / ((1 + w) ** projection_years)
    ev_mc = disc_mc + dtv_mc
    eq_mc = ev_mc - total_debt + total_cash
    mc_prices[s] = eq_mc / shares_outstanding

print(f"  Mean Implied Price:  ${np.mean(mc_prices):.2f}")
print(f"  Median:              ${np.median(mc_prices):.2f}")
print(f"  10th Percentile:     ${np.percentile(mc_prices, 10):.2f}")
print(f"  25th Percentile:     ${np.percentile(mc_prices, 25):.2f}")
print(f"  75th Percentile:     ${np.percentile(mc_prices, 75):.2f}")
print(f"  90th Percentile:     ${np.percentile(mc_prices, 90):.2f}")
prob = np.mean(mc_prices > current_price) * 100
print(f"  Prob. Undervalued:   {prob:.1f}%")

print(f"\n{'='*65}")
print(f"{'='*65}\n")

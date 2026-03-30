"""
Discounted Cash Flow (DCF) Valuation Model
============================================
A 5-year DCF model to estimate the intrinsic value of publicly traded companies.

Features:
    - 5-year revenue and cash flow projections
    - WACC calculation via CAPM
    - Terminal Value (Perpetual Growth + Exit Multiple methods)
    - Sensitivity analysis on WACC and Terminal Growth Rate
    - Monte Carlo simulation (10,000 runs) for probabilistic valuation
    - Bull / Base / Bear scenario analysis
    - Comparable company analysis (peer multiples)
    - Implied growth rate via bisection method
    - Automatic financial data retrieval via yfinance

Author: Emanuele Sturzo
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import json
import os
from argparse import ArgumentParser


class DCFModel:
    """
    5-Year Discounted Cash Flow Model

    Calculates the intrinsic value of a publicly traded company
    using projected Free Cash Flows discounted at the WACC.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    terminal_growth_rate : float
        Long-term perpetual growth rate (default: 0.03)
    risk_free_ticker : str
        Ticker for the risk-free rate proxy (default: '^TNX' = US 10Y Treasury)
    market_ticker : str
        Ticker for market return proxy (default: 'VTI' = Vanguard Total Stock Market)
    projection_years : int
        Number of years to project (default: 5)
    """

    def __init__(self, ticker, terminal_growth_rate=0.03, risk_free_ticker="^TNX",
                 market_ticker="VTI", projection_years=5):

        self.ticker = ticker.upper()
        self.projection_years = projection_years
        self.terminal_growth_rate = terminal_growth_rate

        # Fetch stock data
        self.stock = yf.Ticker(self.ticker)
        self.income = self.stock.income_stmt
        self.balance = self.stock.balance_sheet
        self.cashflow = self.stock.cash_flow
        self.info = self.stock.info

        # Weights for historical averaging (more recent = higher weight)
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.tax_weights = np.array([0.5, 0.3, 0.2])

        # Core company data
        self.market_cap = self.info.get("marketCap", 0)
        self.total_debt = self.info.get("totalDebt", 0)
        self.total_cash = self.info.get("totalCash", 0)
        self.shares_outstanding = self.info.get("sharesOutstanding", 0)
        self.beta = self.info.get("beta", 1.0)
        self.current_price = self.info.get("previousClose", 0)
        self.company_name = self.info.get("shortName", self.ticker)
        self.sector = self.info.get("sector", "N/A")
        self.industry = self.info.get("industry", "N/A")

        # Interest expense
        self.interest_expense = abs(self._safe_get(self.income, "Interest Expense", 0))

        # Market data
        self.risk_free_rate = yf.Ticker(risk_free_ticker).info.get("previousClose", 4.0) / 100
        market_info = yf.Ticker(market_ticker).info
        self.market_return = market_info.get("threeYearAverageReturn",
                             market_info.get("fiveYearAverageReturn", 0.10))

        # Historical financials
        self.hist_revenue = self._get_line_item(self.income, "Total Revenue")
        self.hist_ebit = self._get_line_item(self.income, "EBIT")
        self.hist_tax = self._get_line_item(self.income, "Tax Provision")
        self.hist_da = self._get_line_item(self.cashflow, "Depreciation And Amortization", absolute=True)
        self.hist_capex = self._get_line_item(self.cashflow, "Capital Expenditure", absolute=True)
        self.hist_nwc_change = self._get_line_item(self.cashflow, "Change In Working Capital")

        # Run the model
        self._build_model()

    # ─── Data Helpers ────────────────────────────────────────────────

    @staticmethod
    def _safe_get(df, label, default=0):
        """Safely retrieve a value from a financial statement."""
        try:
            val = df.loc[label].iloc[0]
            return val if pd.notna(val) else default
        except (KeyError, IndexError):
            return default

    def _get_line_item(self, df, label, absolute=False):
        """Extract a line item series, trimming the oldest year for consistency."""
        try:
            series = df.loc[label][:-1]
            return abs(series) if absolute else series
        except KeyError:
            return pd.Series([0] * 4)

    # ─── Projection Engine ───────────────────────────────────────────

    def _build_model(self):
        """Sequentially build all model components."""
        self.projected_revenue = self._project_revenue()
        self.ebit_margin = self._calc_weighted_margin(self.hist_ebit, self.hist_revenue)
        self.projected_ebit = self.projected_revenue * self.ebit_margin

        self.tax_rate = self._calc_tax_rate()
        self.projected_tax = self.projected_ebit * self.tax_rate
        self.projected_ebiat = self.projected_ebit - self.projected_tax

        self.da_margin = self._calc_weighted_margin(self.hist_da, self.hist_revenue)
        self.projected_da = self.projected_revenue * self.da_margin

        self.capex_margin = self._calc_weighted_margin(self.hist_capex, self.hist_revenue)
        self.projected_capex = self.projected_revenue * self.capex_margin

        self.nwc_margin = self._calc_weighted_margin(self.hist_nwc_change, self.hist_revenue)
        self.projected_nwc = self.projected_revenue * self.nwc_margin

        self.projected_fcf = self._calc_fcf()
        self.wacc = self._calc_wacc()
        self.terminal_value = self._calc_terminal_value_perpetual()
        self.enterprise_value = self._calc_dcf()
        self.implied_share_price = self._calc_implied_price()
        self.margin_of_safety = (self.implied_share_price - self.current_price) / self.current_price

    def _project_revenue(self):
        """Project future revenue using analyst consensus estimates."""
        try:
            estimates = self.stock.revenue_estimate
            rev_current = estimates.loc["0y", "avg"]
            rev_growth = 1 + estimates.loc["+1y", "growth"]
        except Exception:
            # Fallback: use historical CAGR
            rev_current = self.hist_revenue.iloc[0]
            if len(self.hist_revenue) >= 3:
                rev_growth = 1 + (self.hist_revenue.iloc[0] / self.hist_revenue.iloc[2]) ** (1/2) - 1
            else:
                rev_growth = 1.05

        self.revenue_growth_rate = rev_growth - 1
        return pd.Series([rev_current * (rev_growth ** i) for i in range(self.projection_years)])

    def _calc_weighted_margin(self, numerator, denominator):
        """Calculate a weighted-average margin from historical data."""
        try:
            margins = numerator / denominator
            valid = margins.dropna()
            n = min(len(valid), len(self.weights))
            if n == 0:
                return 0.0
            return float(np.average(valid.iloc[:n], weights=self.weights[:n]))
        except Exception:
            return 0.0

    def _calc_tax_rate(self):
        """Calculate effective tax rate with safety bounds."""
        try:
            rates = (self.hist_tax / self.hist_ebit)[:3]
            valid = rates.dropna()
            n = min(len(valid), len(self.tax_weights))
            if n == 0:
                return 0.21  # US statutory rate
            rate = float(np.average(valid.iloc[:n], weights=self.tax_weights[:n]))
            return np.clip(rate, 0.0, 0.50)  # Bound between 0% and 50%
        except Exception:
            return 0.21

    def _calc_fcf(self):
        """
        Free Cash Flow = EBIAT + D&A - CapEx - Change in NWC

        FCF represents cash available to all capital providers (debt + equity).
        """
        return self.projected_ebiat + self.projected_da - self.projected_capex - self.projected_nwc

    # ─── WACC ────────────────────────────────────────────────────────

    def _calc_wacc(self):
        """
        Weighted Average Cost of Capital

        WACC = Wd * Rd * (1 - Tax) + We * Re

        Where:
            Wd = Weight of Debt = Debt / (Debt + Market Cap)
            Rd = Cost of Debt = Interest Expense / Total Debt
            We = Weight of Equity = Market Cap / (Debt + Market Cap)
            Re = Cost of Equity via CAPM = Rf + β * (Rm - Rf)
        """
        total_capital = self.total_debt + self.market_cap
        if total_capital == 0:
            return 0.10  # Fallback

        weight_debt = self.total_debt / total_capital
        weight_equity = self.market_cap / total_capital
        cost_of_debt = self.interest_expense / max(self.total_debt, 1)
        cost_of_equity = self.risk_free_rate + self.beta * (self.market_return - self.risk_free_rate)

        wacc = weight_debt * cost_of_debt * (1 - self.tax_rate) + weight_equity * cost_of_equity
        return np.clip(wacc, 0.03, 0.30)  # Reasonable bounds

    # ─── Terminal Value ──────────────────────────────────────────────

    def _calc_terminal_value_perpetual(self):
        """
        Gordon Growth Model:
        TV = FCF_n * (1 + g) / (WACC - g)
        """
        final_fcf = self.projected_fcf.iloc[-1]
        if self.wacc <= self.terminal_growth_rate:
            return final_fcf * 20  # Fallback cap
        return final_fcf * (1 + self.terminal_growth_rate) / (self.wacc - self.terminal_growth_rate)

    def _calc_terminal_value_exit_multiple(self, ev_ebitda_multiple=12.0):
        """
        Exit Multiple Method:
        TV = Final Year EBITDA * EV/EBITDA Multiple
        """
        final_ebitda = self.projected_ebit.iloc[-1] + self.projected_da.iloc[-1]
        return final_ebitda * ev_ebitda_multiple

    # ─── DCF Calculation ─────────────────────────────────────────────

    def _calc_dcf(self):
        """
        Enterprise Value = Σ (FCF_t / (1+WACC)^(t+0.5)) + TV / (1+WACC)^n

        Uses mid-year convention (t + 0.5) to reflect cash flows received
        throughout the year rather than only at year-end.
        """
        discounted_fcf = sum(
            fcf / ((1 + self.wacc) ** (0.5 + i))
            for i, fcf in enumerate(self.projected_fcf)
        )
        discounted_tv = self.terminal_value / ((1 + self.wacc) ** self.projection_years)
        return discounted_fcf + discounted_tv

    def _calc_implied_price(self):
        """Equity Value = Enterprise Value - Debt + Cash → Price per Share"""
        equity_value = self.enterprise_value - self.total_debt + self.total_cash
        if self.shares_outstanding == 0:
            return 0
        return equity_value / self.shares_outstanding

    # ─── Sensitivity Analysis ────────────────────────────────────────

    def sensitivity_analysis(self, wacc_range=None, tgr_range=None):
        """
        Build a sensitivity table: Implied Share Price as a function
        of WACC and Terminal Growth Rate.

        Returns
        -------
        pd.DataFrame
            Matrix of implied prices indexed by WACC (rows) and TGR (cols)
        """
        if wacc_range is None:
            wacc_range = np.arange(
                max(self.wacc - 0.02, 0.04),
                self.wacc + 0.025,
                0.005
            )
        if tgr_range is None:
            tgr_range = np.arange(0.01, 0.045, 0.005)

        results = {}
        for tgr in tgr_range:
            col = {}
            for w in wacc_range:
                if w <= tgr:
                    col[f"{w:.1%}"] = np.nan
                    continue
                disc_fcf = sum(
                    fcf / ((1 + w) ** (0.5 + i))
                    for i, fcf in enumerate(self.projected_fcf)
                )
                tv = self.projected_fcf.iloc[-1] * (1 + tgr) / (w - tgr)
                disc_tv = tv / ((1 + w) ** self.projection_years)
                ev = disc_fcf + disc_tv
                equity = ev - self.total_debt + self.total_cash
                price = equity / max(self.shares_outstanding, 1)
                col[f"{w:.1%}"] = round(price, 2)
            results[f"{tgr:.1%}"] = col

        df = pd.DataFrame(results)
        df.index.name = "WACC \\ TGR"
        return df

    # ─── Implied Growth Rate ─────────────────────────────────────────

    def calc_implied_growth_rate(self, tolerance=0.01, max_iterations=100):
        """
        Use bisection to find the revenue growth rate that sets
        the implied share price equal to the current market price.
        """
        base_rev = self.hist_revenue.iloc[0]

        def price_at_growth(g):
            rev = pd.Series([base_rev * ((1 + g) ** i) for i in range(self.projection_years)])
            ebit = rev * self.ebit_margin
            tax = ebit * self.tax_rate
            ebiat = ebit - tax
            da = rev * self.da_margin
            capex = rev * self.capex_margin
            nwc = rev * self.nwc_margin
            fcf = ebiat + da - capex - nwc
            disc = sum(f / ((1 + self.wacc) ** (0.5 + i)) for i, f in enumerate(fcf))
            if self.wacc <= self.terminal_growth_rate:
                tv = fcf.iloc[-1] * 20
            else:
                tv = fcf.iloc[-1] * (1 + self.terminal_growth_rate) / (self.wacc - self.terminal_growth_rate)
            disc_tv = tv / ((1 + self.wacc) ** self.projection_years)
            ev = disc + disc_tv
            eq = ev - self.total_debt + self.total_cash
            return eq / max(self.shares_outstanding, 1)

        lo, hi = -0.50, 1.00
        for _ in range(max_iterations):
            mid = (lo + hi) / 2
            p = price_at_growth(mid)
            if abs(p - self.current_price) < tolerance:
                return mid
            if p > self.current_price:
                hi = mid
            else:
                lo = mid
        return mid  # Best approximation

    # ─── Monte Carlo Simulation ─────────────────────────────────────

    def monte_carlo(self, n_simulations=10000, seed=42):
        """
        Run a Monte Carlo simulation by randomizing key inputs:
            - Revenue growth rate (normal dist around base estimate)
            - EBIT margin (normal dist around historical avg)
            - WACC (normal dist around calculated WACC)
            - Terminal growth rate (uniform 1.5%–4.0%)

        Returns
        -------
        dict with keys: prices (array), mean, median, p10, p25, p75, p90,
                        prob_undervalued (% of sims where implied > current)
        """
        rng = np.random.default_rng(seed)

        growth_samples = rng.normal(self.revenue_growth_rate, 0.03, n_simulations)
        margin_samples = rng.normal(self.ebit_margin, 0.02, n_simulations)
        wacc_samples = rng.normal(self.wacc, 0.015, n_simulations)
        tgr_samples = rng.uniform(0.015, 0.04, n_simulations)

        base_rev = self.hist_revenue.iloc[0]
        prices = np.zeros(n_simulations)

        for s in range(n_simulations):
            g = growth_samples[s]
            m = np.clip(margin_samples[s], 0.05, 0.60)
            w = np.clip(wacc_samples[s], 0.04, 0.25)
            tgr = tgr_samples[s]

            if w <= tgr:
                tgr = w - 0.01

            rev = np.array([base_rev * ((1 + g) ** i) for i in range(self.projection_years)])
            ebit = rev * m
            ebiat = ebit * (1 - self.tax_rate)
            da = rev * self.da_margin
            cx = rev * self.capex_margin
            nwc = rev * self.nwc_margin
            fcf = ebiat + da - cx - nwc

            disc = sum(fcf[i] / ((1 + w) ** (0.5 + i)) for i in range(self.projection_years))
            tv = fcf[-1] * (1 + tgr) / (w - tgr)
            disc_tv = tv / ((1 + w) ** self.projection_years)
            ev = disc + disc_tv
            eq = ev - self.total_debt + self.total_cash
            prices[s] = eq / max(self.shares_outstanding, 1)

        return {
            "prices": prices,
            "mean": float(np.mean(prices)),
            "median": float(np.median(prices)),
            "std": float(np.std(prices)),
            "p10": float(np.percentile(prices, 10)),
            "p25": float(np.percentile(prices, 25)),
            "p75": float(np.percentile(prices, 75)),
            "p90": float(np.percentile(prices, 90)),
            "prob_undervalued": float(np.mean(prices > self.current_price) * 100),
        }

    # ─── Scenario Analysis ───────────────────────────────────────────

    def scenario_analysis(self):
        """
        Bull / Base / Bear cases with different revenue growth and margin
        assumptions. Returns a DataFrame with implied prices per scenario.

        Bull:  +2% growth, +1% margin vs base
        Base:  model defaults
        Bear:  -3% growth, -2% margin vs base
        """
        scenarios = {
            "Bull": {"growth_adj": 0.02, "margin_adj": 0.01},
            "Base": {"growth_adj": 0.00, "margin_adj": 0.00},
            "Bear": {"growth_adj": -0.03, "margin_adj": -0.02},
        }
        base_rev = self.hist_revenue.iloc[0]
        results = {}

        for name, adj in scenarios.items():
            g = self.revenue_growth_rate + adj["growth_adj"]
            m = self.ebit_margin + adj["margin_adj"]
            rev = pd.Series([base_rev * ((1 + g) ** i) for i in range(self.projection_years)])
            ebit = rev * m
            ebiat = ebit * (1 - self.tax_rate)
            da = rev * self.da_margin
            cx = rev * self.capex_margin
            nwc = rev * self.nwc_margin
            fcf = ebiat + da - cx - nwc

            disc = sum(f / ((1 + self.wacc) ** (0.5 + i)) for i, f in enumerate(fcf))
            tgr = self.terminal_growth_rate
            if self.wacc <= tgr:
                tv = fcf.iloc[-1] * 20
            else:
                tv = fcf.iloc[-1] * (1 + tgr) / (self.wacc - tgr)
            disc_tv = tv / ((1 + self.wacc) ** self.projection_years)
            ev = disc + disc_tv
            eq = ev - self.total_debt + self.total_cash
            price = eq / max(self.shares_outstanding, 1)

            results[name] = {
                "Revenue Growth": f"{g:.1%}",
                "EBIT Margin": f"{m:.1%}",
                "Implied Price": f"${price:.2f}",
                "vs Current": f"{(price - self.current_price)/self.current_price:.1%}",
            }

        return pd.DataFrame(results).T

    # ─── Exit Multiple Terminal Value ────────────────────────────────

    def exit_multiple_valuation(self, ev_ebitda_multiple=None):
        """
        Calculate implied price using EV/EBITDA exit multiple for terminal value
        instead of the Gordon Growth perpetuity method.

        If no multiple is provided, uses the sector median from the model's data.
        Default fallback is 12x.
        """
        if ev_ebitda_multiple is None:
            ev_ebitda_multiple = self._get_sector_multiple()

        final_ebitda = self.projected_ebit.iloc[-1] + self.projected_da.iloc[-1]
        tv_exit = final_ebitda * ev_ebitda_multiple

        discounted_fcf = sum(
            fcf / ((1 + self.wacc) ** (0.5 + i))
            for i, fcf in enumerate(self.projected_fcf)
        )
        discounted_tv = tv_exit / ((1 + self.wacc) ** self.projection_years)
        ev = discounted_fcf + discounted_tv
        eq = ev - self.total_debt + self.total_cash
        price = eq / max(self.shares_outstanding, 1)

        return {
            "method": "Exit Multiple (EV/EBITDA)",
            "multiple": ev_ebitda_multiple,
            "terminal_value": tv_exit,
            "enterprise_value": ev,
            "implied_price": price,
        }

    def _get_sector_multiple(self):
        """Return a reasonable EV/EBITDA multiple based on sector."""
        sector_multiples = {
            "Technology": 20.0, "Healthcare": 15.0, "Financial Services": 10.0,
            "Consumer Cyclical": 12.0, "Consumer Defensive": 14.0,
            "Industrials": 12.0, "Energy": 6.0, "Utilities": 10.0,
            "Real Estate": 15.0, "Basic Materials": 8.0,
            "Communication Services": 12.0,
        }
        return sector_multiples.get(self.sector, 12.0)

    # ─── Comparable Company Analysis ─────────────────────────────────

    def comparable_analysis(self, peer_tickers=None):
        """
        Pull valuation multiples for peer companies and compare.

        Parameters
        ----------
        peer_tickers : list of str
            Ticker symbols for comparable companies.
            If None, attempts to find peers automatically via sector.

        Returns
        -------
        pd.DataFrame with columns: Ticker, Name, Price, Market Cap,
                                    P/E, EV/EBITDA, EV/Revenue, P/S
        """
        if peer_tickers is None:
            peer_tickers = self._get_default_peers()

        all_tickers = [self.ticker] + [t for t in peer_tickers if t != self.ticker]
        rows = []

        for t in all_tickers:
            try:
                s = yf.Ticker(t)
                info = s.info
                price = info.get("previousClose", 0)
                mcap = info.get("marketCap", 0)
                pe = info.get("trailingPE", None)
                ev = info.get("enterpriseValue", 0)
                ebitda = info.get("ebitda", None)
                rev = info.get("totalRevenue", None)

                ev_ebitda = round(ev / ebitda, 1) if ebitda and ebitda > 0 else None
                ev_rev = round(ev / rev, 1) if rev and rev > 0 else None
                ps = round(mcap / rev, 1) if rev and rev > 0 else None

                rows.append({
                    "Ticker": t,
                    "Name": info.get("shortName", t),
                    "Price": f"${price:.2f}",
                    "Mkt Cap ($B)": f"${mcap/1e9:.1f}",
                    "P/E": f"{pe:.1f}" if pe else "N/A",
                    "EV/EBITDA": f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A",
                    "EV/Revenue": f"{ev_rev:.1f}x" if ev_rev else "N/A",
                    "P/S": f"{ps:.1f}x" if ps else "N/A",
                })
            except Exception:
                rows.append({"Ticker": t, "Name": "Error", "Price": "N/A"})

        df = pd.DataFrame(rows)
        return df

    def _get_default_peers(self):
        """Return default peer tickers based on sector."""
        peer_map = {
            "Technology": ["MSFT", "GOOGL", "META", "NVDA", "ORCL"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
            "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
            "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
            "Industrials": ["CAT", "HON", "UPS", "GE", "BA"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
            "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA"],
        }
        return peer_map.get(self.sector, ["SPY"])

    # ─── Summary & Output ────────────────────────────────────────────

    def projection_table(self):
        """Build a formatted projection summary table."""
        years = [f"Year {i+1}" for i in range(self.projection_years)]
        data = {
            "Revenue": self.projected_revenue.values,
            "EBIT": self.projected_ebit.values,
            "EBIAT": self.projected_ebiat.values,
            "D&A": self.projected_da.values,
            "CapEx": self.projected_capex.values,
            "ΔNWC": self.projected_nwc.values,
            "Free Cash Flow": self.projected_fcf.values,
        }
        df = pd.DataFrame(data, index=years).T
        return df.map(lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M")

    def summary(self):
        """Print a full valuation summary."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  DCF VALUATION SUMMARY — {self.company_name} ({self.ticker})")
        print(f"{sep}")
        print(f"  Sector:            {self.sector}")
        print(f"  Industry:          {self.industry}")
        print(f"  Current Price:     ${self.current_price:.2f}")
        print(f"  Shares Outstanding:{self.shares_outstanding/1e9:.3f}B")
        print(f"{sep}")

        print(f"\n  KEY ASSUMPTIONS")
        print(f"  {'─'*40}")
        print(f"  Revenue Growth (Yr1): {self.revenue_growth_rate:.2%}")
        print(f"  EBIT Margin:          {self.ebit_margin:.2%}")
        print(f"  Effective Tax Rate:   {self.tax_rate:.2%}")
        print(f"  D&A / Revenue:        {self.da_margin:.2%}")
        print(f"  CapEx / Revenue:      {self.capex_margin:.2%}")
        print(f"  ΔNWC / Revenue:       {self.nwc_margin:.2%}")

        print(f"\n  WACC COMPONENTS")
        print(f"  {'─'*40}")
        total_cap = self.total_debt + self.market_cap
        print(f"  Risk-Free Rate:       {self.risk_free_rate:.2%}")
        print(f"  Market Return:        {self.market_return:.2%}")
        print(f"  Beta:                 {self.beta:.2f}")
        print(f"  Debt Weight:          {self.total_debt/total_cap:.2%}")
        print(f"  Equity Weight:        {self.market_cap/total_cap:.2%}")
        print(f"  WACC:                 {self.wacc:.2%}")
        print(f"  Terminal Growth Rate: {self.terminal_growth_rate:.2%}")

        print(f"\n  PROJECTED FREE CASH FLOWS")
        print(f"  {'─'*40}")
        print(self.projection_table().to_string())

        print(f"\n  VALUATION OUTPUT")
        print(f"  {'─'*40}")
        print(f"  Terminal Value:       ${self.terminal_value/1e9:.2f}B")
        print(f"  Enterprise Value:     ${self.enterprise_value/1e9:.2f}B")
        print(f"  (-) Total Debt:       ${self.total_debt/1e9:.2f}B")
        print(f"  (+) Total Cash:       ${self.total_cash/1e9:.2f}B")
        eq = self.enterprise_value - self.total_debt + self.total_cash
        print(f"  Equity Value:         ${eq/1e9:.2f}B")
        print(f"  Implied Share Price:  ${self.implied_share_price:.2f}")
        print(f"  Current Market Price: ${self.current_price:.2f}")

        direction = "UNDERVALUED ↑" if self.margin_of_safety > 0 else "OVERVALUED ↓"
        print(f"  Margin of Safety:     {self.margin_of_safety:.2%} ({direction})")

        # Implied growth rate
        try:
            igr = self.calc_implied_growth_rate()
            print(f"  Implied Growth Rate:  {igr:.2%}")
        except Exception:
            pass

        print(f"\n  SENSITIVITY ANALYSIS (Implied Price)")
        print(f"  {'─'*40}")
        print(self.sensitivity_analysis().to_string())

        # Scenario Analysis
        print(f"\n  SCENARIO ANALYSIS")
        print(f"  {'─'*40}")
        print(self.scenario_analysis().to_string())

        # Exit Multiple Valuation
        print(f"\n  EXIT MULTIPLE VALUATION")
        print(f"  {'─'*40}")
        em = self.exit_multiple_valuation()
        print(f"  Method:              {em['method']}")
        print(f"  EV/EBITDA Multiple:  {em['multiple']:.1f}x")
        print(f"  Terminal Value:      ${em['terminal_value']/1e9:.2f}B")
        print(f"  Implied Price:       ${em['implied_price']:.2f}")

        # Monte Carlo
        print(f"\n  MONTE CARLO SIMULATION (10,000 runs)")
        print(f"  {'─'*40}")
        mc = self.monte_carlo()
        print(f"  Mean Implied Price:  ${mc['mean']:.2f}")
        print(f"  Median:              ${mc['median']:.2f}")
        print(f"  10th Percentile:     ${mc['p10']:.2f}")
        print(f"  25th Percentile:     ${mc['p25']:.2f}")
        print(f"  75th Percentile:     ${mc['p75']:.2f}")
        print(f"  90th Percentile:     ${mc['p90']:.2f}")
        print(f"  Prob. Undervalued:   {mc['prob_undervalued']:.1f}%")

        print(f"\n{sep}\n")

    def to_json(self, filepath=None):
        """Export model results to JSON."""
        mc = self.monte_carlo()
        em = self.exit_multiple_valuation()
        output = {
            "ticker": self.ticker,
            "company": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "current_price": round(self.current_price, 2),
            "implied_price_dcf": round(self.implied_share_price, 2),
            "implied_price_exit_multiple": round(em["implied_price"], 2),
            "margin_of_safety": round(self.margin_of_safety, 4),
            "wacc": round(self.wacc, 4),
            "terminal_growth_rate": round(self.terminal_growth_rate, 4),
            "ebit_margin": round(self.ebit_margin, 4),
            "tax_rate": round(self.tax_rate, 4),
            "enterprise_value": round(self.enterprise_value, 2),
            "terminal_value": round(self.terminal_value, 2),
            "projected_fcf": [round(float(f), 2) for f in self.projected_fcf],
            "monte_carlo": {
                "mean": round(mc["mean"], 2),
                "median": round(mc["median"], 2),
                "p10": round(mc["p10"], 2),
                "p90": round(mc["p90"], 2),
                "prob_undervalued": round(mc["prob_undervalued"], 1),
            },
        }
        if filepath:
            with open(filepath, "w") as f:
                json.dump(output, f, indent=2)
        return output


# ─── CLI Entry Point ─────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    parser = ArgumentParser(description="5-Year DCF Valuation Model")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--tgr", type=float, default=0.03, help="Terminal Growth Rate (default: 0.03)")
    parser.add_argument("--riskfree", default="^TNX", help="Risk-free rate ticker (default: ^TNX)")
    parser.add_argument("--market", default="VTI", help="Market return ticker (default: VTI)")
    parser.add_argument("--json", default=None, help="Export results to JSON file")
    parser.add_argument("--comps", nargs="*", default=None,
                        help="Run comparable company analysis (optional: list peer tickers)")
    args = parser.parse_args()

    model = DCFModel(
        ticker=args.ticker,
        terminal_growth_rate=args.tgr,
        risk_free_ticker=args.riskfree,
        market_ticker=args.market,
    )

    model.summary()

    if args.comps is not None:
        print(f"\n  COMPARABLE COMPANY ANALYSIS")
        print(f"  {'─'*40}")
        peers = args.comps if args.comps else None
        print(model.comparable_analysis(peers).to_string(index=False))
        print()

    if args.json:
        model.to_json(args.json)
        print(f"  Results exported to {args.json}")


if __name__ == "__main__":
    main()

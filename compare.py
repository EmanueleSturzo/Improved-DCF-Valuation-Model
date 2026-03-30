"""
Multi-Stock Comparison Tool
============================
Run the DCF model across multiple tickers and produce
a side-by-side valuation comparison table.

Usage:
    python compare.py --tickers AAPL MSFT GOOGL META
"""

import warnings
import pandas as pd
from argparse import ArgumentParser
from dcf_model import DCFModel


def compare(tickers, tgr=0.03):
    """Run DCF on a list of tickers and return a comparison DataFrame."""
    results = []

    for t in tickers:
        try:
            model = DCFModel(ticker=t, terminal_growth_rate=tgr)
            igr = model.calc_implied_growth_rate()
            results.append({
                "Ticker": model.ticker,
                "Company": model.company_name,
                "Sector": model.sector,
                "Price": f"${model.current_price:.2f}",
                "Implied Price": f"${model.implied_share_price:.2f}",
                "Margin of Safety": f"{model.margin_of_safety:.1%}",
                "WACC": f"{model.wacc:.2%}",
                "EBIT Margin": f"{model.ebit_margin:.1%}",
                "Rev Growth": f"{model.revenue_growth_rate:.1%}",
                "Implied Growth": f"{igr:.1%}",
                "EV ($B)": f"${model.enterprise_value/1e9:.1f}",
                "Signal": "BUY" if model.margin_of_safety > 0.15
                          else "HOLD" if model.margin_of_safety > -0.10
                          else "SELL",
            })
            print(f"  ✓ {t}")
        except Exception as e:
            print(f"  ✗ {t}: {e}")

    df = pd.DataFrame(results)
    return df


def main():
    warnings.filterwarnings("ignore")
    parser = ArgumentParser(description="Compare DCF valuations across multiple stocks")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols")
    parser.add_argument("--tgr", type=float, default=0.03, help="Terminal Growth Rate")
    parser.add_argument("--output", default=None, help="Save to CSV file")
    args = parser.parse_args()

    print(f"\n  Running DCF model for {len(args.tickers)} stocks...\n")
    df = compare(args.tickers, args.tgr)

    print(f"\n{'='*100}")
    print("  VALUATION COMPARISON")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"  Saved to {args.output}\n")


if __name__ == "__main__":
    main()

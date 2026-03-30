"""
DCF Model Visualization
========================
Generate charts for the DCF valuation model:
    - Projected FCF waterfall
    - Sensitivity heatmap
    - Valuation bridge (EV → Equity Value → Price)

Usage:
    python visualize.py --ticker AAPL
    python visualize.py --ticker MSFT --save

Requires: matplotlib, seaborn
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from argparse import ArgumentParser

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from dcf_model import DCFModel


def plot_fcf_projection(model, ax=None):
    """Bar chart of projected Free Cash Flows with growth annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    years = [f"Year {i+1}" for i in range(model.projection_years)]
    fcf = model.projected_fcf.values / 1e9  # Convert to billions

    colors = ["#2563eb" if f >= 0 else "#dc2626" for f in fcf]
    bars = ax.bar(years, fcf, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, fcf):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fcf)*0.02,
                f"${val:.2f}B", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(f"{model.ticker} — Projected Free Cash Flow", fontsize=13, fontweight="bold")
    ax.set_ylabel("Free Cash Flow ($B)")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_sensitivity_heatmap(model, ax=None):
    """Heatmap of implied share prices across WACC and TGR scenarios."""
    sens = model.sensitivity_analysis()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    data = sens.astype(float)

    if HAS_SEABORN:
        sns.heatmap(data, annot=True, fmt=".0f", cmap="RdYlGn", center=model.current_price,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Implied Price ($)"})
    else:
        im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Implied Price ($)")

    ax.set_title(f"{model.ticker} — Sensitivity Analysis", fontsize=13, fontweight="bold")
    ax.set_xlabel("Terminal Growth Rate")
    ax.set_ylabel("WACC")
    return ax


def plot_valuation_bridge(model, ax=None):
    """Waterfall chart showing EV → Equity Value → Implied Price."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Build waterfall components
    dcf_value = sum(
        fcf / ((1 + model.wacc) ** (0.5 + i))
        for i, fcf in enumerate(model.projected_fcf)
    ) / 1e9
    tv_value = (model.terminal_value / ((1 + model.wacc) ** model.projection_years)) / 1e9
    debt = -model.total_debt / 1e9
    cash = model.total_cash / 1e9

    labels = ["PV of FCFs", "PV of Terminal\nValue", "Enterprise\nValue",
              "(−) Debt", "(+) Cash", "Equity\nValue"]
    ev = dcf_value + tv_value
    eq = ev + debt + cash
    values = [dcf_value, tv_value, ev, debt, cash, eq]

    # Colors
    colors = ["#2563eb", "#2563eb", "#1e40af", "#dc2626", "#16a34a", "#1e40af"]

    # Waterfall logic
    bottoms = [0, dcf_value, 0, ev, ev + debt, 0]
    heights = [dcf_value, tv_value, ev, abs(debt), cash, eq]

    bars = ax.bar(labels, heights, bottom=bottoms, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55)

    for bar, h, b in zip(bars, heights, bottoms):
        val = h if b >= 0 else -h
        y = b + h + max(values)*0.02
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"${h:.1f}B", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(f"{model.ticker} — Valuation Bridge ($B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value ($B)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_monte_carlo(model, ax=None):
    """Histogram of Monte Carlo simulation results."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    mc = model.monte_carlo()
    prices = mc["prices"]
    prices_clean = prices[(prices > np.percentile(prices, 1)) & (prices < np.percentile(prices, 99))]

    ax.hist(prices_clean, bins=80, color="#2563eb", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(model.current_price, color="#dc2626", linewidth=2, linestyle="--", label=f"Current ${model.current_price:.0f}")
    ax.axvline(mc["median"], color="#16a34a", linewidth=2, linestyle="--", label=f"Median ${mc['median']:.0f}")
    ax.axvline(mc["p10"], color="#9ca3af", linewidth=1, linestyle=":", label=f"P10 ${mc['p10']:.0f}")
    ax.axvline(mc["p90"], color="#9ca3af", linewidth=1, linestyle=":", label=f"P90 ${mc['p90']:.0f}")

    ax.set_title(f"{model.ticker} — Monte Carlo (n=10,000)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Implied Share Price ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_scenario_analysis(model, ax=None):
    """Bar chart of Bull/Base/Bear implied prices."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = model.scenario_analysis()
    names = scenarios.index.tolist()
    prices = [float(scenarios.loc[n, "Implied Price"].replace("$", "")) for n in names]
    colors = ["#16a34a", "#2563eb", "#dc2626"]

    bars = ax.barh(names, prices, color=colors, edgecolor="white", height=0.5)
    ax.axvline(model.current_price, color="#f59e0b", linewidth=2, linestyle="--",
               label=f"Current ${model.current_price:.0f}")

    for bar, val in zip(bars, prices):
        ax.text(val + max(prices)*0.02, bar.get_y() + bar.get_height()/2,
                f"${val:.0f}", ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_title(f"{model.ticker} — Scenario Analysis", fontsize=13, fontweight="bold")
    ax.set_xlabel("Implied Share Price ($)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def generate_report(model, save=False, output_dir="output"):
    """Generate a full-page valuation report with 6 charts."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        f"DCF Valuation Report — {model.company_name} ({model.ticker})\n"
        f"Implied Price: ${model.implied_share_price:.2f}  |  "
        f"Current: ${model.current_price:.2f}  |  "
        f"Margin of Safety: {model.margin_of_safety:.1%}",
        fontsize=15, fontweight="bold", y=0.98
    )

    # Row 1: FCF Projection + Sensitivity Heatmap
    plot_fcf_projection(model, ax=axes[0, 0])
    plot_sensitivity_heatmap(model, ax=axes[0, 1])

    # Row 2: Valuation Bridge + Monte Carlo
    plot_valuation_bridge(model, ax=axes[1, 0])
    plot_monte_carlo(model, ax=axes[1, 1])

    # Row 3: Scenario Analysis + Key Metrics
    plot_scenario_analysis(model, ax=axes[2, 0])

    ax6 = axes[2, 1]
    ax6.axis("off")
    mc = model.monte_carlo()
    em = model.exit_multiple_valuation()
    metrics_text = (
        f"{'KEY METRICS':^40}\n"
        f"{'─'*40}\n"
        f"{'WACC:':<25}{model.wacc:.2%}\n"
        f"{'Terminal Growth Rate:':<25}{model.terminal_growth_rate:.2%}\n"
        f"{'EBIT Margin:':<25}{model.ebit_margin:.2%}\n"
        f"{'Tax Rate:':<25}{model.tax_rate:.2%}\n"
        f"{'Beta:':<25}{model.beta:.2f}\n"
        f"{'D&A / Revenue:':<25}{model.da_margin:.2%}\n"
        f"{'CapEx / Revenue:':<25}{model.capex_margin:.2%}\n"
        f"{'─'*40}\n"
        f"{'DCF Price:':<25}${model.implied_share_price:.2f}\n"
        f"{'Exit Multiple Price:':<25}${em['implied_price']:.2f}\n"
        f"{'Monte Carlo Median:':<25}${mc['median']:.2f}\n"
        f"{'MC P10–P90 Range:':<25}${mc['p10']:.0f}–${mc['p90']:.0f}\n"
        f"{'Prob. Undervalued:':<25}{mc['prob_undervalued']:.1f}%\n"
        f"{'─'*40}\n"
        f"{'Enterprise Value:':<25}${model.enterprise_value/1e9:.2f}B\n"
        f"{'Terminal Value:':<25}${model.terminal_value/1e9:.2f}B\n"
    )
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{model.ticker}_dcf_report.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Report saved to {filepath}")
        plt.close()
    else:
        plt.show()


def main():
    warnings.filterwarnings("ignore")
    parser = ArgumentParser(description="Generate DCF valuation charts")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--tgr", type=float, default=0.03, help="Terminal Growth Rate")
    parser.add_argument("--save", action="store_true", help="Save charts to output/")
    args = parser.parse_args()

    print(f"\n  Building DCF model for {args.ticker}...")
    model = DCFModel(ticker=args.ticker, terminal_growth_rate=args.tgr)
    print(f"  Generating report...")
    generate_report(model, save=args.save)
    print(f"  Done.\n")


if __name__ == "__main__":
    main()

from pathlib import Path
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'report'
FIGURES = REPORT / 'figures'
LABELS = {'off_market': 'Without asking price', 'list_aware': 'With asking price'}
COLORS = {'off_market': '#286b91', 'list_aware': '#c16a30'}

def finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches='tight')
    plt.close(fig)

def build_figures():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    monthly = pd.read_csv(ROOT / 'analysis/eda/monthly.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    dates = pd.to_datetime(monthly.month)
    axes[0].plot(dates, monthly.median_price / 1000, color='#286b91')
    axes[0].set_ylabel('Median sale price ($ thousands)')
    axes[1].plot(dates, monthly.median_sold_to_list, color='#c16a30')
    axes[1].axhline(1, color='grey', linewidth=0.8)
    axes[1].set_ylabel('Median sale price / asking price')
    for ax in axes:
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(axis='x', rotation=30)
    finish(fig, 'market.png')
    errors = pd.read_csv(ROOT / 'research_v2/diagnostics/error_breakdown.csv')
    fig, ax = plt.subplots(figsize=(9, 3))
    for track in LABELS:
        rows = errors[(errors.track == track) & (errors.grouping == 'month')].sort_values('group')
        ax.plot(rows['group'], rows.mdape, marker='o', label=LABELS[track], color=COLORS[track])
    ax.set_ylabel('Median percentage error')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(frameon=False)
    finish(fig, 'monthly_error.png')
    bands = ['under_500k', '500k_to_1m', '1m_to_2m', 'over_2m']
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, track in enumerate(LABELS):
        rows = errors[(errors.track == track) & (errors.grouping == 'price_band')].set_index('group').loc[bands]
        ax.bar([x + i * 0.36 for x in range(4)], rows.mdape, width=0.36, label=LABELS[track], color=COLORS[track])
    ax.set_xticks([x + 0.18 for x in range(4)], ['Below $500k', r'\$500k to \$1m', r'\$1m to \$2m', '$2m or more'])
    ax.set_ylabel('Median percentage error')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    finish(fig, 'price_error.png')

def build_price_extremes():
    rows = pd.read_csv(ROOT / 'analysis/final_checks/price_groups.csv')
    labels = ['Cheapest 10%', 'Middle 80%', 'Most expensive 10%']
    fig, ax = plt.subplots(figsize=(9, 2.5))
    for i, track in enumerate(LABELS):
        values = rows[rows.track == track].set_index('group').loc[labels, 'mdape_pct']
        bars = ax.bar([x + i * 0.36 for x in range(3)], values, width=0.36, color=COLORS[track], label=LABELS[track])
        ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set_xticks([x + 0.18 for x in range(3)], labels)
    ax.set_ylabel('Median percentage error')
    ax.set_ylim(0, 17)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    finish(fig, 'price_extremes.png')

def main():
    build_figures()
    build_price_extremes()
    compiler = shutil.which('pdflatex')
    if not compiler:
        print('Graphs saved. To build the PDF, upload the report folder to Overleaf and compile report.tex.')
        return
    for _ in range(2):
        result = subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error', 'report.tex'], cwd=REPORT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
    print('Saved report/report.pdf')
if __name__ == '__main__':
    main()

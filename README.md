# Predicting FIFA World Cup Outcomes

A project that investigates **upset patterns** and **stage-specific dynamics** in FIFA World Cup matches, with a forward look at the 2026 North American World Cup.

> **Course:** CSCI-UA 381 — Programming Tools for the Data Scientist  
> **Team:** Asher Kim (ck3292), Jason Liu (jl13869)

---

## Quick Start

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# 0. Install uv (one-time, skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 1. Create a virtual environment (uv picks a compatible Python automatically)
uv venv --python 3.12
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell

# 2. Install dependencies into the venv
uv pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter

# 3a. Run interactively
uv run jupyter notebook finalproject.ipynb

# 3b. — OR — execute end-to-end from the terminal
uv run jupyter nbconvert --to notebook --execute finalproject.ipynb \
    --output finalproject.ipynb \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3
```

End-to-end execution takes about 60–90 seconds on a modern laptop.

> `uv run` automatically uses the local `.venv`, so the explicit `source .venv/bin/activate` step is optional if you always prefix commands with `uv run`.

---

## Prerequisites

| Requirement  | Version       | Notes                   |
| ------------ | ------------- | ----------------------- |
| uv           | ≥ 0.4         | environment manager     |
| Python       | 3.10 or newer | Tested on 3.12          |
| pandas       | ≥ 2.0         | data manipulation       |
| numpy        | ≥ 1.24        | numerical operations    |
| scipy        | ≥ 1.10        | statistical tests       |
| scikit-learn | ≥ 1.3         | machine-learning models |
| matplotlib   | ≥ 3.7         | plotting                |
| seaborn      | ≥ 0.13        | statistical plots       |
| jupyter      | latest        | notebook interface      |

> No internet connection is required at run time — the two CSV files in this folder are the only data sources.

---

## Project Structure

```
ds381 final/
├── README.md                       ← this file
├── PROJECT_SUMMARY.md              ← detailed write-up of every section + results
├── finalproject.ipynb              ← main deliverable (32 cells, runs end-to-end)
├── results.csv                     ← raw international match results (1872–2024)
├── fifa_ranking-2024-06-20.csv     ← FIFA World Rankings (1992–2024)
└── utils/
    └── features.py                 ← legacy helper module — its functions are now
                                      inlined into the notebook (cell 11), so this
                                      file is no longer required for execution
```

The notebook is **self-contained**. As long as `results.csv` and `fifa_ranking-2024-06-20.csv` sit next to the notebook, it runs without any extra files.

---

## What the Notebook Does

| Section                                             | What happens                                                                                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **§1–3 (cells 1–10)**                               | Load CSVs, filter to FIFA World Cup, derive `result`, merge FIFA rankings, build the rule-based upset label, three intro bar charts.                                     |
| **Inlined helpers (cell 11)**                       | Defines `compute_recent_form` and `label_upset` (originally in `utils/features.py`).                                                                                     |
| **Recent-form & probabilistic upset (cells 12–13)** | Adds `home_lastN_*`, `away_lastN_*`, `recent_win_pct`, `upset_prob_binary`, `upset_severity`, `p_home`, `p_away`.                                                        |
| **§4 Extended feature engineering (cells 14–17)**   | Drops 72 NaN-score rows, adds `stage` / `is_knockout`, leakage-free head-to-head history, altitude / continent, rest days.                                               |
| **§5 Comprehensive EDA (cells 18–21)**              | Descriptive statistics, skew / kurtosis, correlations, group summaries; six-panel dashboard plus extra time-trend / scatter plots.                                       |
| **§6 Statistical tests (cells 22–23)**              | Chi-square, Welch's t, Mann–Whitney U, one-way ANOVA, Pearson + Spearman, two-proportion Z-test.                                                                         |
| **§7 Machine learning (cells 24–28)**               | Three multiclass models + most-frequent baseline + 5-fold stratified CV; confusion matrix + feature importance; stage-specific Random Forest; binary upset GBM with ROC. |
| **§8 2026 World Cup insights (cells 29–30)**        | Altitude-bucket analysis of goals & upsets + Welch's t-test.                                                                                                             |
| **§9 Conclusion (cell 31)**                         | Findings, recommendations, limitations, and explicit alignment with stated objectives.                                                                                   |

For a deeper explanation of every section, see **`PROJECT_SUMMARY.md`**.

---

## Running Tips

- **Run All from top to bottom.** Cells reuse `df_wc` cumulatively, so out-of-order execution will leave columns missing.
- **Plots open inline.** If your Jupyter front-end doesn't render figures, add `%matplotlib inline` to the top of the notebook.
- **Random seeds.** The ML pipeline uses `random_state=42` everywhere, so results are deterministic across machines (modulo BLAS / scikit-learn version drift).
- **Long-running cell.** Cell 12 (recent-form computation) iterates row-by-row over ~570 matches and takes ~10–20 s. This is normal.

---

## Troubleshooting

| Symptom                                                                        | Likely cause                                                                                    | Fix                                                                                                                                                           |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `command not found: uv`                                                        | uv is not installed or not on `PATH`                                                            | Run the install command in the Quick Start, then restart your shell (or `source ~/.zshrc` / `~/.bashrc`).                                                     |
| `ModuleNotFoundError: No module named 'seaborn'` (or any other listed package) | Dependency not installed in the active Python environment                                       | Re-run `uv pip install …` from the Quick Start, ensuring the `.venv` is activated (or use `uv run` to auto-resolve).                                          |
| `NoSuchKernel: conda-base-py` when running `nbconvert`                         | Notebook metadata refers to a kernel that doesn't exist on your machine                         | Add `--ExecutePreprocessor.kernel_name=python3` (already in the Quick Start command) or open the notebook in Jupyter and *Kernel → Change kernel → Python 3*. |
| `FileNotFoundError: results.csv`                                               | Notebook was opened from a different working directory                                          | `cd` into the project folder before launching Jupyter, or move the CSVs next to the notebook.                                                                 |
| `AssertionError: assert "home_lastN_wins" in df_wc.columns`                    | Cell 11 (inlined helpers) was skipped, so cell 12's call to `compute_recent_form` was never run | Re-run from cell 1, or use *Cell → Run All* / *Run All Above*.                                                                                                |
| Plots don't appear                                                             | Front-end didn't pick up the matplotlib inline backend                                          | Insert `%matplotlib inline` at the top, restart the kernel, and *Run All*.                                                                                    |

---

## How to Reproduce the Reported Results

```bash
git clone <this-repo>          # or just download the folder
cd "ds381 final"
uv venv --python 3.12 && source .venv/bin/activate
uv pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
uv run jupyter nbconvert --to notebook --execute finalproject.ipynb \
    --output finalproject.ipynb \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3
```

The headline numbers (Random Forest test accuracy ≈ 0.53, ANOVA p ≈ 2 × 10⁻⁷, etc.) are reproducible to the third decimal on any machine running the package versions listed above.

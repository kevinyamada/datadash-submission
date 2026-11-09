# Toronto home price prediction

This is our work for SSU Data Dash 2026. The goal is to predict sale prices for synthetic Toronto homes. There are two tasks: estimate a home without its asking price, and predict a sale when its asking price is known.

This public submission is on the `main` branch. Team members: Kevin Ryuichi Yamada and Luka Tosic.

[Report PDF](report/report.pdf) · [Presentation video](https://www.youtube.com/watch?v=Sgtuz8VY2iQ) · [Editable report source](report/report.tex)

The video is 4 minutes 42 seconds. The two final prediction files are in `submissions`.

## Results

Each validation month from December 2025 to May 2026 uses only sales before that month. There are 10,069 homes per task.

| Task | Model | Median percentage error | Mean percentage error |
| :--- | :--- | ---: | ---: |
| Without asking price | LightGBM and spline ridge blend | 7.44% | 9.87% |
| With asking price | CatBoost | 4.02% | 5.09% |

Training once before February 2026 and predicting February to May gives median errors of 7.64% and 4.14%. These are historical results, not hidden test results. The official months were seen in earlier work. Later experiments reused historical periods and found no convincing replacement for the current models.

## Verification status

On October 28, Kevin reported all 34 tests passing in 10.69 seconds in Python 3.13.5 on Windows. The pipeline test uses small training settings while exercising the selected model families. The original full settings pipeline test also passed separately. Both actual submission files passed the submission checker. These checks do not measure hidden test accuracy.

Additional saved prediction comparisons did not justify replacing the selected models. Their outputs are in `research_v4`, with decisions explained in `ANALYSIS.md`. A short meeting guide is in `report/meeting_notes.md`.

## Run the project

Use Python 3.13 for the closest match to the recorded Windows environment. From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe run.py validate
.\.venv\Scripts\python.exe run.py report
.\.venv\Scripts\python.exe run.py predict
.\.venv\Scripts\python.exe run.py check
```

The last two commands write and check `submissions/submission_off_market.csv` and `submissions/submission_list_aware.csv`. They must contain one row per required home, with columns `ml_num,predicted_price`.

To reproduce the separate analysis files used in the report:

```powershell
.\.venv\Scripts\python.exe validate_v2.py
.\.venv\Scripts\python.exe frozen_check_v2.py
.\.venv\Scripts\python.exe diagnostics_v2.py
.\.venv\Scripts\python.exe build_report.py
```

The last command uses saved results to make the graphs. If `pdflatex` is installed, it also builds the PDF. Otherwise, upload the `report` folder to Overleaf and compile `report.tex`. A compiled PDF is included in this repository.

For the full test suite and optional research models:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

The large research scripts are optional and can take many hours. They are not needed to reproduce the final predictions. Do not rerun them just to check the project.

## How the models work

The first model predicts log sale price. It combines LightGBM with spline ridge regression using an equal weight geometric mean. LightGBM includes a bias correction estimated using only an earlier split of its own training data. The spline model adds smooth curves, neighbourhood trends and TF IDF description features.

The second model predicts log sale price divided by asking price. CatBoost learns the adjustment, which is then converted back to dollars. It uses property features and fixed description flags.

Floor area ranges use midpoints. Categories, missing value handling and text preprocessing are learned inside each training split. The model settings are stored in `model_config_v2.json`.

Neither model uses sale date, days on market or the external valuation as a predictor. Sale date is used only to split historical records. The first model never uses asking price. IDs are used to match rows, not to predict prices.

## Limits

All data are synthetic. These results do not prove the models work on real Toronto homes. Expensive and unusual homes are harder to predict. The spline component can extrapolate poorly in some market periods.

Intervals estimated from December and January errors covered 90.00% of later sales without asking price and 87.34% with asking price under monthly refitting. The target was 90%. These are measured historical results, not coverage guarantees.

The starter uses sale month cutoffs. Some homes were listed before those cutoffs, so this is not an exact reconstruction of information available on each original listing date. Additional historical tests include a separate subset listed after the cutoff.

## Files

| File or folder | Purpose |
| :--- | :--- |
| `models.py` | Selects final models and keeps the previous models |
| `advanced_models.py` | Shared features and model fitting |
| `spline_model.py` | Spline regression and description text |
| `features.py` | Basic property features |
| `report` | PDF, LaTeX source, graphs and video script |
| `research_v2` | Selection results, validation and diagnostics |
| `research_v3` | Summary and manifest from the larger comparison |
| `analysis/eda` | Initial data analysis |
| `submissions` | Predictions for the two hidden test files |

The only predictive data source is https://huggingface.co/datasets/Baliyang/SSU_data_dash at revision `96a258439e2a85f650699be8339e52d8ac5614b7`. The download script checks file hashes. The original starter is https://github.com/erwinzhang7/SSU-Data-Dash. Its licence and attribution are retained.

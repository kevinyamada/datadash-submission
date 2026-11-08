# Local model comparisons

Run model fitting on Kevin's computer. ChatGPT prepares code and analyses returned results.

Install requirements-research.txt, then run local_research.py. The runner verifies the raw data, compares configurations sequentially, saves each completed result and creates research_results.zip. Send that ZIP back for analysis. Successful completed configurations are skipped when resuming. Errors are logged.

This first phase trains on sales before July 2025 and evaluates July-October 2025. It does not replace the current submission models. Shortlist promising models for a second historical window before selecting and reporting official December-May validation.

Use --threads 4, or --threads 2 while using the computer for other tasks. Keep the computer awake. Ctrl+C stops training; rerun the same command to resume. To package partial results without training, run python local_research.py --package-only.

The catalogue compares LightGBM, CatBoost, XGBoost, ExtraTrees, histogram gradient boosting, ridge and spline ridge. Variants examine objectives, model size, recency, description features, strictly earlier-sale comparables, area offsets and earlier-residual calibration. Simple blends are evaluated after shortlisting.

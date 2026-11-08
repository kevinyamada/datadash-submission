# Model comparison protocol

Started October 27, 2026. The original competition-solution commit is 276d717fe363de065f91584e7dbbe9ef9bac3828.

Rules were checked against the organiser's handout.md and rubric.md before this work. No real property data, external valuation inputs, hidden labels, or off-market asking-price features are permitted. Every transformation learns from the current training portion only. No test-table distribution is used to select models. The ID is used for alignment and never as a predictor.

The original December 2025-May 2026 evaluation has already been viewed. It is a reporting period, not an untouched holdout. New selection uses earlier historical windows, with full logs retained. We cannot undo earlier exploratory access to labelled data and will disclose that fact.

Stage 1: Train before July 1, 2025 and predict July-October 2025 with a frozen model. Compare a fixed catalogue of model families, objectives, tree sizes, recency choices and feature sets. Both tracks use identical home IDs and timing. Model failures and runtimes remain in the log.

Stage 2: Train before November 1, 2024 and predict November 2024-February 2025. Retain a small shortlist from stage 1, including the incumbent and family diversity. Rank by average MdAPE over the two historical windows and use MAPE and stability as secondary criteria. Test simple equal-weight blends; avoid unrestricted weight optimization.

Stage 3: Lock the final configurations and weights. Run the six official monthly folds and a frozen February-May check. If the new method clearly fails the reporting check, retain the incumbent and disclose that decision, rather than secretly tuning the same months. Report all decisions.

Use full training data to write submissions only after selection. Recheck IDs, columns, finite positive predictions, tests, subgroup errors and uncertainty. Preserve the old submission as a benchmark. Update documentation with fresh measured results only.

Compute is staged rather than attempting an impossible exhaustive search. Fixed seeds, fixed dataset hashes, resumeable outputs and append-only experiment records support reproduction.

Additional prespecified feature tests before official scoring: strict earlier-sale comparable summaries, spline ridge with/without TF-IDF, log-area offsets, and a reduced market-only list-aware feature set. The initial model-family runs continue; no official-period scores have been used in this selection. The enhanced-feature regex fixes dated versus updated and adds interpretable condition flags.

Resource adjustment: the initial 250-tree ExtraTrees run was interrupted before scoring as parallel jobs approached the 8 GiB memory limit. Its replacement uses 128 trees, depth at most 24 and at least 8 samples per leaf. XGBoost runs completed beforehand are retained. No result was available for the interrupted forest.

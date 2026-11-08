# Final model diagnostics

The runtime and price group files were supplied from Kevin's completed Windows run on November 3, 2026. Runtime used four CPU threads and all 150,000 labelled sales. The first batch and three repeated predictions are reported separately. These are measurements from one machine and one fit per task, not a general speed guarantee. No submissions were changed by the benchmark.

Price groups use the 10th and 90th percentiles of actual official validation sale price. The boundaries are $440,480 and $1,761,240. Each tail has 1,007 homes per task, while the middle has 8,055. Groups are retrospective diagnostics and never model inputs.

The property type table was derived separately from the same saved official predictions. Run python summarise_price_types.py to reproduce price_extremes_by_type.csv without training. Thresholds are shared across all property types. No homes appear for some combinations, and groups below 30 should be treated as descriptive only. Even the expensive condo group has only 45 homes, so its result should not be treated as a precise future error estimate.

Run python build_report.py to rebuild figures and the PDF when pdfLaTeX is available. The report and video script present the same final models and findings. Team names and an accessible video URL still need to be finalised.

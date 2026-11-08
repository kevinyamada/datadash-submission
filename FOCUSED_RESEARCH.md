# Focused checks using saved predictions

Run focused_research.py in the final project folder. Pass the original project folder with --saved-root so the script can find the earlier saved predictions or historical_stress_results.zip. The script reads those files without changing them. It does not download data, train models, use hidden prices or change submissions.

The first historical window, July to October 2023, supplies earlier errors for calibration. Evaluation uses March to June 2024, July to October 2024 and March to June 2025. Corrections in each period use only outcomes from completed earlier windows. All these windows were inspected during previous work. This is exploratory model selection, not an untouched test.

## Point predictions

Compare geometric blends with LightGBM weights 0.50, 0.75 and 1.00. The current weight is 0.50. Evaluate each with and without a price correction. The correction groups homes by their predicted price, never their unknown actual price. It applies only above a predicted price of $1.5 million. Earlier median log residuals are shrunk toward zero using n divided by n plus 500, require at least 200 earlier examples and are capped at a factor between 1 divided by 1.10 and 1.10.

A candidate is flagged for further confirmation only if its average median percentage error improves by at least 0.10 percentage points, average mean percentage error worsens by no more than 0.10 points, no window worsens by more than 0.20 points in median error, and at least two of three windows improve. These are practical screening rules, not statistical significance tests. No automatic replacement occurs. Read property, actual price, month and listing timing breakdowns too. Actual price groups are used only for retrospective error analysis.

This correction tests bias adjustment, not a different training loss. Further training should only follow a specific useful result from this check.

## Intervals

For the model with asking price, compare earlier pooled residuals, the latest completed historical window and conservative property type radii. The latter uses the larger of the pooled radius and the property radius, requiring at least 200 earlier cases for the property radius. The intended coverage is 90 percent.

If official monthly predictions are available, also compare the original fixed December and January calibration with expanding calibration, the latest two completed months and conservative property type radii. This monthly update assumes outcomes become available each month. It cannot be used as evidence that intervals will improve for four hidden months without new outcomes. The historical checks keep calibration fixed over each four month horizon.

Read coverage together with width and interval score. Wider intervals can increase coverage without making the point estimate better. The interval score penalises width and missed outcomes; lower is better. Neither temporal scheme guarantees future coverage under market drift.

## Outputs and runtime

The script saves a separate timestamped folder in focused_results. It writes each completed window before proceeding and packages the CSV and JSON results into a ZIP beside the folder. A separate parent process stops the worker at the specified time limit, which defaults to 15 minutes. There are no paid services. If files are missing, it stops with their names instead of repeating training. Existing results remain intact; each rerun creates a separate output folder.

The ZIP contains the plan, source file hashes, per window scores, error groups, interval coverage and width, fitted correction factors, a ranking and run status. A failed or interrupted run can still produce a ZIP, so check run_status.json before interpreting results. Send the ZIP for review before changing the final model or report.

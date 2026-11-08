# Results and decisions

The current models are the equal weight LightGBM and spline ridge blend without asking price, and CatBoost with asking price. The official monthly validation errors are 7.44% and 4.02% MdAPE. Mean errors are 9.87% and 5.09%.

We first screened 70 fits, confirmed 15 shortlisted fits on a second historical period, and compared individual models with simple equal weight pairs. The selected models were then checked on the official months and on February to May with no refitting. That longer horizon produced median errors of 7.64% and 4.14%.

The later historical stress test found that the blend beat our previous model in all four periods, but its spline component was unstable. In March to June 2025, the spline alone scored 15.03% MdAPE. LightGBM scored 7.73% and the blend scored 8.89%.

The larger search completed 420 comparisons. The best new candidate without asking price averaged 7.2857% median error across three historical windows, compared with 7.2895% for the selected blend. The candidate had worse mean error and lost two of the three periods. Existing CatBoost remained best for the other track. We therefore kept both selected models. The small difference does not justify replacing a model after many comparisons.

These windows were reused for development. They are not untouched test data. The official months were also seen during previous work. The report states these limitations.

The estimated intervals target 90% coverage. With monthly refits they achieved 90.00% without asking price and 87.34% with it. Without refitting for four months they achieved 89.81% and 87.30%. We report this shortfall instead of claiming guaranteed coverage.

The archived experiment outputs are evidence of what was tried, including approaches we did not select. They should not be overwritten to make the experiment history look cleaner.

## Focused checks on October 28

The saved results are in research_v4/focused and research_v4/recent_blend. Both runs completed successfully. Neither trained new models or changed submissions.

A blend with 75% LightGBM and 25% spline improved all three older evaluation periods. Average window MdAPE fell from 8.076% to 7.380%, and average MAPE from 10.313% to 9.725%. The expensive home correction added almost no overall benefit and worsened errors for actual prices above $2 million in two periods relative to the uncorrected proposed blend.

The more recent check was mixed. November 2024 to February 2025 MdAPE improved from 7.020% to 6.883%, but July to October 2025 worsened from 7.270% to 7.331%. Across those two windows, average MdAPE improved only 0.038 percentage points and average MAPE worsened 0.043 points. The March 2024 overlap is not an additional independent result. We kept the original equal weight blend.

A conservative property based interval method raised coverage from 87.34% to 92.14% on reused February to May monthly validation. It widened intervals and improved the normalised interval score from 31.85 to 30.40. It also recalibrates as outcomes become available, unlike four future months without new sale prices. Historical frozen checks gave about 92.75% coverage, compared with 90.97% for pooled earlier residuals. This remains exploratory and was not adopted as a final interval method.

No new official point prediction score was generated. The report continues to use 7.44% and 4.02% from the selected models. These checks strengthen the experiment record, not a claim of fresh independent validation.

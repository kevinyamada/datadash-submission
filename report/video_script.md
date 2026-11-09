# Video script

Companion script for the four page report. Recorded presentation: https://www.youtube.com/watch?v=Sgtuz8VY2iQ (4 minutes 42 seconds). Public code: https://github.com/kevinyamada/datadash-submission. Presenters: Kevin Ryuichi Yamada and Luka Tosic.

## Problem and data

When someone buys a home, they want to know what it will actually sell for. Our project answers this in two settings: before an asking price is available, and after the home is listed. We need separate models because asking price provides a strong clue that would be unavailable when valuing a home before listing.

We used 150,000 synthetic sales from January 2021 to May 2026. The competition also provides two separate sets of homes to predict, with their sale prices hidden.

Our data checks found that many floor areas were given as ranges, so we used their midpoints. We excluded external valuation estimates because they are missing from both prediction files and often dated after the sale.

## Models

Inputs include property type and style, bedrooms, bathrooms, parking, garage, floor area, lot dimensions, neighbourhood and coordinates, listing date features, and description features.

Without asking price, we combine LightGBM with spline ridge regression. LightGBM learns interactions between property features. The spline model fits smoother relationships and also uses description text.

Both predict log sale price. We average their predictions on that scale, with equal weight, and convert the result back to dollars.

With asking price, we use CatBoost to predict the adjustment between asking price and the eventual sale price.

The log scale makes proportional differences more comparable, so expensive homes have less influence than when using squared dollar errors. We also control model complexity through hyperparameters. Limits on tree size and L2 penalties help reduce overfitting. Ridge regression shrinks the spline model’s coefficients. These controls encourage patterns that generalize, rather than fitting every detail of the training data.

Asking price is excluded from the first task. Neither track uses days on market or sale date as a prediction feature.

## Testing and selection

We train on earlier sales and predict later months. Learned preprocessing uses only the training portion.

We compared model configurations and combinations across historical periods. A larger search involved 420 fits, but did not find a convincing replacement for our selected models.

The report also shows a supplementary weighted score combining prediction errors with runtime. In that comparison, our two selected blend components rank first and second for the task without asking price, while CatBoost ranks second for the task with it.

This was an additional comparison after our original selection. It does not independently prove that the blend is best.

## Results

Across December 2025 to May 2026, median absolute percentage error was 7.44 percent without asking price and 4.02 percent with it. This means half the absolute percentage errors were at or below those values.

Our previous models scored 9.55 and 4.11 percent, so the larger improvement was in the harder task without asking price.

We also trained once and predicted four months without updating. Errors were 7.64 and 4.14 percent. These are historical validation results, not hidden competition scores. The periods were reused during development.

We checked the cheapest and most expensive ten percent of validation homes, with 1,007 homes in each group. Without asking price, median error was 8.33 percent for the cheapest group and 12.96 percent for the most expensive. With asking price, these fell to 3.41 and 5.01 percent.

Property types matter too. Among the most expensive homes, the 775 detached properties had 12.52 percent median error without asking price. The 45 condo apartments had 32.48 percent. That small group needs caution. Actual prices define these groups only for analysing past errors, never for making predictions.

The development search took much longer than using the final models. On our computer with four CPU threads, training took about two minutes for the blend and six minutes for CatBoost. Each predicted roughly four thousand homes in under two seconds, excluding data loading.

## Uncertainty and recommendation

We estimated price ranges using earlier prediction errors. Ranges targeting 90 percent coverage covered about 90 percent of later sales without asking price, but only 87 percent with it. We report that shortfall rather than claiming a guarantee.

Our recommendation is to use these estimates to compare homes and plan a budget, alongside a range. Expensive or unusual properties deserve closer review.

All the data are synthetic. Real use would require suitable real data and further validation. Our code and results are available so the work can be checked and reproduced.

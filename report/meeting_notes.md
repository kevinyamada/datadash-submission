# Meeting guide

## The short explanation

I explored the supplied data, built and compared models, checked how they behave on future sales, and prepared the predictions, graphs and draft report. The project estimates a home's sale price in two settings: without an asking price and with one.

## Data and preparation

There is one Hugging Face collection with three files: 150,000 labelled sales for training, 3,974 homes without asking prices to predict, and 4,064 homes with asking prices to predict. Everything is synthetic.

I checked prices, missing values, property types and changes over time. I used floor area range midpoints and features from property details, location and listing descriptions. I excluded unavailable external valuations and information from after a sale. Asking price is only used in the second task.

## Models and training

Without asking price, we combine LightGBM and spline ridge regression equally on the log scale. LightGBM learns detailed interactions. Splines learn smoother relationships and include description text. Training on log price reduces the influence of very large dollar amounts. Predictions are converted back to dollars.

With asking price, CatBoost learns log sale price divided by asking price, then converts that adjustment into dollars. Asking price is already very informative.

## Testing and results

Train on earlier sales and predict later months. Repeat from December 2025 through May 2026. Median percentage errors are 7.44% without asking price and 4.02% with it. Half the percentage errors fall below the median. This is not classification accuracy and not the hidden competition score.

A separate four month forecast without updating produced 7.64% and 4.14% median errors. Validation periods were reused during development, which we disclose.

## Why these models

We compared model families, settings, features and blends. The large search ran 420 fits across seven families. Repeated training made it slow, especially CatBoost. It found no convincing replacement. Later checks reused predictions to test blend changes, price corrections and interval methods. Those did not justify changing the final models.

## Limits and next steps

Expensive homes are harder to value. Intervals are not guaranteed, and the method with asking price missed its 90% coverage target. Synthetic data results do not establish performance on real Toronto homes.

All 34 software tests passed, both submission files passed their checks, and the PDF is four pages. The code and results are on the model research branch of Kevin's repository. Remaining work is reviewing the report, adding full team names, recording the video, adding its accessible link and giving organisers repository access. No further experiments are planned now.

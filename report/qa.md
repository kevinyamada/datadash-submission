# Questions we should be able to answer

## What does 7.44% mean?

It is the median absolute percentage error over the official validation homes. Half the errors are at or below that value. It is not a claim that every prediction is within 7.44%, and it should not be converted into a classification accuracy score.

## Why two models?

Knowing the asking price changes the information available. The model without it must estimate value from the home's characteristics. The model with it learns an adjustment to an already informative price.

## Why a blend?

The tree and spline models make different errors. Their equal weight geometric mean performed well in the original selection periods. The blend is not always best, and we disclose the periods where the spline component struggled.

## Why not choose the best number from the large search?

Its advantage was only 0.0038 percentage points in average median error, it had worse mean error and it lost two periods. Searching more candidates makes tiny improvements easier to find by chance.

## Did you use future sales?

Each training set contains only sales before its cutoff. Learned preprocessing uses that training set only. The starter evaluates by sale month, however, and some homes were listed before the cutoff. We disclose this and separately check homes listed after the cutoff.

## Are the intervals guaranteed?

No. They use earlier log prediction errors. We measure later coverage and report it. Market changes and shared monthly shocks can make future coverage different.

## How would you use this in practice?

As a budget and comparison aid, with a range and a clear route to closer review. It would need suitable real data, validation and monitoring before real deployment.

## What still needs to be done before submission?

Confirm team details, record the video, add its link, check the PDF length and give organisers repository access. The private repository is not automatically accessible to judges.

## Did the final improvement ideas work?

Changing the blend helped older periods but was mixed in more recent periods. The price correction gave little benefit and sometimes made expensive home errors worse. Alternative intervals increased coverage with wider ranges, but the monthly update setting does not match four future months without outcomes. None was adopted.

## Why did the large search take so long?

It trained 420 model fits across seven families, both tasks and several periods. CatBoost consumed about 58% of recorded fitting time. The user observed about 20 hours elapsed, while model timers totalled about 10.6 hours, so the timers do not explain all elapsed time. The search was too broad for the improvement it produced. Later checks reused saved predictions instead.

## Do the software tests pass?

Yes. Kevin reported 34 passed in 10.69 seconds on October 28. The old full settings pipeline test also passed, taking almost 18 minutes. The revised test uses small training settings without changing the real models or submissions. Random test data errors are not competition validation scores.

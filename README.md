# TikTok Video Engagement Prediction

Predicts a TikTok video's Day-30 cumulative view count from its first 5 days of engagement data. Built for the WeCloudData DS Bootcamp in-class Kaggle competition (RMSE-scored, individual submission).

## Problem
Given video metadata, creator statistics, and engagement metrics for Days 0–5 only, predict `target_day30_views` for a hidden test set. View counts follow a heavy power-law distribution — a handful of viral videos dominate — so RMSE punishes large over-predictions severely.

## Approach
Instead of predicting the raw view count directly, the model predicts the **growth multiple** (`Day-30 views ÷ Day-5 views`) and multiplies it back by the known Day-5 view count. This reframing handles the power-law tail much better than predicting raw counts.

**Features (Days 0–5 only, no data leakage):**
- Cumulative plays, likes, comments, shares, collects, downloads per day
- Daily increments and growth ratios (e.g. `views_day5 / views_day1`)
- Engagement rates (likes/plays, shares/plays, etc.)
- Creator statistics (followers, total favorites, video count) taken **only at the video's post date and at Day 5** — never later, to avoid leakage
- Video metadata: duration, topic, hour/day posted

**Model — a 3-way blend:**
1. **Baseline** — a single constant growth ratio (`sum(y)/sum(views_day5)`) applied to every video
2. **Ridge Regression** — predicts the growth multiple, weighted by video size (`views_day5`) so large videos (which dominate RMSE) get more influence, with the multiple clipped to `[1.0, 4.0]` to prevent wild over-predictions
3. **Gradient Boosting** (`HistGradientBoostingRegressor`) — predicts `log(growth multiple)`, then calibrated with a scaling factor that minimizes RMSE directly, and shrunk 50/50 toward the baseline for stability

Final prediction = average of all three models.

## Validation
Repeated 5-fold cross-validation (2 repeats) was used instead of a single train/test split, since a small number of viral videos in any one split can swing RMSE dramatically.

| Model | CV RMSE |
|---|---|
| Baseline | ~72,800 |
| Ridge | ~61,900 |
| Gradient Boosting | ~65,200 |
| **Blend (final)** | **~63,200** |

The blend was chosen over the lowest-RMSE individual model because it showed the lowest variance across folds — more reliable on unseen data than a model that scored best on this particular sample.

## Files
- `tiktok_engagement_solution.ipynb` — full, reproducible notebook (Run All reproduces `submission.csv`)
- `submission.csv` — final predictions
- `README.md` — this file

## How to run
1. Open `tiktok_engagement_solution.ipynb` in Google Colab
2. Upload `train_videos.csv`, `test_videos.csv`, `creators_daily.csv`, `engagement_daily.csv`, `sample_submission.csv`
3. Runtime → Run all
4. `submission.csv` is generated and downloaded automatically

## Author
[Your name here]

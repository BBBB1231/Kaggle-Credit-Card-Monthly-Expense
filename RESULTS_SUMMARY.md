# Model Results Summary

## Objective
Build a predictive model to estimate monthly credit card spending using feature transformation, imputation methods, and LASSO regression.

## Data Overview
- **Training Data**: 40,000 customers with 23 features
- **Test Data**: 10,000 customers with 22 features (no target)
- **Target Variable**: monthly_spend (continuous)
- **Evaluation Metric**: RMSE (Root Mean Squared Error)

## Missing Values
### Analysis Data
- online_shopping_freq: 2,008 missing (5.02%)
- education_level: 1,199 missing (2.99%)
- utility_payment_count: 798 missing (1.99%)

### Scoring Data
- online_shopping_freq: 492 missing (4.92%)
- education_level: 301 missing (3.01%)
- utility_payment_count: 202 missing (2.02%)

## Feature Engineering

### Categorical Features (Label Encoded)
- gender
- marital_status
- education_level
- region
- employment_status
- card_type

### Numerical Features (15 features)
- age, owns_home, has_auto_loan
- annual_income, credit_score, credit_limit, tenure
- num_transactions, avg_transaction_value, online_shopping_freq
- reward_points_balance, travel_frequency, utility_payment_count
- num_children, num_credit_cards

### Feature Transformation
- **StandardScaler**: Applied to all features (both categorical and numerical after encoding)
- **Normalization**: Features scaled to have mean=0 and std=1

## Imputation Methods Comparison

All three imputation methods were tested with LASSO regression (alpha=1.0):

| Method | Validation RMSE | Validation R² | Submission File |
|--------|----------------|---------------|-----------------|
| **KNN (k=5)** | **267.42** | **0.7522** | submission_knn.csv |
| Mean | 267.42 | 0.7521 | submission_mean.csv |
| Median | 267.42 | 0.7521 | submission_median.csv |

## Best Model: KNN Imputation (k=5)

### Model Details
- **Algorithm**: LASSO Regression
- **Alpha**: 1.0
- **Imputation**: K-Nearest Neighbors (k=5)
- **Feature Selection**: 20 out of 21 features retained (LASSO automatically removed 1 feature)

### Performance Metrics
- **Training RMSE**: 266.89
- **Training R²**: 0.7533
- **Validation RMSE**: 267.42
- **Validation R²**: 0.7522

## Key Findings

1. **Similar Performance**: All three imputation methods (mean, median, kNN) produced very similar results, with differences in RMSE < 0.01

2. **Best Method**: KNN imputation (k=5) slightly outperformed mean and median, with the lowest validation RMSE

3. **Feature Importance**: LASSO retained 20 out of 21 features, indicating most features contribute to predicting monthly spend

4. **Model Stability**: Low gap between training and validation RMSE (~0.5) indicates good generalization

## Files Generated

1. **model_pipeline.py**: Complete pipeline implementation
2. **submission_knn.csv**: Best submission file (KNN imputation)
3. **submission_mean.csv**: Mean imputation submission
4. **submission_median.csv**: Median imputation submission

## Recommendations

- Use **submission_knn.csv** for Kaggle submission
- The model achieves R² = 0.75, explaining 75% of variance in monthly spend
- Consider ensemble methods or hyperparameter tuning for further improvements
- Feature engineering (polynomial features, interactions) could potentially improve performance

## Next Steps

1. Submit **submission_knn.csv** to Kaggle
2. Consider trying different LASSO alpha values for regularization tuning
3. Explore other algorithms (Ridge, ElasticNet, Random Forest, XGBoost)
4. Create feature interactions (e.g., income × credit_score)
5. Perform hyperparameter optimization using GridSearchCV or RandomizedSearchCV

"""
FAST LINEAR MODELS - 5-Fold CV + Grid Search Only (No PCA, No Random Search)
Optimized for speed while maintaining accuracy
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer

print("=" * 80)
print("FAST LINEAR MODELS - 5-FOLD CV + GRID SEARCH (NO PCA)")
print("=" * 80)

# Load data
print("\n[1/6] Loading data...")
train = pd.read_csv('analysis_data.csv')
test = pd.read_csv('scoring_data.csv')

y = train['monthly_spend'].values
test_ids = test['customer_id'].values

train = train.drop(['customer_id', 'monthly_spend'], axis=1)
test = test.drop(['customer_id'], axis=1)

print(f"Training: {train.shape}, Test: {test.shape}")

print("\n[2/6] Feature Engineering...")

def engineer_features(df):
    df = df.copy()

    # Core features
    df['total_trans_value'] = df['num_transactions'] * df['avg_transaction_value']
    df['income_credit_product'] = df['annual_income'] * df['credit_limit'] / 1e9
    df['income_credit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1)
    df['credit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)
    df['income_per_card'] = df['annual_income'] / (df['num_credit_cards'] + 1)

    # Rewards
    df['rewards_per_trans'] = df['reward_points_balance'] / (df['num_transactions'] + 1)
    df['rewards_per_income'] = df['reward_points_balance'] / (df['annual_income'] + 1)

    # Activity
    df['online_ratio'] = df['online_shopping_freq'] / (df['num_transactions'] + 1)
    df['utility_ratio'] = df['utility_payment_count'] / (df['num_transactions'] + 1)
    df['travel_ratio'] = df['travel_frequency'] / (df['num_transactions'] + 1)

    # Demographics
    df['age_income'] = df['age'] * df['annual_income'] / 1e4
    df['age_credit_score'] = df['age'] * df['credit_score'] / 100

    # Spending power
    df['spending_power'] = (df['annual_income'] * df['credit_score']) / 1e5
    df['monthly_income'] = df['annual_income'] / 12

    # Log transforms
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_credit_limit'] = np.log1p(df['credit_limit'])
    df['log_rewards'] = np.log1p(df['reward_points_balance'])

    # Polynomials
    df['credit_limit_sq'] = df['credit_limit'] ** 2 / 1e8
    df['income_sq'] = df['annual_income'] ** 2 / 1e9
    df['num_trans_sq'] = df['num_transactions'] ** 2

    return df

train = engineer_features(train)
test = engineer_features(test)

print(f"Features: {train.shape[1]}")

print("\n[3/6] Preprocessing...")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
            test[col].fillna('missing', inplace=True)
        else:
            median_val = train[col].median()
            train[col].fillna(median_val, inplace=True)
            test[col].fillna(median_val, inplace=True)

for col in train.columns:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

scaler = RobustScaler()
X_scaled = scaler.fit_transform(train.values)
X_test_scaled = scaler.transform(test.values)

# RFE Feature Selection
print("\n[4/6] Feature Selection...")
ridge_selector = Ridge(alpha=10.0)
rfe = RFE(estimator=ridge_selector, n_features_to_select=40, step=3)
rfe.fit(X_scaled, y)
X_rfe = rfe.transform(X_scaled)
X_test_rfe = rfe.transform(X_test_scaled)
print(f"RFE selected: {X_rfe.shape[1]} features")

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))

print("\n[5/6] Grid Search for Optimal Hyperparameters...")
print("="*80)

results = []

# Ridge on RFE
print("Ridge on RFE features...")
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 50, 100, 200, 500]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=kf, scoring=rmse_scorer, n_jobs=-1)
ridge_grid.fit(X_rfe, y)
print(f"  Best alpha: {ridge_grid.best_params_['alpha']}")
print(f"  CV RMSE: {-ridge_grid.best_score_:.4f}")
results.append(('Ridge_RFE', ridge_grid.best_estimator_, -ridge_grid.best_score_, X_rfe, X_test_rfe))

# Ridge on all scaled
print("Ridge on all scaled features...")
ridge_grid2 = GridSearchCV(Ridge(), ridge_params, cv=kf, scoring=rmse_scorer, n_jobs=-1)
ridge_grid2.fit(X_scaled, y)
print(f"  Best alpha: {ridge_grid2.best_params_['alpha']}")
print(f"  CV RMSE: {-ridge_grid2.best_score_:.4f}")
results.append(('Ridge_Scaled', ridge_grid2.best_estimator_, -ridge_grid2.best_score_, X_scaled, X_test_scaled))

# LASSO on RFE
print("LASSO on RFE features...")
lasso_params = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=kf, scoring=rmse_scorer, n_jobs=-1)
lasso_grid.fit(X_rfe, y)
print(f"  Best alpha: {lasso_grid.best_params_['alpha']}")
print(f"  CV RMSE: {-lasso_grid.best_score_:.4f}")
results.append(('LASSO_RFE', lasso_grid.best_estimator_, -lasso_grid.best_score_, X_rfe, X_test_rfe))

# LASSO on all scaled
print("LASSO on all scaled features...")
lasso_grid2 = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=kf, scoring=rmse_scorer, n_jobs=-1)
lasso_grid2.fit(X_scaled, y)
print(f"  Best alpha: {lasso_grid2.best_params_['alpha']}")
print(f"  CV RMSE: {-lasso_grid2.best_score_:.4f}")
results.append(('LASSO_Scaled', lasso_grid2.best_estimator_, -lasso_grid2.best_score_, X_scaled, X_test_scaled))

# ElasticNet on RFE (Grid not Random)
print("ElasticNet on RFE features...")
elastic_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic_grid = GridSearchCV(ElasticNet(max_iter=10000), elastic_params, cv=kf, scoring=rmse_scorer, n_jobs=-1)
elastic_grid.fit(X_rfe, y)
print(f"  Best alpha: {elastic_grid.best_params_['alpha']}")
print(f"  Best l1_ratio: {elastic_grid.best_params_['l1_ratio']}")
print(f"  CV RMSE: {-elastic_grid.best_score_:.4f}")
results.append(('ElasticNet_RFE', elastic_grid.best_estimator_, -elastic_grid.best_score_, X_rfe, X_test_rfe))

# Sort by RMSE
results.sort(key=lambda x: x[2])

print("\n" + "="*80)
print("[6/6] Model Ranking and Predictions...")
print("="*80)

print("\nAll Models Ranked:")
for i, (name, model, cv_rmse, X_feat, X_test_feat) in enumerate(results, 1):
    print(f"{i}. {name:20s} - CV RMSE: {cv_rmse:.4f}")

# Ensemble top 3
top_3 = results[:3]
weights = []
predictions = []

print(f"\nEnsemble of top 3 models:")
for name, model, cv_rmse, X_feat, X_test_feat in top_3:
    weight = 1.0 / cv_rmse
    weights.append(weight)
    pred = model.predict(X_test_feat)
    predictions.append(pred)

weights = np.array(weights) / sum(weights)
for (name, _, cv_rmse, _, _), w in zip(top_3, weights):
    print(f"  {name}: {w:.4f}")

final_predictions = np.average(predictions, axis=0, weights=weights)

submission = pd.DataFrame({
    'customer_id': test_ids,
    'monthly_spend': final_predictions
})

submission.to_csv('submission_linear_optimized.csv', index=False)

print("\n" + "="*80)
print("LINEAR MODEL COMPLETE!")
print("="*80)
print(f"Best Model: {results[0][0]}")
print(f"Best CV RMSE: {results[0][2]:.4f}")
print(f"\nSubmission: submission_linear_optimized.csv")
print(f"Mean: ${submission['monthly_spend'].mean():.2f}")
print(f"Median: ${submission['monthly_spend'].median():.2f}")
print("\nFirst 10:")
print(submission.head(10))
print("="*80)

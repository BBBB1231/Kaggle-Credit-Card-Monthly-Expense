"""
ADVANCED Credit Card Spending Prediction
Goal: Break 220 RMSE
Advanced techniques:
- Target transformation
- Hyperparameter tuning
- Stacking ensemble
- Feature selection
- More aggressive feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED MODEL - TARGET: < 220 RMSE")
print("=" * 80)

# Load data
print("\n[1/8] Loading data...")
train = pd.read_csv('analysis_data.csv')
test = pd.read_csv('scoring_data.csv')

y = train['monthly_spend'].values
test_ids = test['customer_id'].values

train = train.drop(['customer_id', 'monthly_spend'], axis=1)
test = test.drop(['customer_id'], axis=1)

print(f"Train: {train.shape}, Test: {test.shape}")

print("\n[2/8] AGGRESSIVE Feature Engineering...")

def create_advanced_features(df):
    """Comprehensive feature engineering for maximum performance"""
    df = df.copy()

    # === TRANSACTION FEATURES ===
    df['total_trans_value'] = df['num_transactions'] * df['avg_transaction_value']
    df['trans_value_squared'] = df['total_trans_value'] ** 2 / 100000
    df['trans_per_tenure'] = df['num_transactions'] / (df['tenure'] + 1)
    df['avg_trans_value_log'] = np.log1p(df['avg_transaction_value'])
    df['trans_momentum'] = df['num_transactions'] * df['tenure']

    # === INCOME & CREDIT FEATURES ===
    df['income_limit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1)
    df['income_limit_product'] = df['annual_income'] * df['credit_limit'] / 1000000
    df['limit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)
    df['income_per_card'] = df['annual_income'] / (df['num_credit_cards'] + 1)
    df['spending_power'] = df['annual_income'] * df['credit_score'] / 100000
    df['financial_health'] = (df['credit_score'] * df['annual_income']) / (df['credit_limit'] + 1)

    # === REWARD POINTS (super important - 0.67 corr!) ===
    df['rewards_per_trans'] = df['reward_points_balance'] / (df['num_transactions'] + 1)
    df['rewards_per_dollar'] = df['reward_points_balance'] / (df['annual_income'] + 1)
    df['rewards_efficiency'] = df['reward_points_balance'] / (df['total_trans_value'] + 1)
    df['rewards_squared'] = df['reward_points_balance'] ** 2 / 1000000

    # === ACTIVITY FEATURES ===
    df['total_activity'] = df['num_transactions'] + df['online_shopping_freq'].fillna(0) + df['travel_frequency'] + df['utility_payment_count'].fillna(0)
    df['online_ratio'] = df['online_shopping_freq'] / (df['num_transactions'] + 1)
    df['utility_ratio'] = df['utility_payment_count'] / (df['num_transactions'] + 1)
    df['travel_ratio'] = df['travel_frequency'] / (df['num_transactions'] + 1)
    df['shopping_travel_interaction'] = df['online_shopping_freq'] * df['travel_frequency']

    # === DEMOGRAPHIC FEATURES ===
    df['age_income'] = df['age'] * df['annual_income'] / 10000
    df['age_credit'] = df['age'] * df['credit_score'] / 100
    df['age_squared'] = df['age'] ** 2
    df['children_cost'] = df['num_children'] * df['annual_income'] / 10000
    df['age_tenure_interaction'] = df['age'] * df['tenure']

    # === POLYNOMIAL FEATURES for top predictors ===
    df['credit_limit_sq'] = df['credit_limit'] ** 2 / 100000
    df['credit_limit_cube'] = df['credit_limit'] ** 3 / 10000000000
    df['income_sq'] = df['annual_income'] ** 2 / 1000000000
    df['num_trans_sq'] = df['num_transactions'] ** 2
    df['rewards_sq'] = df['reward_points_balance'] ** 2 / 1000000

    # === LOG TRANSFORMATIONS ===
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_credit_limit'] = np.log1p(df['credit_limit'])
    df['log_rewards'] = np.log1p(df['reward_points_balance'])
    df['log_total_activity'] = np.log1p(df['total_activity'])

    # === COMPLEX INTERACTIONS ===
    df['income_trans_rewards'] = (df['annual_income'] * df['num_transactions'] * df['reward_points_balance']) / 1000000000
    df['credit_trans_interaction'] = (df['credit_limit'] * df['num_transactions']) / 10000
    df['age_income_credit'] = (df['age'] * df['annual_income'] * df['credit_score']) / 10000000

    # === RATIOS & RATES ===
    df['trans_per_year'] = df['num_transactions'] * 12  # assuming monthly data
    df['annual_spending_estimate'] = df['total_trans_value'] * 12
    df['credit_usage_rate'] = df['num_transactions'] / (df['credit_limit'] / 100 + 1)

    # === BINNED FEATURES (capture non-linearity) ===
    df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=[0,1,2,3,4])
    df['income_bin'] = pd.qcut(df['annual_income'], q=10, labels=False, duplicates='drop')
    df['credit_bin'] = pd.qcut(df['credit_limit'], q=10, labels=False, duplicates='drop')
    df['trans_bin'] = pd.qcut(df['num_transactions'], q=10, labels=False, duplicates='drop')

    return df

train = create_advanced_features(train)
test = create_advanced_features(test)

print(f"Features: {train.shape[1]} (up from 22)")

print("\n[3/8] Handling missing values...")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
            test[col].fillna('missing', inplace=True)
        else:
            median_val = train[col].median()
            train[col].fillna(median_val, inplace=True)
            test[col].fillna(median_val, inplace=True)

print("\n[4/8] Encoding categorical variables...")
for col in train.columns:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

X = train.values
X_test = test.values

print("\n[5/8] Target transformation (log1p to handle skewness)...")
y_log = np.log1p(y)
print(f"Original target skewness: {pd.Series(y).skew():.3f}")
print(f"Transformed target skewness: {pd.Series(y_log).skew():.3f}")

print("\n[6/8] Training optimized models with stacking...")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions for stacking
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_xgb = []
test_preds_lgb = []
test_preds_cat = []

print("\n" + "="*70)
print("TRAINING XGBoost with optimized hyperparameters...")
print("="*70)
xgb_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred = model.predict(X_val)
    oof_xgb[val_idx] = val_pred
    test_preds_xgb.append(model.predict(X_test))

    rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
    xgb_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

print(f"XGBoost CV RMSE: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")

print("\n" + "="*70)
print("TRAINING LightGBM with optimized hyperparameters...")
print("="*70)
lgb_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=15,
        reg_alpha=0.5,
        reg_lambda=1.5,
        num_leaves=50,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    val_pred = model.predict(X_val)
    oof_lgb[val_idx] = val_pred
    test_preds_lgb.append(model.predict(X_test))

    rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
    lgb_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

print(f"LightGBM CV RMSE: {np.mean(lgb_scores):.4f} (+/- {np.std(lgb_scores):.4f})")

print("\n" + "="*70)
print("TRAINING CatBoost with optimized hyperparameters...")
print("="*70)
cat_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]

    model = CatBoostRegressor(
        iterations=2000,
        depth=7,
        learning_rate=0.03,
        subsample=0.85,
        l2_leaf_reg=3,
        random_state=42 + fold,
        verbose=False,
        thread_count=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )

    val_pred = model.predict(X_val)
    oof_cat[val_idx] = val_pred
    test_preds_cat.append(model.predict(X_test))

    rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
    cat_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

print(f"CatBoost CV RMSE: {np.mean(cat_scores):.4f} (+/- {np.std(cat_scores):.4f})")

print("\n[7/8] Stacking models with Ridge meta-learner...")

# Prepare stacking features
stack_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
stack_test = np.column_stack([
    np.mean(test_preds_xgb, axis=0),
    np.mean(test_preds_lgb, axis=0),
    np.mean(test_preds_cat, axis=0)
])

# Train meta-learner
meta_model = Ridge(alpha=10.0)
meta_model.fit(stack_train, y_log)

# Final predictions
final_pred_log = meta_model.predict(stack_test)
final_pred = np.expm1(final_pred_log)

# Calculate stacking RMSE on OOF
oof_stack_pred_log = meta_model.predict(stack_train)
stack_rmse = np.sqrt(mean_squared_error(y, np.expm1(oof_stack_pred_log)))
print(f"\nStacked model OOF RMSE: {stack_rmse:.4f}")

print("\n[8/8] Creating submission...")
submission = pd.DataFrame({
    'customer_id': test_ids,
    'monthly_spend': final_pred
})

submission.to_csv('submission_advanced.csv', index=False)

print("\n" + "="*80)
print("ADVANCED MODEL COMPLETE!")
print("="*80)
print(f"Best CV RMSE: {min(np.mean(xgb_scores), np.mean(lgb_scores), np.mean(cat_scores)):.4f}")
print(f"Stacked RMSE: {stack_rmse:.4f}")
print(f"\nSubmission saved: submission_advanced.csv")
print(f"Mean prediction: ${submission['monthly_spend'].mean():.2f}")
print(f"Median prediction: ${submission['monthly_spend'].median():.2f}")
print("\nFirst 10 predictions:")
print(submission.head(10))
print("="*80)

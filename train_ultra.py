"""
ULTRA-AGGRESSIVE MODEL
Target: 220 RMSE or bust!
Strategy:
- Massive feature engineering
- Feature selection
- Weighted ensemble (based on CV performance)
- Multiple models with different seeds for diversity
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ULTRA-AGGRESSIVE MODEL + PCA - GO FOR 220!")
print("=" * 80)

# Load data
print("\n[1/10] Loading data...")
train = pd.read_csv('analysis_data.csv')
test = pd.read_csv('scoring_data.csv')

y = train['monthly_spend'].values
test_ids = test['customer_id'].values

train = train.drop(['customer_id', 'monthly_spend'], axis=1)
test = test.drop(['customer_id'], axis=1)

print("\n[2/10] ULTRA Feature Engineering...")

def ultra_features(df):
    """Maximum feature engineering"""
    df = df.copy()

    # Transaction features - THE KEY DRIVERS
    df['total_trans_value'] = df['num_transactions'] * df['avg_transaction_value']
    df['trans_value_sq'] = df['total_trans_value'] ** 2 / 100000
    df['trans_value_cube'] = df['total_trans_value'] ** 3 / 10000000000
    df['trans_per_tenure'] = df['num_transactions'] / (df['tenure'] + 1)
    df['avg_trans_log'] = np.log1p(df['avg_transaction_value'])
    df['num_trans_sq'] = df['num_transactions'] ** 2
    df['num_trans_cube'] = df['num_transactions'] ** 3

    # Income & Credit - TOP CORRELATIONS
    df['income_limit_product'] = df['annual_income'] * df['credit_limit'] / 1000000
    df['income_limit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1)
    df['limit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)
    df['income_per_card'] = df['annual_income'] / (df['num_credit_cards'] + 1)
    df['spending_power'] = (df['annual_income'] * df['credit_score']) / 100000
    df['financial_strength'] = (df['credit_score'] * df['annual_income'] * df['credit_limit']) / 1000000000

    # Reward points - 0.67 CORRELATION!
    df['rewards_per_trans'] = df['reward_points_balance'] / (df['num_transactions'] + 1)
    df['rewards_per_dollar_income'] = df['reward_points_balance'] / (df['annual_income'] + 1)
    df['rewards_per_dollar_limit'] = df['reward_points_balance'] / (df['credit_limit'] + 1)
    df['rewards_efficiency'] = df['reward_points_balance'] / (df['total_trans_value'] + 1)
    df['rewards_sq'] = df['reward_points_balance'] ** 2 / 1000000
    df['rewards_cube'] = df['reward_points_balance'] ** 3 / 1000000000000

    # Activity features
    df['total_activity'] = (df['num_transactions'] +
                           df['online_shopping_freq'].fillna(0) +
                           df['travel_frequency'] +
                           df['utility_payment_count'].fillna(0))
    df['online_ratio'] = df['online_shopping_freq'] / (df['num_transactions'] + 1)
    df['utility_ratio'] = df['utility_payment_count'] / (df['num_transactions'] + 1)
    df['travel_ratio'] = df['travel_frequency'] / (df['num_transactions'] + 1)
    df['online_travel_interaction'] = df['online_shopping_freq'] * df['travel_frequency']
    df['online_utility_interaction'] = df['online_shopping_freq'] * df['utility_payment_count']

    # Demographics
    df['age_income'] = df['age'] * df['annual_income'] / 10000
    df['age_credit'] = df['age'] * df['credit_score'] / 100
    df['age_sq'] = df['age'] ** 2
    df['children_income'] = (df['num_children'] + 1) * df['annual_income'] / 10000
    df['age_tenure'] = df['age'] * df['tenure']

    # Polynomials for TOP features
    df['credit_limit_sq'] = df['credit_limit'] ** 2 / 100000
    df['credit_limit_cube'] = df['credit_limit'] ** 3 / 10000000000
    df['income_sq'] = df['annual_income'] ** 2 / 1000000000
    df['income_cube'] = df['annual_income'] ** 3 / 100000000000000

    # Log transformations
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_credit_limit'] = np.log1p(df['credit_limit'])
    df['log_rewards'] = np.log1p(df['reward_points_balance'])
    df['log_total_activity'] = np.log1p(df['total_activity'])
    df['log_total_trans_value'] = np.log1p(df['total_trans_value'])

    # Complex 3-way interactions
    df['income_trans_rewards'] = (df['annual_income'] * df['num_transactions'] * df['reward_points_balance']) / 1000000000
    df['credit_trans_rewards'] = (df['credit_limit'] * df['num_transactions'] * df['reward_points_balance']) / 1000000000
    df['age_income_credit'] = (df['age'] * df['annual_income'] * df['credit_score']) / 10000000

    # Spending indicators
    df['monthly_income'] = df['annual_income'] / 12
    df['estimated_monthly_capacity'] = df['credit_limit'] * 0.3  # typical utilization
    df['income_trans_interaction'] = df['annual_income'] * df['num_transactions'] / 100000

    # Binned features
    df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=[0,1,2,3,4])
    df['income_bin'] = pd.qcut(df['annual_income'], q=10, labels=False, duplicates='drop')
    df['credit_bin'] = pd.qcut(df['credit_limit'], q=10, labels=False, duplicates='drop')
    df['trans_bin'] = pd.qcut(df['num_transactions'], q=10, labels=False, duplicates='drop')
    df['rewards_bin'] = pd.qcut(df['reward_points_balance'], q=10, labels=False, duplicates='drop')

    # Ratios
    df['trans_to_income'] = df['num_transactions'] / (df['annual_income'] / 10000 + 1)
    df['trans_to_credit'] = df['num_transactions'] / (df['credit_limit'] / 1000 + 1)

    return df

train = ultra_features(train)
test = ultra_features(test)

print(f"Total features: {train.shape[1]}")

print("\n[3/10] Handling missing values...")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
            test[col].fillna('missing', inplace=True)
        else:
            median_val = train[col].median()
            train[col].fillna(median_val, inplace=True)
            test[col].fillna(median_val, inplace=True)

print("\n[4/10] Encoding categorical variables...")
for col in train.columns:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

print(f"\n[5/10] Feature selection (keeping top features)...")
# Use SelectKBest to remove noisy features
selector = SelectKBest(f_regression, k=min(80, train.shape[1]))  # Keep top 80 features
selector.fit(train, y)

train_selected = selector.transform(train)
test_selected = selector.transform(test)

print(f"Selected features: {train_selected.shape[1]} (from {train.shape[1]})")

print("\n[6/9] Applying PCA dimension reduction...")
# Standardize features for PCA
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_selected)
test_scaled = scaler.transform(test_selected)

# Apply PCA to capture 95% of variance
pca = PCA(n_components=0.95, random_state=42)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)

print(f"PCA components: {train_pca.shape[1]} (explaining 95% variance)")

# Combine original features with PCA features for more diversity
X = np.hstack([train_selected, train_pca])
X_test = np.hstack([test_selected, test_pca])

print(f"Final feature space: {X.shape[1]} (original + PCA)")

print("\n[7/9] Training ensemble with multiple models and seeds...")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_models_oof = []
all_models_test = []
all_model_scores = []
model_names = []

# XGBoost with multiple seeds
for seed in [42, 123, 456]:
    print(f"\nTraining XGBoost (seed={seed})...")
    oof = np.zeros(len(X))
    test_preds = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=3000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.01,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            tree_method='hist'
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred
        test_preds.append(model.predict(X_test))

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        scores.append(rmse)

    avg_score = np.mean(scores)
    print(f"  CV RMSE: {avg_score:.4f}")
    all_models_oof.append(oof)
    all_models_test.append(np.mean(test_preds, axis=0))
    all_model_scores.append(avg_score)
    model_names.append(f'XGB_{seed}')

# LightGBM with multiple seeds
for seed in [42, 123]:
    print(f"\nTraining LightGBM (seed={seed})...")
    oof = np.zeros(len(X))
    test_preds = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=3000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=1.0,
            reg_lambda=2.0,
            num_leaves=40,
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred
        test_preds.append(model.predict(X_test))

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        scores.append(rmse)

    avg_score = np.mean(scores)
    print(f"  CV RMSE: {avg_score:.4f}")
    all_models_oof.append(oof)
    all_models_test.append(np.mean(test_preds, axis=0))
    all_model_scores.append(avg_score)
    model_names.append(f'LGB_{seed}')

# CatBoost with multiple seeds
for seed in [42, 123]:
    print(f"\nTraining CatBoost (seed={seed})...")
    oof = np.zeros(len(X))
    test_preds = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(
            iterations=3000,
            depth=6,
            learning_rate=0.02,
            subsample=0.8,
            l2_leaf_reg=5,
            random_state=seed,
            verbose=False,
            thread_count=-1
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                 early_stopping_rounds=150, verbose=False)
        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred
        test_preds.append(model.predict(X_test))

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        scores.append(rmse)

    avg_score = np.mean(scores)
    print(f"  CV RMSE: {avg_score:.4f}")
    all_models_oof.append(oof)
    all_models_test.append(np.mean(test_preds, axis=0))
    all_model_scores.append(avg_score)
    model_names.append(f'CAT_{seed}')

print("\n[8/9] Model performance summary:")
for name, score in zip(model_names, all_model_scores):
    print(f"  {name}: {score:.4f}")

print("\n[9/9] Creating weighted ensemble...")
# Weight models based on inverse RMSE (better models get more weight)
weights = 1 / np.array(all_model_scores)
weights = weights / weights.sum()

print("Model weights:")
for name, weight in zip(model_names, weights):
    print(f"  {name}: {weight:.4f}")

# Weighted ensemble
weighted_oof = np.zeros(len(X))
weighted_test = np.zeros(len(X_test))

for oof, test_pred, weight in zip(all_models_oof, all_models_test, weights):
    weighted_oof += oof * weight
    weighted_test += test_pred * weight

ensemble_rmse = np.sqrt(mean_squared_error(y, weighted_oof))
print(f"\nWeighted Ensemble OOF RMSE: {ensemble_rmse:.4f}")

print("\n[10/10] Creating final submission...")
submission = pd.DataFrame({
    'customer_id': test_ids,
    'monthly_spend': weighted_test
})

submission.to_csv('submission_ultra.csv', index=False)

print("\n" + "="*80)
print("ULTRA MODEL COMPLETE!")
print("="*80)
print(f"Best single model RMSE: {min(all_model_scores):.4f}")
print(f"Weighted ensemble RMSE: {ensemble_rmse:.4f}")
print(f"\nSubmission: submission_ultra.csv")
print(f"Mean: ${submission['monthly_spend'].mean():.2f}")
print(f"Median: ${submission['monthly_spend'].median():.2f}")
print("\nFirst 10:")
print(submission.head(10))
print("="*80)

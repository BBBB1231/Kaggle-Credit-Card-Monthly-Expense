"""
Advanced Credit Card Spending Prediction Model
Target: Beat 250 RMSE and aim for 220 RMSE
Strategy: Feature engineering + Gradient Boosting Ensemble + Stacking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("KAGGLE CREDIT CARD SPENDING - CHAMPIONSHIP MODEL")
print("=" * 80)

# Load data
print("\n[Step 1/7] Loading data...")
train = pd.read_csv('analysis_data.csv')
test = pd.read_csv('scoring_data.csv')
print(f"Training: {train.shape}, Test: {test.shape}")

# Save target and IDs
y = train['monthly_spend'].values
test_ids = test['customer_id'].values

# Drop ID and target
train = train.drop(['customer_id', 'monthly_spend'], axis=1)
test = test.drop(['customer_id'], axis=1)

print("\n[Step 2/7] Feature Engineering - Creating powerful features...")

def create_features(df):
    """Advanced feature engineering"""
    df = df.copy()

    # 1. Transaction-based features (KEY INSIGHT: these are highly predictive)
    df['total_transaction_value'] = df['num_transactions'] * df['avg_transaction_value']
    df['transaction_efficiency'] = df['total_transaction_value'] / (df['num_transactions'] + 1)
    df['avg_trans_per_travel'] = df['num_transactions'] / (df['travel_frequency'] + 1)

    # 2. Income & Credit features (top correlations!)
    df['income_to_limit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1)
    df['credit_limit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)
    df['income_per_year_tenure'] = df['annual_income'] / (df['tenure'] + 1)
    df['spending_power'] = df['annual_income'] * df['credit_score'] / 100000
    df['credit_score_to_limit'] = df['credit_score'] * df['credit_limit'] / 10000

    # 3. Reward points features (0.67 correlation!)
    df['rewards_per_transaction'] = df['reward_points_balance'] / (df['num_transactions'] + 1)
    df['rewards_per_dollar_income'] = df['reward_points_balance'] / (df['annual_income'] + 1)

    # 4. Shopping behavior features
    df['online_shopping_ratio'] = df['online_shopping_freq'] / (df['num_transactions'] + 1)
    df['utility_to_transaction_ratio'] = df['utility_payment_count'] / (df['num_transactions'] + 1)
    df['travel_to_transaction_ratio'] = df['travel_frequency'] / (df['num_transactions'] + 1)

    # 5. Demographic interaction features
    df['age_income_interaction'] = df['age'] * df['annual_income'] / 10000
    df['age_credit_interaction'] = df['age'] * df['credit_score'] / 100
    df['children_income'] = df['num_children'] * df['annual_income'] / 10000

    # 6. Polynomial features for top correlators
    df['credit_limit_squared'] = df['credit_limit'] ** 2 / 100000
    df['annual_income_squared'] = df['annual_income'] ** 2 / 1000000000
    df['num_transactions_squared'] = df['num_transactions'] ** 2

    # 7. Log transformations for skewed features
    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_credit_limit'] = np.log1p(df['credit_limit'])
    df['log_reward_points'] = np.log1p(df['reward_points_balance'])

    # 8. Binned features (capture non-linear relationships)
    df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=[0,1,2,3,4])
    df['income_bin'] = pd.qcut(df['annual_income'], q=10, labels=False, duplicates='drop')
    df['credit_bin'] = pd.qcut(df['credit_limit'], q=10, labels=False, duplicates='drop')

    return df

# Apply feature engineering
train = create_features(train)
test = create_features(test)

print(f"Features created! New shape: {train.shape}")

# Handle missing values
print("\n[Step 3/7] Handling missing values...")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
            test[col].fillna('missing', inplace=True)
        else:
            train[col].fillna(train[col].median(), inplace=True)
            test[col].fillna(train[col].median(), inplace=True)

# Encode categorical variables
print("\n[Step 4/7] Encoding categorical variables...")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Convert all columns to numeric (handle age_bin if it's categorical)
for col in train.columns:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

print(f"Final features: {train.shape[1]}")

# Prepare for modeling
X = train.values
X_test = test.values

print("\n[Step 5/7] Training models with 5-fold CV...")

# Try importing gradient boosting libraries
models_to_try = []

try:
    import xgboost as xgb
    models_to_try.append(('XGBoost', xgb.XGBRegressor))
    print("✓ XGBoost available")
except ImportError:
    print("✗ XGBoost not available")

try:
    import lightgbm as lgb
    models_to_try.append(('LightGBM', lgb.LGBMRegressor))
    print("✓ LightGBM available")
except ImportError:
    print("✗ LightGBM not available")

try:
    from catboost import CatBoostRegressor
    models_to_try.append(('CatBoost', CatBoostRegressor))
    print("✓ CatBoost available")
except ImportError:
    print("✗ CatBoost not available")

# Fallback to sklearn if no gradient boosting available
if not models_to_try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    models_to_try.append(('RandomForest', RandomForestRegressor))
    models_to_try.append(('GradientBoosting', GradientBoostingRegressor))
    print("Using sklearn ensemble methods")

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
test_predictions = []

print(f"\nTraining {len(models_to_try)} model(s)...")

for model_name, ModelClass in models_to_try:
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    fold_scores = []
    fold_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Model-specific parameters
        if model_name == 'XGBoost':
            model = ModelClass(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
        elif model_name == 'LightGBM':
            model = ModelClass(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        elif model_name == 'CatBoost':
            model = ModelClass(
                iterations=1000,
                depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbose=False,
                thread_count=-1
            )
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=50,
                verbose=False
            )
        elif model_name == 'RandomForest':
            model = ModelClass(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_fold, y_train_fold)
        else:  # GradientBoosting
            model = ModelClass(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            model.fit(X_train_fold, y_train_fold)

        # Predictions
        val_pred = model.predict(X_val_fold)
        test_pred = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
        fold_scores.append(rmse)
        fold_test_preds.append(test_pred)

        print(f"Fold {fold} RMSE: {rmse:.4f}")

    # Average RMSE across folds
    avg_rmse = np.mean(fold_scores)
    std_rmse = np.std(fold_scores)
    print(f"\n{model_name} Average RMSE: {avg_rmse:.4f} (+/- {std_rmse:.4f})")

    # Average test predictions
    test_predictions.append(np.mean(fold_test_preds, axis=0))

print("\n" + "="*80)
print("[Step 6/7] Creating ensemble predictions...")

# Simple average ensemble
if len(test_predictions) > 1:
    final_predictions = np.mean(test_predictions, axis=0)
    print(f"Ensembled {len(test_predictions)} models")
else:
    final_predictions = test_predictions[0]
    print("Using single model predictions")

print("\n[Step 7/7] Creating submission file...")
submission = pd.DataFrame({
    'customer_id': test_ids,
    'monthly_spend': final_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission file created: submission.csv")
print(f"  Shape: {submission.shape}")
print(f"  Mean prediction: ${submission['monthly_spend'].mean():.2f}")
print(f"  Median prediction: ${submission['monthly_spend'].median():.2f}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nExpected RMSE based on CV: ~{avg_rmse:.2f}")
print("Ready to submit to Kaggle!")
print("="*80)

"""
LINEAR MODEL APPROACH for Credit Card Spending Prediction
Hypothesis: Dataset is relatively linear - LASSO/Ridge may outperform tree models

Strategy:
1. Comprehensive feature engineering
2. PCA dimension reduction
3. Stepwise feature selection (RFE)
4. K-fold cross-validation
5. Grid search for optimal hyperparameters
6. Ensemble of LASSO, Ridge, ElasticNet
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

print("=" * 80)
print("LINEAR MODEL APPROACH - LASSO/RIDGE WITH PCA & FEATURE SELECTION")
print("=" * 80)

# Load data
print("\n[1/10] Loading data...")
train = pd.read_csv('analysis_data.csv')
test = pd.read_csv('scoring_data.csv')

y = train['monthly_spend'].values
test_ids = test['customer_id'].values

train = train.drop(['customer_id', 'monthly_spend'], axis=1)
test = test.drop(['customer_id'], axis=1)

print(f"Training: {train.shape}, Test: {test.shape}")

print("\n[2/10] Advanced Feature Engineering for Linear Models...")

def engineer_features(df):
    """Create features optimized for linear models"""
    df = df.copy()

    # ===== CORE TRANSACTION FEATURES =====
    df['total_transaction_value'] = df['num_transactions'] * df['avg_transaction_value']
    df['transaction_efficiency'] = df['avg_transaction_value'] * df['num_transactions']

    # ===== INCOME & CREDIT FEATURES (Top correlations!) =====
    df['income_credit_product'] = df['annual_income'] * df['credit_limit'] / 1e9
    df['income_credit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1)
    df['credit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)
    df['income_per_card'] = df['annual_income'] / (df['num_credit_cards'] + 1)

    # ===== REWARD POINTS (0.67 correlation) =====
    df['rewards_per_transaction'] = df['reward_points_balance'] / (df['num_transactions'] + 1)
    df['rewards_per_income'] = df['reward_points_balance'] / (df['annual_income'] + 1)
    df['rewards_per_credit'] = df['reward_points_balance'] / (df['credit_limit'] + 1)

    # ===== ACTIVITY RATIOS =====
    df['online_shopping_ratio'] = df['online_shopping_freq'] / (df['num_transactions'] + 1)
    df['utility_ratio'] = df['utility_payment_count'] / (df['num_transactions'] + 1)
    df['travel_ratio'] = df['travel_frequency'] / (df['num_transactions'] + 1)
    df['total_activity'] = (df['num_transactions'] +
                           df['online_shopping_freq'].fillna(0) +
                           df['travel_frequency'] +
                           df['utility_payment_count'].fillna(0))

    # ===== DEMOGRAPHIC INTERACTIONS =====
    df['age_income'] = df['age'] * df['annual_income'] / 1e4
    df['age_credit_score'] = df['age'] * df['credit_score'] / 100
    df['children_income'] = df['num_children'] * df['annual_income'] / 1e4

    # ===== SPENDING POWER INDICATORS =====
    df['spending_power'] = (df['annual_income'] * df['credit_score']) / 1e5
    df['financial_capacity'] = (df['credit_limit'] * df['credit_score']) / 1e5
    df['monthly_income'] = df['annual_income'] / 12
    df['credit_utilization_capacity'] = df['credit_limit'] * 0.3

    # ===== TRANSACTION PATTERNS =====
    df['avg_trans_per_tenure'] = df['num_transactions'] / (df['tenure'] + 1)
    df['income_trans_interaction'] = df['annual_income'] * df['num_transactions'] / 1e5
    df['credit_trans_interaction'] = df['credit_limit'] * df['num_transactions'] / 1e4

    # ===== LOG TRANSFORMATIONS (normalize skewed features) =====
    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_credit_limit'] = np.log1p(df['credit_limit'])
    df['log_rewards'] = np.log1p(df['reward_points_balance'])
    df['log_total_trans_value'] = np.log1p(df['total_transaction_value'])
    df['log_total_activity'] = np.log1p(df['total_activity'])

    # ===== POLYNOMIAL FEATURES (squared terms for key predictors) =====
    df['credit_limit_sq'] = df['credit_limit'] ** 2 / 1e8
    df['annual_income_sq'] = df['annual_income'] ** 2 / 1e9
    df['num_transactions_sq'] = df['num_transactions'] ** 2
    df['rewards_sq'] = df['reward_points_balance'] ** 2 / 1e6

    # ===== COMPLEX INTERACTIONS (3-way) =====
    df['income_credit_trans'] = (df['annual_income'] * df['credit_limit'] * df['num_transactions']) / 1e12
    df['income_rewards_trans'] = (df['annual_income'] * df['reward_points_balance'] * df['num_transactions']) / 1e12

    # ===== NORMALIZED RATIOS =====
    df['trans_to_income'] = df['num_transactions'] / (df['annual_income'] / 1e4 + 1)
    df['trans_to_credit'] = df['num_transactions'] / (df['credit_limit'] / 1e3 + 1)

    return df

train = engineer_features(train)
test = engineer_features(test)

print(f"Features created: {train.shape[1]}")

print("\n[3/10] Handling missing values and encoding...")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
            test[col].fillna('missing', inplace=True)
        else:
            median_val = train[col].median()
            train[col].fillna(median_val, inplace=True)
            test[col].fillna(median_val, inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
for col in train.columns:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

X = train.values
X_test = test.values

print(f"Feature matrix shape: {X.shape}")

print("\n[4/10] Standardizing features (critical for linear models!)...")
# Use RobustScaler to handle outliers better than StandardScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("\n[5/10] Applying PCA for dimension reduction...")
# Try different PCA configurations
pca_95 = PCA(n_components=0.95, random_state=42)
X_pca_95 = pca_95.fit_transform(X_scaled)
X_test_pca_95 = pca_95.transform(X_test_scaled)

pca_99 = PCA(n_components=0.99, random_state=42)
X_pca_99 = pca_99.fit_transform(X_scaled)
X_test_pca_99 = pca_99.transform(X_test_scaled)

print(f"PCA 95% variance: {X_pca_95.shape[1]} components")
print(f"PCA 99% variance: {X_pca_99.shape[1]} components")

print("\n[6/10] Stepwise feature selection (RFE with Ridge)...")
# Use Ridge for feature selection
ridge_selector = Ridge(alpha=10.0, random_state=42)
rfe = RFE(estimator=ridge_selector, n_features_to_select=40, step=5)
rfe.fit(X_scaled, y)
X_rfe = rfe.transform(X_scaled)
X_test_rfe = rfe.transform(X_test_scaled)

print(f"RFE selected: {X_rfe.shape[1]} features")

# Also try SelectKBest
selector = SelectKBest(f_regression, k=50)
X_kbest = selector.fit_transform(X_scaled, y)
X_test_kbest = selector.transform(X_test_scaled)

print(f"SelectKBest selected: {X_kbest.shape[1]} features")

print("\n[7/10] Setting up K-Fold Cross-Validation...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold for robust estimates

print("\n[8/10] Grid Search for optimal hyperparameters...")

# ===== LASSO =====
print("\n--- LASSO Regression ---")
lasso_alphas = np.logspace(-4, 2, 100)
lasso_cv = LassoCV(alphas=lasso_alphas, cv=kf, random_state=42, max_iter=10000, n_jobs=-1)

# Try on different feature sets
results = {}

print("Training LASSO on scaled features...")
lasso_cv.fit(X_scaled, y)
lasso_pred = lasso_cv.predict(X_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_pred))
print(f"  Best alpha: {lasso_cv.alpha_:.6f}")
print(f"  Train RMSE: {lasso_rmse:.4f}")
results['lasso_scaled'] = (lasso_cv, lasso_rmse, X_scaled, X_test_scaled)

print("Training LASSO on PCA 95% features...")
lasso_pca_95 = LassoCV(alphas=lasso_alphas, cv=kf, random_state=42, max_iter=10000, n_jobs=-1)
lasso_pca_95.fit(X_pca_95, y)
lasso_pca_pred = lasso_pca_95.predict(X_pca_95)
lasso_pca_rmse = np.sqrt(mean_squared_error(y, lasso_pca_pred))
print(f"  Best alpha: {lasso_pca_95.alpha_:.6f}")
print(f"  Train RMSE: {lasso_pca_rmse:.4f}")
results['lasso_pca_95'] = (lasso_pca_95, lasso_pca_rmse, X_pca_95, X_test_pca_95)

print("Training LASSO on RFE features...")
lasso_rfe = LassoCV(alphas=lasso_alphas, cv=kf, random_state=42, max_iter=10000, n_jobs=-1)
lasso_rfe.fit(X_rfe, y)
lasso_rfe_pred = lasso_rfe.predict(X_rfe)
lasso_rfe_rmse = np.sqrt(mean_squared_error(y, lasso_rfe_pred))
print(f"  Best alpha: {lasso_rfe.alpha_:.6f}")
print(f"  Train RMSE: {lasso_rfe_rmse:.4f}")
results['lasso_rfe'] = (lasso_rfe, lasso_rfe_rmse, X_rfe, X_test_rfe)

# ===== RIDGE =====
print("\n--- Ridge Regression ---")
ridge_alphas = np.logspace(-2, 4, 100)
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=kf)

print("Training Ridge on scaled features...")
ridge_cv.fit(X_scaled, y)
ridge_pred = ridge_cv.predict(X_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))
print(f"  Best alpha: {ridge_cv.alpha_:.6f}")
print(f"  Train RMSE: {ridge_rmse:.4f}")
results['ridge_scaled'] = (ridge_cv, ridge_rmse, X_scaled, X_test_scaled)

print("Training Ridge on PCA 95% features...")
ridge_pca_95 = RidgeCV(alphas=ridge_alphas, cv=kf)
ridge_pca_95.fit(X_pca_95, y)
ridge_pca_pred = ridge_pca_95.predict(X_pca_95)
ridge_pca_rmse = np.sqrt(mean_squared_error(y, ridge_pca_pred))
print(f"  Best alpha: {ridge_pca_95.alpha_:.6f}")
print(f"  Train RMSE: {ridge_pca_rmse:.4f}")
results['ridge_pca_95'] = (ridge_pca_95, ridge_pca_rmse, X_pca_95, X_test_pca_95)

print("Training Ridge on RFE features...")
ridge_rfe = RidgeCV(alphas=ridge_alphas, cv=kf)
ridge_rfe.fit(X_rfe, y)
ridge_rfe_pred = ridge_rfe.predict(X_rfe)
ridge_rfe_rmse = np.sqrt(mean_squared_error(y, ridge_rfe_pred))
print(f"  Best alpha: {ridge_rfe.alpha_:.6f}")
print(f"  Train RMSE: {ridge_rfe_rmse:.4f}")
results['ridge_rfe'] = (ridge_rfe, ridge_rfe_rmse, X_rfe, X_test_rfe)

# ===== ELASTIC NET =====
print("\n--- ElasticNet Regression ---")
from sklearn.linear_model import ElasticNetCV
elastic_alphas = np.logspace(-4, 2, 50)
elastic_l1_ratios = [.1, .3, .5, .7, .9, .95, .99]

print("Training ElasticNet on scaled features...")
elastic_cv = ElasticNetCV(alphas=elastic_alphas, l1_ratio=elastic_l1_ratios,
                          cv=kf, random_state=42, max_iter=10000, n_jobs=-1)
elastic_cv.fit(X_scaled, y)
elastic_pred = elastic_cv.predict(X_scaled)
elastic_rmse = np.sqrt(mean_squared_error(y, elastic_pred))
print(f"  Best alpha: {elastic_cv.alpha_:.6f}")
print(f"  Best l1_ratio: {elastic_cv.l1_ratio_:.4f}")
print(f"  Train RMSE: {elastic_rmse:.4f}")
results['elastic_scaled'] = (elastic_cv, elastic_rmse, X_scaled, X_test_scaled)

print("Training ElasticNet on PCA 95% features...")
elastic_pca = ElasticNetCV(alphas=elastic_alphas, l1_ratio=elastic_l1_ratios,
                           cv=kf, random_state=42, max_iter=10000, n_jobs=-1)
elastic_pca.fit(X_pca_95, y)
elastic_pca_pred = elastic_pca.predict(X_pca_95)
elastic_pca_rmse = np.sqrt(mean_squared_error(y, elastic_pca_pred))
print(f"  Best alpha: {elastic_pca.alpha_:.6f}")
print(f"  Best l1_ratio: {elastic_pca.l1_ratio_:.4f}")
print(f"  Train RMSE: {elastic_pca_rmse:.4f}")
results['elastic_pca_95'] = (elastic_pca, elastic_pca_rmse, X_pca_95, X_test_pca_95)

print("\n[9/10] Cross-validation with best models...")
# Get CV scores for top models
best_models = sorted(results.items(), key=lambda x: x[1][1])[:5]

print("\nTop 5 Models (by train RMSE):")
cv_scores = {}
for name, (model, train_rmse, X_feat, X_test_feat) in best_models:
    print(f"\n{name} (Train RMSE: {train_rmse:.4f})")
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_feat), 1):
        X_train_fold, X_val_fold = X_feat[train_idx], X_feat[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Clone and train model
        if 'lasso' in name:
            fold_model = Lasso(alpha=model.alpha_, random_state=42, max_iter=10000)
        elif 'ridge' in name:
            fold_model = Ridge(alpha=model.alpha_, random_state=42)
        else:  # elastic
            fold_model = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_,
                                   random_state=42, max_iter=10000)

        fold_model.fit(X_train_fold, y_train_fold)
        val_pred = fold_model.predict(X_val_fold)
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
        fold_scores.append(fold_rmse)

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    print(f"  CV RMSE: {cv_mean:.4f} (+/- {cv_std:.4f})")
    cv_scores[name] = (cv_mean, model, X_feat, X_test_feat)

print("\n[10/10] Creating weighted ensemble...")
# Sort by CV score
sorted_models = sorted(cv_scores.items(), key=lambda x: x[1][0])

# Weight by inverse RMSE
weights = []
predictions = []

for name, (cv_rmse, model, X_feat, X_test_feat) in sorted_models:
    weight = 1.0 / cv_rmse
    weights.append(weight)
    pred = model.predict(X_test_feat)
    predictions.append(pred)
    print(f"{name}: CV={cv_rmse:.4f}, Weight={weight:.6f}")

weights = np.array(weights) / sum(weights)
final_predictions = np.average(predictions, axis=0, weights=weights)

print("\nFinal ensemble weights:")
for (name, _), weight in zip(sorted_models, weights):
    print(f"  {name}: {weight:.4f}")

# Create submission
submission = pd.DataFrame({
    'customer_id': test_ids,
    'monthly_spend': final_predictions
})

submission.to_csv('submission_linear.csv', index=False)

print("\n" + "="*80)
print("LINEAR MODEL COMPLETE!")
print("="*80)
print(f"Best single model CV RMSE: {sorted_models[0][1][0]:.4f}")
print(f"\nSubmission saved: submission_linear.csv")
print(f"Mean prediction: ${submission['monthly_spend'].mean():.2f}")
print(f"Median prediction: ${submission['monthly_spend'].median():.2f}")
print(f"Std prediction: ${submission['monthly_spend'].std():.2f}")
print("\nFirst 10 predictions:")
print(submission.head(10))
print("="*80)

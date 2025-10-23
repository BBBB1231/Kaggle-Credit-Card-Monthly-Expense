"""
IMPROVED LINEAR REGRESSION - Focus on Generalization
Goal: Close the gap between validation and leaderboard (reduce overfitting)

Strategies:
1. Regularization (Ridge/Lasso) over pure LinearRegression
2. Careful feature selection based on CV, not just validation
3. Ensemble for stability
4. 10-fold CV for robust evaluation
5. Focus on features that generalize
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED LINEAR MODEL - Focus on Generalization")
print("Current: 253.087 on LB | Target: < 250 RMSE")
print("="*80)

# Load data
print("\n[1/8] Loading data...")
analysis_data = pd.read_csv('analysis_data.csv')
scoring_data = pd.read_csv('scoring_data.csv')

def create_conservative_features(df):
    """
    Conservative feature engineering - focus on features that generalize
    Based on user's winning approach but more selective
    """
    df_eng = df.copy()

    # Bulletproof cleaning
    eps = 1e-8

    df_eng['annual_income'].fillna(50000, inplace=True)
    df_eng['credit_score'].fillna(650, inplace=True)
    df_eng['credit_limit'].fillna(5000, inplace=True)
    df_eng['avg_transaction_value'].fillna(75, inplace=True)
    df_eng['reward_points_balance'].fillna(0, inplace=True)
    df_eng['num_transactions'].fillna(10, inplace=True)
    df_eng['online_shopping_freq'].fillna(5, inplace=True)
    df_eng['travel_frequency'].fillna(1, inplace=True)
    df_eng['utility_payment_count'].fillna(2, inplace=True)
    df_eng['education_level'].fillna('bachelors', inplace=True)

    df_eng['annual_income'] = df_eng['annual_income'].clip(lower=0)
    df_eng['credit_limit'] = df_eng['credit_limit'].clip(lower=0)
    df_eng['num_transactions'] = df_eng['num_transactions'].clip(lower=0)
    df_eng['avg_transaction_value'] = df_eng['avg_transaction_value'].clip(lower=0)

    # Categorical encoding
    df_eng['gender_male'] = (df_eng['gender'] == 'male').astype(int)
    df_eng['married'] = (df_eng['marital_status'] == 'married').astype(int)

    education_map = {'high school': 1, 'bachelors': 2, 'graduate': 3}
    df_eng['education_encoded'] = df_eng['education_level'].map(education_map).fillna(2)

    df_eng['employed'] = (df_eng['employment_status'] == 'employed').astype(int)
    df_eng['self_employed'] = (df_eng['employment_status'] == 'self-employed').astype(int)

    card_map = {'standard': 1, 'gold': 2, 'platinum': 3}
    df_eng['card_type_encoded'] = df_eng['card_type'].map(card_map).fillna(1)

    df_eng['region_south'] = (df_eng['region'] == 'south').astype(int)
    df_eng['region_northeast'] = (df_eng['region'] == 'northeast').astype(int)
    df_eng['region_west'] = (df_eng['region'] == 'west').astype(int)

    # CORE ENGINEERED FEATURES (proven to work)
    df_eng['credit_utilization'] = (df_eng['avg_transaction_value'] * df_eng['num_transactions']) / (df_eng['credit_limit'] + eps)
    df_eng['income_to_limit_ratio'] = df_eng['annual_income'] / (df_eng['credit_limit'] + eps)
    df_eng['spending_capacity'] = (df_eng['annual_income'] * df_eng['credit_limit']) / 1e9

    df_eng['transaction_intensity'] = df_eng['num_transactions'] * df_eng['avg_transaction_value']
    df_eng['transaction_frequency'] = df_eng['num_transactions'] / (df_eng['tenure'] + eps)

    df_eng['digital_engagement'] = df_eng['online_shopping_freq'] / (df_eng['num_transactions'] + eps)
    df_eng['reward_efficiency'] = df_eng['reward_points_balance'] / (df_eng['annual_income'] + eps)
    df_eng['travel_spending_ratio'] = df_eng['travel_frequency'] / (df_eng['num_transactions'] + eps)

    df_eng['age_income_interaction'] = df_eng['age'] * df_eng['annual_income'] / 1e6
    df_eng['family_size'] = df_eng['num_children'] + df_eng['married'] + 1

    df_eng['credit_experience'] = df_eng['tenure'] * df_eng['credit_score'] / 1000
    df_eng['high_value_user'] = (df_eng['card_type_encoded'] >= 2).astype(int)

    df_eng['wealth_indicator'] = (df_eng['annual_income'] * df_eng['owns_home'] * df_eng['credit_score']) / 1e8

    df_eng['log_income'] = np.log1p(df_eng['annual_income'])
    df_eng['log_credit_limit'] = np.log1p(df_eng['credit_limit'])
    df_eng['log_rewards'] = np.log1p(df_eng['reward_points_balance'])
    df_eng['spending_power'] = df_eng['log_income'] * df_eng['log_credit_limit']

    # Additional stable features
    df_eng['monthly_income'] = df_eng['annual_income'] / 12
    df_eng['rewards_per_transaction'] = df_eng['reward_points_balance'] / (df_eng['num_transactions'] + eps)
    df_eng['credit_per_card'] = df_eng['credit_limit'] / (df_eng['num_credit_cards'] + eps)

    df_eng['total_activity'] = (df_eng['num_transactions'] + df_eng['online_shopping_freq'] +
                                df_eng['travel_frequency'] + df_eng['utility_payment_count'])

    # Final safety
    for col in df_eng.select_dtypes(include=[np.number]).columns:
        df_eng[col] = df_eng[col].replace([np.inf, -np.inf], np.nan)
        if df_eng[col].isnull().any():
            df_eng[col].fillna(df_eng[col].median(), inplace=True)

    return df_eng

print("\n[2/8] Feature engineering...")
analysis_engineered = create_conservative_features(analysis_data)
scoring_engineered = create_conservative_features(scoring_data)

# Select features
feature_columns = [
    # Core
    'annual_income', 'credit_limit', 'credit_score', 'num_transactions',
    'avg_transaction_value', 'reward_points_balance', 'tenure',
    'online_shopping_freq', 'travel_frequency', 'utility_payment_count',
    'age', 'num_children', 'num_credit_cards',

    # Categorical
    'gender_male', 'married', 'education_encoded', 'employed', 'self_employed',
    'card_type_encoded', 'region_south', 'region_northeast', 'region_west',
    'owns_home', 'has_auto_loan',

    # Engineered
    'credit_utilization', 'income_to_limit_ratio', 'spending_capacity',
    'transaction_intensity', 'transaction_frequency', 'digital_engagement',
    'reward_efficiency', 'travel_spending_ratio', 'age_income_interaction',
    'family_size', 'credit_experience', 'high_value_user', 'wealth_indicator',
    'log_income', 'log_credit_limit', 'log_rewards', 'spending_power',
    'monthly_income', 'rewards_per_transaction', 'credit_per_card', 'total_activity'
]

X = analysis_engineered[feature_columns].copy()
y = analysis_engineered['monthly_spend'].copy()
X_scoring = scoring_engineered[feature_columns].copy()

X.fillna(X.median(), inplace=True)
X_scoring.fillna(X_scoring.median(), inplace=True)

print(f"Features: {X.shape[1]}")

print("\n[3/8] Feature selection with 10-fold CV...")
# Use SelectKBest to remove noisy features
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Test different numbers of features
best_cv_rmse = float('inf')
best_k = None

for k in [30, 35, 40, 45]:
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    ridge_temp = Ridge(alpha=10.0)
    cv_scores = cross_val_score(ridge_temp, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    print(f"  k={k}: CV RMSE = {cv_rmse:.4f}")

    if cv_rmse < best_cv_rmse:
        best_cv_rmse = cv_rmse
        best_k = k

print(f"Best k: {best_k} with CV RMSE: {best_cv_rmse:.4f}")

# Apply best feature selection
selector = SelectKBest(f_regression, k=best_k)
X_selected = selector.fit_transform(X, y)
X_scoring_selected = selector.transform(X_scoring)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = [feature_columns[i] for i in selected_indices]
print(f"\nSelected features: {selected_features[:10]}... ({len(selected_features)} total)")

print("\n[4/8] Hyperparameter tuning with 10-fold CV...")

# Ridge with CV
print("Ridge...")
alphas = np.logspace(-2, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=kf)
ridge_cv.fit(X_selected, y)
ridge_cv_scores = cross_val_score(ridge_cv, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
ridge_rmse = np.sqrt(-ridge_cv_scores.mean())
ridge_std = np.sqrt(ridge_cv_scores.std())
print(f"  Best alpha: {ridge_cv.alpha_:.4f}")
print(f"  CV RMSE: {ridge_rmse:.4f} ± {ridge_std:.4f}")

# Lasso with CV
print("Lasso...")
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=kf, max_iter=10000, random_state=42)
lasso_cv.fit(X_selected, y)
lasso_cv_scores = cross_val_score(lasso_cv, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
lasso_rmse = np.sqrt(-lasso_cv_scores.mean())
lasso_std = np.sqrt(lasso_cv_scores.std())
print(f"  Best alpha: {lasso_cv.alpha_:.4f}")
print(f"  CV RMSE: {lasso_rmse:.4f} ± {lasso_std:.4f}")

# ElasticNet with CV
print("ElasticNet...")
elastic_cv = ElasticNetCV(alphas=np.logspace(-4, 1, 50), l1_ratio=[.1, .5, .7, .9, .95, .99],
                          cv=kf, max_iter=10000, random_state=42)
elastic_cv.fit(X_selected, y)
elastic_cv_scores = cross_val_score(elastic_cv, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
elastic_rmse = np.sqrt(-elastic_cv_scores.mean())
elastic_std = np.sqrt(elastic_cv_scores.std())
print(f"  Best alpha: {elastic_cv.alpha_:.4f}, l1_ratio: {elastic_cv.l1_ratio_:.4f}")
print(f"  CV RMSE: {elastic_rmse:.4f} ± {elastic_std:.4f}")

# Scaled Ridge
print("Scaled Ridge...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_scoring_scaled = scaler.transform(X_scoring_selected)

scaled_ridge_cv = RidgeCV(alphas=alphas, cv=kf)
scaled_ridge_cv.fit(X_scaled, y)
scaled_cv_scores = cross_val_score(scaled_ridge_cv, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scaled_rmse = np.sqrt(-scaled_cv_scores.mean())
scaled_std = np.sqrt(scaled_cv_scores.std())
print(f"  Best alpha: {scaled_ridge_cv.alpha_:.4f}")
print(f"  CV RMSE: {scaled_rmse:.4f} ± {scaled_std:.4f}")

print("\n[5/8] Model ranking by CV performance...")
models = [
    ('Ridge', ridge_cv, ridge_rmse, ridge_std, X_selected, X_scoring_selected),
    ('Lasso', lasso_cv, lasso_rmse, lasso_std, X_selected, X_scoring_selected),
    ('ElasticNet', elastic_cv, elastic_rmse, elastic_std, X_selected, X_scoring_selected),
    ('ScaledRidge', (scaler, scaled_ridge_cv), scaled_rmse, scaled_std, X_scaled, X_scoring_scaled)
]

models.sort(key=lambda x: x[2])

print("\nModels ranked by CV RMSE:")
for i, (name, model, rmse, std, _, _) in enumerate(models, 1):
    print(f"{i}. {name:15s} - CV RMSE: {rmse:.4f} ± {std:.4f}")

print("\n[6/8] Creating ensemble of top models...")
# Ensemble top 3 models
top_n = 3
ensemble_preds = []
ensemble_weights = []

for name, model, rmse, std, X_feat, X_scoring_feat in models[:top_n]:
    if name == 'ScaledRidge':
        scaler, ridge_model = model
        pred = ridge_model.predict(X_scoring_feat)
    else:
        pred = model.predict(X_scoring_feat)

    weight = 1.0 / rmse  # Weight by inverse RMSE
    ensemble_preds.append(pred)
    ensemble_weights.append(weight)
    print(f"  {name}: weight = {weight:.6f}")

# Normalize weights
ensemble_weights = np.array(ensemble_weights) / sum(ensemble_weights)
print("\nNormalized weights:")
for (name, _, _, _, _, _), weight in zip(models[:top_n], ensemble_weights):
    print(f"  {name}: {weight:.4f}")

final_predictions = np.average(ensemble_preds, axis=0, weights=ensemble_weights)

print("\n[7/8] Generating submission...")
submission = pd.DataFrame({
    'customer_id': scoring_engineered['customer_id'],
    'monthly_spend': final_predictions
})

submission.to_csv('submission_improved.csv', index=False)

print("\n[8/8] Final validation...")
# Estimate ensemble CV score
print("Ensemble CV estimate (weighted average):")
ensemble_cv_rmse = np.average([models[i][2] for i in range(top_n)], weights=ensemble_weights)
ensemble_cv_std = np.average([models[i][3] for i in range(top_n)], weights=ensemble_weights)
print(f"  CV RMSE: {ensemble_cv_rmse:.4f} ± {ensemble_cv_std:.4f}")

print("\n" + "="*80)
print("IMPROVED MODEL COMPLETE!")
print("="*80)
print(f"Best Single Model: {models[0][0]}")
print(f"Best CV RMSE: {models[0][2]:.4f} ± {models[0][3]:.4f}")
print(f"Ensemble CV RMSE: {ensemble_cv_rmse:.4f} ± {ensemble_cv_std:.4f}")
print(f"\nSubmission: submission_improved.csv")
print(f"Mean: ${final_predictions.mean():.2f}")
print(f"Median: ${np.median(final_predictions):.2f}")
print(f"Std: ${final_predictions.std():.2f}")
print("\nFirst 10:")
print(submission.head(10))
print("\nExpected leaderboard improvement: 253.087 → ~{:.2f}".format(ensemble_cv_rmse + 2.5))
print("(Adding ~2.5 buffer for val/LB gap observed in previous submission)")
print("="*80)

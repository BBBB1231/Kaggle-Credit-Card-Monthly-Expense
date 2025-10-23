"""
INNOVATIVE FEATURE ENGINEERING APPROACH
Goal: Break through 253.087 with creative new features
Strategy: Explore feature space we haven't tried yet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INNOVATIVE FEATURE ENGINEERING")
print("Current Best: 253.087 | Goal: < 250 RMSE")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
analysis_data = pd.read_csv('analysis_data.csv')
scoring_data = pd.read_csv('scoring_data.csv')

def create_innovative_features(df):
    """
    Brand new features we haven't tried yet
    """
    df_eng = df.copy()
    eps = 1e-8

    # Safe cleaning (proven approach)
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

    # Basic encoding
    df_eng['gender_male'] = (df_eng['gender'] == 'male').astype(int)
    df_eng['married'] = (df_eng['marital_status'] == 'married').astype(int)
    education_map = {'high school': 1, 'bachelors': 2, 'graduate': 3}
    df_eng['education_encoded'] = df_eng['education_level'].map(education_map).fillna(2)
    df_eng['employed'] = (df_eng['employment_status'] == 'employed').astype(int)
    df_eng['self_employed'] = (df_eng['employment_status'] == 'self-employed').astype(int)
    card_map = {'standard': 1, 'gold': 2, 'platinum': 3}
    df_eng['card_type_encoded'] = df_eng['card_type'].map(card_map).fillna(1)

    # ========== INNOVATIVE FEATURES START HERE ==========

    print("  Creating innovative features...")

    # 1. BINNED/DISCRETIZED FEATURES (capture non-linearity better than polynomials)
    df_eng['income_bin'] = pd.qcut(df_eng['annual_income'], q=10, labels=False, duplicates='drop')
    df_eng['credit_bin'] = pd.qcut(df_eng['credit_limit'], q=10, labels=False, duplicates='drop')
    df_eng['age_bin'] = pd.cut(df_eng['age'], bins=[0, 25, 35, 45, 55, 100], labels=False)
    df_eng['trans_bin'] = pd.qcut(df_eng['num_transactions'], q=5, labels=False, duplicates='drop')

    # Bin interactions
    df_eng['income_credit_bin'] = df_eng['income_bin'] * 10 + df_eng['credit_bin']
    df_eng['age_income_bin'] = df_eng['age_bin'] * 10 + df_eng['income_bin']

    # 2. RATIO FEATURES (new combinations we haven't tried)
    df_eng['reward_to_limit'] = df_eng['reward_points_balance'] / (df_eng['credit_limit'] + eps)
    df_eng['reward_to_tenure'] = df_eng['reward_points_balance'] / (df_eng['tenure'] + eps)
    df_eng['trans_to_tenure'] = df_eng['num_transactions'] / (df_eng['tenure'] + eps)
    df_eng['online_to_travel'] = df_eng['online_shopping_freq'] / (df_eng['travel_frequency'] + eps)
    df_eng['utility_to_travel'] = df_eng['utility_payment_count'] / (df_eng['travel_frequency'] + eps)
    df_eng['cards_to_tenure'] = df_eng['num_credit_cards'] / (df_eng['tenure'] + eps)

    # 3. SPENDING BEHAVIOR PATTERNS
    df_eng['avg_spend_per_transaction'] = df_eng['avg_transaction_value']
    df_eng['estimated_monthly_volume'] = df_eng['num_transactions'] * df_eng['avg_transaction_value']
    df_eng['spend_to_income_ratio'] = df_eng['estimated_monthly_volume'] / (df_eng['annual_income'] / 12 + eps)
    df_eng['spend_to_limit_ratio'] = df_eng['estimated_monthly_volume'] / (df_eng['credit_limit'] + eps)

    # 4. LIFESTYLE INDICATORS
    df_eng['high_earner'] = (df_eng['annual_income'] > df_eng['annual_income'].quantile(0.75)).astype(int)
    df_eng['frequent_traveler'] = (df_eng['travel_frequency'] > 3).astype(int)
    df_eng['digital_native'] = ((df_eng['online_shopping_freq'] > 10) & (df_eng['age'] < 40)).astype(int)
    df_eng['premium_customer'] = ((df_eng['card_type_encoded'] >= 2) & (df_eng['credit_score'] > 700)).astype(int)

    # 5. FINANCIAL HEALTH SCORE (composite)
    df_eng['financial_health'] = (
        (df_eng['credit_score'] / 850 * 0.4) +
        (df_eng['annual_income'] / df_eng['annual_income'].max() * 0.3) +
        (df_eng['tenure'] / df_eng['tenure'].max() * 0.15) +
        (df_eng['owns_home'] * 0.15)
    )

    # 6. ENGAGEMENT SCORE
    df_eng['engagement_score'] = (
        df_eng['num_transactions'] +
        df_eng['online_shopping_freq'] * 1.5 +
        df_eng['travel_frequency'] * 2 +
        df_eng['utility_payment_count']
    )

    # 7. LEVERAGE/RISK FEATURES
    df_eng['leverage_ratio'] = df_eng['credit_limit'] / (df_eng['annual_income'] + eps)
    df_eng['income_security'] = df_eng['annual_income'] * df_eng['employed']
    df_eng['wealth_proxy'] = df_eng['annual_income'] * df_eng['owns_home'] * (df_eng['age'] / 100)

    # 8. REWARD OPTIMIZATION
    df_eng['reward_rate'] = df_eng['reward_points_balance'] / (df_eng['estimated_monthly_volume'] + eps)
    df_eng['reward_velocity'] = df_eng['reward_points_balance'] / (df_eng['tenure'] * 12 + eps)

    # 9. AGE-BASED FEATURES
    df_eng['age_squared'] = df_eng['age'] ** 2
    df_eng['age_cubed'] = df_eng['age'] ** 3 / 10000
    df_eng['young_professional'] = ((df_eng['age'] < 35) & (df_eng['employed'] == 1)).astype(int)
    df_eng['established'] = ((df_eng['age'] > 45) & (df_eng['owns_home'] == 1)).astype(int)

    # 10. TRANSACTION COMPLEXITY
    df_eng['payment_diversity'] = (
        (df_eng['online_shopping_freq'] > 0).astype(int) +
        (df_eng['travel_frequency'] > 0).astype(int) +
        (df_eng['utility_payment_count'] > 0).astype(int)
    )

    # 11. CREDIT CARD SOPHISTICATION
    df_eng['card_portfolio_value'] = df_eng['num_credit_cards'] * df_eng['card_type_encoded']
    df_eng['limit_per_card'] = df_eng['credit_limit'] / (df_eng['num_credit_cards'] + eps)
    df_eng['cards_per_year'] = df_eng['num_credit_cards'] / (df_eng['tenure'] + eps)

    # 12. LOG-SCALE INTERACTIONS (NEW!)
    df_eng['log_income'] = np.log1p(df_eng['annual_income'])
    df_eng['log_limit'] = np.log1p(df_eng['credit_limit'])
    df_eng['log_rewards'] = np.log1p(df_eng['reward_points_balance'])
    df_eng['log_trans'] = np.log1p(df_eng['num_transactions'])

    df_eng['log_income_limit'] = df_eng['log_income'] * df_eng['log_limit']
    df_eng['log_income_rewards'] = df_eng['log_income'] * df_eng['log_rewards']
    df_eng['log_limit_trans'] = df_eng['log_limit'] * df_eng['log_trans']

    # 13. SQRT TRANSFORMATIONS (alternative to log)
    df_eng['sqrt_income'] = np.sqrt(df_eng['annual_income'])
    df_eng['sqrt_limit'] = np.sqrt(df_eng['credit_limit'])
    df_eng['sqrt_rewards'] = np.sqrt(df_eng['reward_points_balance'])

    # 14. PERCENTILE RANKS (relative positioning)
    df_eng['income_percentile'] = df_eng['annual_income'].rank(pct=True)
    df_eng['limit_percentile'] = df_eng['credit_limit'].rank(pct=True)
    df_eng['reward_percentile'] = df_eng['reward_points_balance'].rank(pct=True)

    # 15. CROSS-CATEGORY INTERACTIONS
    df_eng['premium_young'] = df_eng['card_type_encoded'] * (df_eng['age'] < 40).astype(int)
    df_eng['married_homeowner'] = df_eng['married'] * df_eng['owns_home']
    df_eng['educated_employed'] = df_eng['education_encoded'] * df_eng['employed']

    # Final safety
    for col in df_eng.select_dtypes(include=[np.number]).columns:
        df_eng[col] = df_eng[col].replace([np.inf, -np.inf], np.nan)
        if df_eng[col].isnull().any():
            df_eng[col].fillna(df_eng[col].median(), inplace=True)

    return df_eng

print("\n[2/6] Applying innovative feature engineering...")
analysis_engineered = create_innovative_features(analysis_data)
scoring_engineered = create_innovative_features(scoring_data)

# Get all numeric features
all_features = [col for col in analysis_engineered.select_dtypes(include=[np.number]).columns
                if col not in ['customer_id', 'monthly_spend']]

print(f"Total features created: {len(all_features)}")

X = analysis_engineered[all_features].copy()
y = analysis_engineered['monthly_spend'].copy()
X_scoring = scoring_engineered[all_features].copy()

print(f"Feature matrix: {X.shape}")

print("\n[3/6] Testing feature importance...")
# Quick test to see which features matter
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Test with Ridge to get feature importance
ridge_test = Ridge(alpha=10.0)
ridge_test.fit(X, y)

# Get feature importance (absolute coefficients)
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': np.abs(ridge_test.coef_)
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20))

# Select top features
top_k = 50
top_features = feature_importance.head(top_k)['feature'].tolist()
print(f"\nUsing top {top_k} features")

X_selected = X[top_features]
X_scoring_selected = X_scoring[top_features]

print("\n[4/6] Training models with new features...")

# Ridge with CV
from sklearn.linear_model import RidgeCV
alphas = np.logspace(-2, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=kf)
ridge_cv.fit(X_selected, y)
ridge_scores = cross_val_score(ridge_cv, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
ridge_rmse = np.sqrt(-ridge_scores.mean())
ridge_std = np.sqrt(ridge_scores.std())
print(f"Ridge: {ridge_rmse:.4f} ± {ridge_std:.4f} (alpha={ridge_cv.alpha_:.4f})")

# Lasso with CV
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=kf, max_iter=10000, random_state=42)
lasso_cv.fit(X_selected, y)
lasso_scores = cross_val_score(lasso_cv, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
lasso_rmse = np.sqrt(-lasso_scores.mean())
lasso_std = np.sqrt(lasso_scores.std())
print(f"Lasso: {lasso_rmse:.4f} ± {lasso_std:.4f} (alpha={lasso_cv.alpha_:.4f})")

# ElasticNet
from sklearn.linear_model import ElasticNetCV
elastic_cv = ElasticNetCV(alphas=np.logspace(-4, 1, 30), l1_ratio=[.5, .7, .9, .95],
                          cv=kf, max_iter=10000, random_state=42)
elastic_cv.fit(X_selected, y)
elastic_scores = cross_val_score(elastic_cv, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
elastic_rmse = np.sqrt(-elastic_scores.mean())
elastic_std = np.sqrt(elastic_scores.std())
print(f"ElasticNet: {elastic_rmse:.4f} ± {elastic_std:.4f} (alpha={elastic_cv.alpha_:.4f})")

print("\n[5/6] Creating ensemble...")
models = [
    ('Ridge', ridge_cv, ridge_rmse),
    ('Lasso', lasso_cv, lasso_rmse),
    ('ElasticNet', elastic_cv, elastic_rmse)
]
models.sort(key=lambda x: x[2])

print("\nModel ranking:")
for i, (name, model, rmse) in enumerate(models, 1):
    print(f"{i}. {name}: {rmse:.4f}")

# Ensemble top 2
ensemble_preds = []
ensemble_weights = []
for name, model, rmse in models[:2]:
    pred = model.predict(X_scoring_selected)
    weight = 1.0 / rmse
    ensemble_preds.append(pred)
    ensemble_weights.append(weight)
    print(f"  {name}: weight={weight:.6f}")

ensemble_weights = np.array(ensemble_weights) / sum(ensemble_weights)
final_predictions = np.average(ensemble_preds, axis=0, weights=ensemble_weights)

print("\n[6/6] Generating submission...")
submission = pd.DataFrame({
    'customer_id': scoring_engineered['customer_id'],
    'monthly_spend': final_predictions
})

submission.to_csv('submission_innovative.csv', index=False)

print("\n" + "="*80)
print("INNOVATIVE FEATURES MODEL COMPLETE!")
print("="*80)
print(f"Best Model: {models[0][0]}")
print(f"Best CV RMSE: {models[0][2]:.4f}")
print(f"Ensemble CV: ~{np.average([m[2] for m in models[:2]], weights=ensemble_weights):.4f}")
print(f"\nSubmission: submission_innovative.csv")
print(f"Mean: ${final_predictions.mean():.2f}")
print(f"Median: ${np.median(final_predictions):.2f}")
print(f"\nPrevious best: 253.087")
print(f"Expected improvement: YES if CV < 250.5")
print("="*80)

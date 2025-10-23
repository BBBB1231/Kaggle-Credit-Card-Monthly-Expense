"""
ENHANCED MODEL based on winning 253 RMSE approach
Combining proven techniques with additional optimizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED CHAMPIONSHIP MODEL - Based on 253 RMSE Winner")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
analysis_data = pd.read_csv('analysis_data.csv')
scoring_data = pd.read_csv('scoring_data.csv')

print(f"Analysis data: {analysis_data.shape}")
print(f"Scoring data: {scoring_data.shape}")

def create_elite_features(df):
    """
    Enhanced feature engineering with bulletproof data cleaning
    Based on winning 253 RMSE approach + additional features
    """
    df_eng = df.copy()

    # ========== BULLETPROOF DATA CLEANING ==========
    print("  Cleaning data with safe defaults...")

    # Handle missing values with domain knowledge
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

    # Ensure no negative values
    df_eng['annual_income'] = df_eng['annual_income'].clip(lower=0)
    df_eng['credit_limit'] = df_eng['credit_limit'].clip(lower=0)
    df_eng['num_transactions'] = df_eng['num_transactions'].clip(lower=0)
    df_eng['avg_transaction_value'] = df_eng['avg_transaction_value'].clip(lower=0)

    # ========== CATEGORICAL ENCODING ==========
    print("  Encoding categorical variables...")

    df_eng['gender_male'] = (df_eng['gender'] == 'male').astype(int)
    df_eng['married'] = (df_eng['marital_status'] == 'married').astype(int)

    education_map = {'high school': 1, 'bachelors': 2, 'graduate': 3}
    df_eng['education_encoded'] = df_eng['education_level'].map(education_map).fillna(2)

    df_eng['employed'] = (df_eng['employment_status'] == 'employed').astype(int)
    df_eng['self_employed'] = (df_eng['employment_status'] == 'self-employed').astype(int)
    df_eng['unemployed'] = (df_eng['employment_status'] == 'unemployed').astype(int)

    card_map = {'standard': 1, 'gold': 2, 'platinum': 3}
    df_eng['card_type_encoded'] = df_eng['card_type'].map(card_map).fillna(1)

    df_eng['region_south'] = (df_eng['region'] == 'south').astype(int)
    df_eng['region_northeast'] = (df_eng['region'] == 'northeast').astype(int)
    df_eng['region_west'] = (df_eng['region'] == 'west').astype(int)
    df_eng['region_midwest'] = (df_eng['region'] == 'midwest').astype(int)

    # ========== CORE FEATURES (From winning approach) ==========
    print("  Creating core engineered features...")

    eps = 1e-8  # Safe epsilon for division

    # Financial ratios
    df_eng['credit_utilization'] = (df_eng['avg_transaction_value'] * df_eng['num_transactions']) / (df_eng['credit_limit'] + eps)
    df_eng['income_to_limit_ratio'] = df_eng['annual_income'] / (df_eng['credit_limit'] + eps)
    df_eng['spending_capacity'] = (df_eng['annual_income'] * df_eng['credit_limit']) / 1e9

    # Transaction patterns
    df_eng['transaction_intensity'] = df_eng['num_transactions'] * df_eng['avg_transaction_value']
    df_eng['transaction_frequency'] = df_eng['num_transactions'] / (df_eng['tenure'] + eps)
    df_eng['avg_monthly_transactions'] = df_eng['num_transactions'] / 12

    # Behavioral features
    df_eng['digital_engagement'] = df_eng['online_shopping_freq'] / (df_eng['num_transactions'] + eps)
    df_eng['reward_efficiency'] = df_eng['reward_points_balance'] / (df_eng['annual_income'] + eps)
    df_eng['travel_spending_ratio'] = df_eng['travel_frequency'] / (df_eng['num_transactions'] + eps)

    # Demographics
    df_eng['age_income_interaction'] = df_eng['age'] * df_eng['annual_income'] / 1e6
    df_eng['family_size'] = df_eng['num_children'] + df_eng['married'] + 1
    df_eng['dependency_ratio'] = df_eng['num_children'] / (df_eng['annual_income'] / 10000 + eps)

    # Credit behavior
    df_eng['credit_experience'] = df_eng['tenure'] * df_eng['credit_score'] / 1000
    df_eng['multi_card_user'] = (df_eng['num_credit_cards'] > 1).astype(int)
    df_eng['high_value_user'] = (df_eng['card_type_encoded'] >= 2).astype(int)

    # Advanced interactions
    df_eng['wealth_indicator'] = (df_eng['annual_income'] * df_eng['owns_home'] * df_eng['credit_score']) / 1e8
    df_eng['financial_stability'] = (df_eng['employed'] * df_eng['owns_home'] * df_eng['tenure']) / 10

    # Log transformations
    df_eng['log_income'] = np.log1p(df_eng['annual_income'])
    df_eng['log_credit_limit'] = np.log1p(df_eng['credit_limit'])
    df_eng['spending_power'] = df_eng['log_income'] * df_eng['log_credit_limit']

    # ========== ADDITIONAL ELITE FEATURES ==========
    print("  Adding additional elite features...")

    # More reward features (highly correlated!)
    df_eng['rewards_per_transaction'] = df_eng['reward_points_balance'] / (df_eng['num_transactions'] + eps)
    df_eng['rewards_per_credit'] = df_eng['reward_points_balance'] / (df_eng['credit_limit'] + eps)
    df_eng['log_rewards'] = np.log1p(df_eng['reward_points_balance'])

    # Income features
    df_eng['monthly_income'] = df_eng['annual_income'] / 12
    df_eng['income_per_card'] = df_eng['annual_income'] / (df_eng['num_credit_cards'] + eps)
    df_eng['credit_per_card'] = df_eng['credit_limit'] / (df_eng['num_credit_cards'] + eps)

    # Polynomial features for top predictors
    df_eng['credit_limit_sq'] = df_eng['credit_limit'] ** 2 / 1e8
    df_eng['income_sq'] = df_eng['annual_income'] ** 2 / 1e9
    df_eng['num_trans_sq'] = df_eng['num_transactions'] ** 2
    df_eng['rewards_sq'] = df_eng['reward_points_balance'] ** 2 / 1e6

    # Activity combinations
    df_eng['total_activity'] = (df_eng['num_transactions'] + df_eng['online_shopping_freq'] +
                               df_eng['travel_frequency'] + df_eng['utility_payment_count'])
    df_eng['utility_ratio'] = df_eng['utility_payment_count'] / (df_eng['num_transactions'] + eps)

    # Age interactions
    df_eng['age_credit_score'] = df_eng['age'] * df_eng['credit_score'] / 100
    df_eng['age_sq'] = df_eng['age'] ** 2

    # ========== FINAL SAFETY CHECK ==========
    print("  Applying final safety checks...")

    for col in df_eng.select_dtypes(include=[np.number]).columns:
        df_eng[col] = df_eng[col].replace([np.inf, -np.inf], np.nan)
        if df_eng[col].isnull().any():
            df_eng[col].fillna(df_eng[col].median(), inplace=True)

    return df_eng

# Apply feature engineering
print("\n[2/7] Applying elite feature engineering...")
analysis_engineered = create_elite_features(analysis_data)
scoring_engineered = create_elite_features(scoring_data)

# Select features
feature_columns = [
    # Core financial
    'annual_income', 'credit_limit', 'credit_score', 'num_transactions',
    'avg_transaction_value', 'reward_points_balance', 'tenure',

    # Behavioral
    'online_shopping_freq', 'travel_frequency', 'utility_payment_count',

    # Demographics
    'age', 'num_children', 'num_credit_cards',

    # Categorical
    'gender_male', 'married', 'education_encoded', 'employed', 'self_employed',
    'card_type_encoded', 'region_south', 'region_northeast', 'region_west',
    'owns_home', 'has_auto_loan',

    # Engineered (winning features)
    'credit_utilization', 'income_to_limit_ratio', 'spending_capacity',
    'transaction_intensity', 'transaction_frequency', 'digital_engagement',
    'reward_efficiency', 'travel_spending_ratio', 'age_income_interaction',
    'family_size', 'dependency_ratio', 'credit_experience', 'multi_card_user',
    'high_value_user', 'wealth_indicator', 'financial_stability', 'spending_power',
    'log_income', 'log_credit_limit',

    # Additional elite features
    'rewards_per_transaction', 'rewards_per_credit', 'log_rewards',
    'monthly_income', 'income_per_card', 'credit_per_card',
    'credit_limit_sq', 'income_sq', 'num_trans_sq', 'rewards_sq',
    'total_activity', 'utility_ratio', 'age_credit_score', 'age_sq'
]

print(f"Selected {len(feature_columns)} elite features")

# Prepare data
X = analysis_engineered[feature_columns].copy()
y = analysis_engineered['monthly_spend'].copy()
X_scoring = scoring_engineered[feature_columns].copy()

print(f"\n[3/7] Data validation...")
print(f"Training shape: {X.shape}")
print(f"Target mean: ${y.mean():.2f}, std: ${y.std():.2f}")

# Final safety check
X.fillna(X.median(), inplace=True)
X_scoring.fillna(X_scoring.median(), inplace=True)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n[4/7] Training comprehensive model arsenal...")
print("="*80)

models = {}
results = {}

# 1. Linear Regression
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
lr_r2 = r2_score(y_val, lr_pred)
models['LinearRegression'] = lr
results['LinearRegression'] = {'RMSE': lr_rmse, 'R2': lr_r2}
print(f"  RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")

# 2. Ridge (multiple alphas)
print("Training Ridge...")
best_ridge_rmse = float('inf')
best_ridge_alpha = None
for alpha in [0.1, 1.0, 5.0, 10.0, 50.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_val)
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
    if ridge_rmse < best_ridge_rmse:
        best_ridge_rmse = ridge_rmse
        best_ridge_alpha = alpha
        best_ridge_model = ridge

ridge_r2 = r2_score(y_val, best_ridge_model.predict(X_val))
models['Ridge'] = best_ridge_model
results['Ridge'] = {'RMSE': best_ridge_rmse, 'R2': ridge_r2}
print(f"  Best alpha: {best_ridge_alpha}, RMSE: {best_ridge_rmse:.4f}, R2: {ridge_r2:.4f}")

# 3. Lasso
print("Training Lasso...")
best_lasso_rmse = float('inf')
best_lasso_alpha = None
for alpha in [0.01, 0.1, 1.0, 5.0]:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_val)
    lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_pred))
    if lasso_rmse < best_lasso_rmse:
        best_lasso_rmse = lasso_rmse
        best_lasso_alpha = alpha
        best_lasso_model = lasso

lasso_r2 = r2_score(y_val, best_lasso_model.predict(X_val))
models['Lasso'] = best_lasso_model
results['Lasso'] = {'RMSE': best_lasso_rmse, 'R2': lasso_r2}
print(f"  Best alpha: {best_lasso_alpha}, RMSE: {best_lasso_rmse:.4f}, R2: {lasso_r2:.4f}")

# 4. ElasticNet
print("Training ElasticNet...")
best_en_rmse = float('inf')
for alpha in [0.1, 1.0, 5.0]:
    for l1_ratio in [0.3, 0.5, 0.7]:
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        en.fit(X_train, y_train)
        en_pred = en.predict(X_val)
        en_rmse = np.sqrt(mean_squared_error(y_val, en_pred))
        if en_rmse < best_en_rmse:
            best_en_rmse = en_rmse
            best_en_model = en

en_r2 = r2_score(y_val, best_en_model.predict(X_val))
models['ElasticNet'] = best_en_model
results['ElasticNet'] = {'RMSE': best_en_rmse, 'R2': en_r2}
print(f"  RMSE: {best_en_rmse:.4f}, R2: {en_r2:.4f}")

# 5. Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeRegressor(max_depth=15, min_samples_split=50, min_samples_leaf=20, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_val)
dt_rmse = np.sqrt(mean_squared_error(y_val, dt_pred))
dt_r2 = r2_score(y_val, dt_pred)
models['DecisionTree'] = dt
results['DecisionTree'] = {'RMSE': dt_rmse, 'R2': dt_r2}
print(f"  RMSE: {dt_rmse:.4f}, R2: {dt_r2:.4f}")

# 6. Scaled Ridge
print("Training Scaled Ridge...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

best_scaled_rmse = float('inf')
for alpha in [1.0, 5.0, 10.0, 20.0]:
    scaled_ridge = Ridge(alpha=alpha)
    scaled_ridge.fit(X_train_scaled, y_train)
    scaled_pred = scaled_ridge.predict(X_val_scaled)
    scaled_rmse = np.sqrt(mean_squared_error(y_val, scaled_pred))
    if scaled_rmse < best_scaled_rmse:
        best_scaled_rmse = scaled_rmse
        best_scaled_alpha = alpha
        best_scaled_model = scaled_ridge
        best_scaler = scaler

scaled_r2 = r2_score(y_val, best_scaled_model.predict(X_val_scaled))
models['ScaledRidge'] = (best_scaler, best_scaled_model)
results['ScaledRidge'] = {'RMSE': best_scaled_rmse, 'R2': scaled_r2}
print(f"  Best alpha: {best_scaled_alpha}, RMSE: {best_scaled_rmse:.4f}, R2: {scaled_r2:.4f}")

# 7. Polynomial Ridge (KEY!)
print("Training Polynomial Ridge...")
poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_val_poly = poly_features.transform(X_val)

print(f"  Polynomial features: {X_train_poly.shape[1]}")

# Select top features if too many
if X_train_poly.shape[1] > 1000:
    selector = SelectKBest(score_func=f_regression, k=500)
    X_train_poly = selector.fit_transform(X_train_poly, y_train)
    X_val_poly = selector.transform(X_val_poly)
    print(f"  Selected top 500 polynomial features")
else:
    selector = None

poly_ridge = Ridge(alpha=10.0)
poly_ridge.fit(X_train_poly, y_train)
poly_pred = poly_ridge.predict(X_val_poly)
poly_rmse = np.sqrt(mean_squared_error(y_val, poly_pred))
poly_r2 = r2_score(y_val, poly_pred)
models['PolynomialRidge'] = (poly_features, selector, poly_ridge)
results['PolynomialRidge'] = {'RMSE': poly_rmse, 'R2': poly_r2}
print(f"  RMSE: {poly_rmse:.4f}, R2: {poly_r2:.4f}")

print("\n[5/7] Model Ranking...")
print("="*80)
sorted_models = sorted(results.items(), key=lambda x: x[1]['RMSE'])

print("\nAll Models Ranked by RMSE:")
for i, (name, metrics) in enumerate(sorted_models, 1):
    rmse = metrics['RMSE']
    r2 = metrics['R2']
    print(f"{i}. {name:20s} - RMSE: {rmse:7.4f}, R2: {r2:6.4f}")

best_model_name = sorted_models[0][0]
best_rmse = sorted_models[0][1]['RMSE']

print(f"\n🏆 CHAMPION: {best_model_name} with RMSE: {best_rmse:.4f}")

print("\n[6/7] Generating predictions...")

# Generate predictions with best model
if best_model_name == 'PolynomialRidge':
    poly_feat, selector, ridge_model = models[best_model_name]
    X_scoring_poly = poly_feat.transform(X_scoring)
    if selector is not None:
        X_scoring_poly = selector.transform(X_scoring_poly)
    final_predictions = ridge_model.predict(X_scoring_poly)
elif best_model_name == 'ScaledRidge':
    scaler, ridge_model = models[best_model_name]
    X_scoring_scaled = scaler.transform(X_scoring)
    final_predictions = ridge_model.predict(X_scoring_scaled)
else:
    final_predictions = models[best_model_name].predict(X_scoring)

# Create submission
submission = pd.DataFrame({
    'customer_id': scoring_engineered['customer_id'],
    'monthly_spend': final_predictions
})

submission.to_csv('submission_enhanced.csv', index=False)

print(f"\n[7/7] Cross-validation verification...")
if best_model_name == 'ScaledRidge':
    pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=best_scaled_alpha))])
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
elif best_model_name == 'DecisionTree':
    cv_scores = cross_val_score(DecisionTreeRegressor(max_depth=15, min_samples_split=50,
                                min_samples_leaf=20, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
else:
    cv_scores = cross_val_score(models[best_model_name], X, y, cv=5, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores.mean())
cv_std = np.sqrt(cv_scores.std())

print("\n" + "="*80)
print("ENHANCED MODEL COMPLETE!")
print("="*80)
print(f"Best Model: {best_model_name}")
print(f"Validation RMSE: {best_rmse:.4f}")
print(f"Cross-validation RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
print(f"\nSubmission: submission_enhanced.csv")
print(f"Mean prediction: ${final_predictions.mean():.2f}")
print(f"Median prediction: ${np.median(final_predictions):.2f}")
print(f"Std prediction: ${final_predictions.std():.2f}")
print("\nFirst 10 predictions:")
print(submission.head(10))
print("="*80)

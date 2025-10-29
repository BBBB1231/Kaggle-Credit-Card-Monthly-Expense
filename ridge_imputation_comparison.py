import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
import time

warnings.filterwarnings('ignore')

print("="*80)
print("RIDGE IMPUTATION COMPARISON: MEAN vs MEDIAN vs KNN")
print("="*80)

# --- 1. Load Data ---
print("\n1. Loading data...")
analysis_df = pd.read_csv('analysis_data.csv')
scoring_df = pd.read_csv('scoring_data.csv')

print(f"   Analysis data: {analysis_df.shape}")
print(f"   Scoring data: {scoring_df.shape}")

# --- 2. Feature Engineering ---
print("\n2. Engineering features...")

def feature_engineer(df):
    """All 12 engineered features"""
    # Ridge V1 Features (5)
    df['total_transaction_value'] = df['num_transactions'] * df['avg_transaction_value']
    df['income_to_limit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1e-6)
    df['limit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)
    df['age_x_tenure'] = df['age'] * df['tenure']
    df['points_per_transaction'] = df['reward_points_balance'] / (df['num_transactions'] + 1e-6)

    # Advanced Features (7)
    df['income_per_child'] = df['annual_income'] / (df['num_children'] + 1)
    df['score_x_income'] = df['credit_score'] * df['annual_income'] / 1000000
    df['limit_utilization'] = df['credit_limit'] / (df['annual_income'] + 1)
    df['online_spend_intensity'] = df['avg_transaction_value'] * df['online_shopping_freq']
    df['credit_density'] = df['num_credit_cards'] / (df['credit_score'] + 1)
    df['transaction_intensity'] = df['num_transactions'] / (df['tenure'] + 1)
    df['age_x_income'] = df['age'] * df['annual_income'] / 100000

    return df

analysis_df = feature_engineer(analysis_df)
scoring_df = feature_engineer(scoring_df)

print(f"   ✓ Total engineered features: 12")

# --- 3. Encode Categorical Features ---
print("\n3. Encoding categorical features...")

categorical_features = ['gender', 'marital_status', 'education_level', 'region',
                        'employment_status', 'owns_home', 'has_auto_loan', 'card_type']

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([analysis_df[col].astype(str), scoring_df[col].astype(str)])
    le.fit(combined)
    analysis_df[col] = le.transform(analysis_df[col].astype(str))
    scoring_df[col] = le.transform(scoring_df[col].astype(str))
    label_encoders[col] = le

print(f"   ✓ Encoded {len(categorical_features)} categorical features")

# --- 4. Define Feature Lists ---
all_features = [
    'age', 'num_children', 'annual_income', 'credit_score', 'num_credit_cards',
    'credit_limit', 'tenure', 'num_transactions', 'avg_transaction_value',
    'online_shopping_freq', 'reward_points_balance', 'travel_frequency',
    'utility_payment_count',
    'gender', 'marital_status', 'education_level', 'region',
    'employment_status', 'owns_home', 'has_auto_loan', 'card_type',
    'total_transaction_value', 'income_to_limit_ratio',
    'limit_per_card', 'age_x_tenure', 'points_per_transaction',
    'income_per_child', 'score_x_income', 'limit_utilization',
    'online_spend_intensity', 'credit_density', 'transaction_intensity', 'age_x_income'
]

# --- 5. Prepare Data ---
X = analysis_df[all_features]
y = analysis_df['monthly_spend']
X_scoring = scoring_df[all_features]

print(f"\n   Training samples: {len(X)}")
print(f"   Total features: {len(all_features)}")

# --- 6. Test All Three Imputation Methods ---
print("\n" + "="*80)
print("TESTING IMPUTATION METHODS")
print("="*80)

methods = {
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'knn': KNNImputer(n_neighbors=5)
}

results = []

for method_name, imputer in methods.items():
    print(f"\n{'='*80}")
    print(f"METHOD: {method_name.upper()} IMPUTATION")
    print(f"{'='*80}")

    start_time = time.time()

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', RobustScaler()),
        ('regressor', Ridge(random_state=42))
    ])

    # Hyperparameter tuning
    param_grid = {
        'regressor__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100, 200]
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    print(f"   Training with 10-fold CV...")
    grid_search.fit(X, y)

    best_alpha = grid_search.best_params_['regressor__alpha']
    best_cv_rmse = -grid_search.best_score_

    # Train final model
    final_model = grid_search.best_estimator_
    final_model.fit(X, y)

    # Training RMSE
    y_train_pred = final_model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))

    # Predictions
    predictions = final_model.predict(X_scoring)

    # Create submission
    submission_df = pd.DataFrame({
        'customer_id': scoring_df['customer_id'],
        'monthly_spend': predictions
    })
    submission_df['monthly_spend'] = submission_df['monthly_spend'].apply(lambda x: max(0, x))

    submission_file = f'submission_ridge_impute_{method_name}.csv'
    submission_df.to_csv(submission_file, index=False)

    elapsed_time = time.time() - start_time

    # Store results
    results.append({
        'Method': method_name.upper(),
        'Best Alpha': best_alpha,
        'CV RMSE': best_cv_rmse,
        'Training RMSE': train_rmse,
        'Time (s)': elapsed_time,
        'Submission File': submission_file
    })

    print(f"   ✓ Best Alpha: {best_alpha}")
    print(f"   ✓ CV RMSE: {best_cv_rmse:.4f}")
    print(f"   ✓ Training RMSE: {train_rmse:.4f}")
    print(f"   ✓ Time: {elapsed_time:.1f}s")
    print(f"   ✓ Saved: {submission_file}")

# --- 7. Final Comparison ---
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame(results)
print("\n" + comparison_df.to_string(index=False))

# Find best method
best_idx = comparison_df['CV RMSE'].idxmin()
best_method = comparison_df.loc[best_idx, 'Method']
best_rmse = comparison_df.loc[best_idx, 'CV RMSE']
best_file = comparison_df.loc[best_idx, 'Submission File']

print("\n" + "="*80)
print("WINNER")
print("="*80)
print(f"✓ Best Imputation Method: {best_method}")
print(f"✓ Best CV RMSE: {best_rmse:.4f}")
print(f"✓ Best Submission File: {best_file}")
print("="*80)

# Calculate improvements
worst_rmse = comparison_df['CV RMSE'].max()
improvement = worst_rmse - best_rmse
improvement_pct = (improvement / worst_rmse) * 100

print(f"\nImprovement vs worst method: {improvement:.4f} ({improvement_pct:.2f}%)")

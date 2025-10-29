import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("RIDGE V3 - ENHANCED FEATURES + KNN IMPUTATION")
print("="*80)

# --- 1. Load Data ---
print("\n1. Loading data...")
analysis_df = pd.read_csv('analysis_data.csv')
scoring_df = pd.read_csv('scoring_data.csv')

print(f"   Analysis data: {analysis_df.shape}")
print(f"   Scoring data: {scoring_df.shape}")

# --- 2. ENHANCED Feature Engineering ---
print("\n2. Engineering features...")

def feature_engineer_v3(df):
    """
    Enhanced feature engineering - all 12 features
    """
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

analysis_df = feature_engineer_v3(analysis_df)
scoring_df = feature_engineer_v3(scoring_df)

print(f"   ✓ Total engineered features: 12")

# --- 3. Encode Categorical Features BEFORE Pipeline ---
# KNN Imputer requires all numeric data, so we encode categoricals first
print("\n3. Encoding categorical features...")

categorical_features = ['gender', 'marital_status', 'education_level', 'region',
                        'employment_status', 'owns_home', 'has_auto_loan', 'card_type']

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    # Fit on combined data
    combined = pd.concat([analysis_df[col].astype(str), scoring_df[col].astype(str)])
    le.fit(combined)
    analysis_df[col] = le.transform(analysis_df[col].astype(str))
    scoring_df[col] = le.transform(scoring_df[col].astype(str))
    label_encoders[col] = le

print(f"   ✓ Encoded {len(categorical_features)} categorical features")

# --- 4. Define Feature Lists ---
all_features = [
    # Original features
    'age', 'num_children', 'annual_income', 'credit_score', 'num_credit_cards',
    'credit_limit', 'tenure', 'num_transactions', 'avg_transaction_value',
    'online_shopping_freq', 'reward_points_balance', 'travel_frequency',
    'utility_payment_count',
    # Categorical (now encoded as numbers)
    'gender', 'marital_status', 'education_level', 'region',
    'employment_status', 'owns_home', 'has_auto_loan', 'card_type',
    # Engineered Features
    'total_transaction_value', 'income_to_limit_ratio',
    'limit_per_card', 'age_x_tenure', 'points_per_transaction',
    'income_per_child', 'score_x_income', 'limit_utilization',
    'online_spend_intensity', 'credit_density', 'transaction_intensity', 'age_x_income'
]

print(f"   ✓ Total features: {len(all_features)}")

# --- 5. Build Pipeline with KNN Imputation ---
print("\n4. Building pipeline with KNN Imputation...")

pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('regressor', Ridge(random_state=42))
])

# --- 6. Prepare Data ---
X = analysis_df[all_features]
y = analysis_df['monthly_spend']

print(f"   Training samples: {len(X)}")
print(f"   Features: {X.shape[1]}")

# --- 7. Hyperparameter Tuning ---
print("\n5. Hyperparameter tuning with GridSearchCV...")

param_grid = {
    'regressor__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100, 200, 500]
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"   Testing {len(param_grid['regressor__alpha'])} alpha values with 10-fold CV...")

grid_search.fit(X, y)

# --- 8. Results ---
print("\n" + "="*80)
print("RESULTS")
print("="*80)

best_alpha = grid_search.best_params_['regressor__alpha']
best_rmse = -grid_search.best_score_

print(f"\n✓ Best alpha: {best_alpha}")
print(f"✓ Best CV RMSE: {best_rmse:.4f}")

# Show top 5 alpha values
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['mean_rmse'] = -results_df['mean_test_score']
results_df = results_df[['param_regressor__alpha', 'mean_rmse']].sort_values('mean_rmse')

print("\nTop 5 Alpha Values:")
print(results_df.head(5).to_string(index=False))

# --- 9. Train Final Model ---
print("\n6. Training final model on all data...")

final_model = grid_search.best_estimator_
final_model.fit(X, y)

y_train_pred = final_model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
print(f"   Training RMSE: {train_rmse:.4f}")

# --- 10. Generate Predictions ---
print("\n7. Generating predictions...")

X_scoring = scoring_df[all_features]
predictions = final_model.predict(X_scoring)

# --- 11. Create Submission ---
print("\n8. Creating submission file...")

submission_df = pd.DataFrame({
    'customer_id': scoring_df['customer_id'],
    'monthly_spend': predictions
})

submission_df['monthly_spend'] = submission_df['monthly_spend'].apply(lambda x: max(0, x))

print(f"   Prediction mean: {submission_df['monthly_spend'].mean():.2f}")
print(f"   Prediction std: {submission_df['monthly_spend'].std():.2f}")

submission_file_name = 'submission_ridge_v3_knn.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\n✓ Submission file '{submission_file_name}' created successfully!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Method: Ridge Regression + KNN Imputation (k=5)")
print(f"Features: {len(all_features)} (12 engineered)")
print(f"Best Alpha: {best_alpha}")
print(f"CV RMSE: {best_rmse:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Submission File: {submission_file_name}")
print("="*80)

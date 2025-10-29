import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("RIDGE V2 - ENHANCED WITH ADVANCED FEATURE ENGINEERING")
print("="*80)

# --- 1. Load Data ---
print("\n1. Loading data...")
try:
    # Local environment
    analysis_df = pd.read_csv('analysis_data.csv')
    scoring_df = pd.read_csv('scoring_data.csv')
except FileNotFoundError:
    # Colab environment
    from google.colab import drive
    drive.mount('/content/drive')
    analysis_df = pd.read_csv('/content/drive/MyDrive/APAN5200/PAC/analysis_data.csv')
    scoring_df = pd.read_csv('/content/drive/MyDrive/APAN5200/PAC/scoring_data.csv')

print(f"   Analysis data: {analysis_df.shape}")
print(f"   Scoring data: {scoring_df.shape}")

# --- 2. ENHANCED Feature Engineering ---
print("\n2. Engineering features...")

def feature_engineer_v2(df):
    """
    Enhanced feature engineering combining Ridge V1 features + additional advanced features
    """
    # === Ridge V1 Features (5) ===

    # Most predictive: total transaction value
    df['total_transaction_value'] = df['num_transactions'] * df['avg_transaction_value']

    # Financial ratios
    df['income_to_limit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1e-6)
    df['limit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)

    # Interactions
    df['age_x_tenure'] = df['age'] * df['tenure']
    df['points_per_transaction'] = df['reward_points_balance'] / (df['num_transactions'] + 1e-6)

    # === NEW Advanced Features (7) ===

    # 1. Income adjusted for household size
    df['income_per_child'] = df['annual_income'] / (df['num_children'] + 1)

    # 2. Creditworthiness × Income (this was 3rd most important in LASSO!)
    df['score_x_income'] = df['credit_score'] * df['annual_income'] / 1000000  # Scale down

    # 3. Credit utilization (inverse ratio)
    df['limit_utilization'] = df['credit_limit'] / (df['annual_income'] + 1)

    # 4. Online shopping spending intensity
    df['online_spend_intensity'] = df['avg_transaction_value'] * df['online_shopping_freq']

    # 5. Credit card density
    df['credit_density'] = df['num_credit_cards'] / (df['credit_score'] + 1)

    # 6. Transaction activity per year
    df['transaction_intensity'] = df['num_transactions'] / (df['tenure'] + 1)

    # 7. Age × Income interaction
    df['age_x_income'] = df['age'] * df['annual_income'] / 100000  # Scale down

    return df

analysis_df = feature_engineer_v2(analysis_df)
scoring_df = feature_engineer_v2(scoring_df)

print(f"   ✓ Total engineered features: 12 (5 from V1 + 7 new)")

# --- 3. Define Column Lists ---
print("\n3. Defining feature columns...")

numerical_features = [
    # Original features
    'age', 'num_children', 'annual_income', 'credit_score', 'num_credit_cards',
    'credit_limit', 'tenure', 'num_transactions', 'avg_transaction_value',
    'online_shopping_freq', 'reward_points_balance', 'travel_frequency',
    'utility_payment_count',
    # Ridge V1 Features (5)
    'total_transaction_value', 'income_to_limit_ratio',
    'limit_per_card', 'age_x_tenure', 'points_per_transaction',
    # NEW Advanced Features (7)
    'income_per_child', 'score_x_income', 'limit_utilization',
    'online_spend_intensity', 'credit_density', 'transaction_intensity', 'age_x_income'
]

categorical_features = [
    'gender', 'marital_status', 'education_level', 'region',
    'employment_status', 'owns_home', 'has_auto_loan', 'card_type'
]

print(f"   ✓ Numerical features: {len(numerical_features)}")
print(f"   ✓ Categorical features: {len(categorical_features)}")

# --- 4. Build Preprocessing Pipelines ---
print("\n4. Building preprocessing pipeline...")

# Numerical Pipeline: Impute with median, then use RobustScaler
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Can also try KNN here
    ('scaler', RobustScaler())  # Better for outliers than StandardScaler
])

# Categorical Pipeline: Impute with 'missing', then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# --- 5. Combine Transformers ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- 6. Create the Full Model Pipeline ---
print("\n5. Creating Ridge regression pipeline...")

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(random_state=42))
])

# --- 7. Define X and y ---
X = analysis_df.drop(['monthly_spend', 'customer_id'], axis=1, errors='ignore')
y = analysis_df['monthly_spend']

print(f"   Training samples: {len(X)}")
print(f"   Target mean: {y.mean():.2f}")
print(f"   Target std: {y.std():.2f}")

# --- 8. Enhanced Hyperparameter Tuning ---
print("\n6. Hyperparameter tuning with GridSearchCV...")

# Expanded alpha search range
param_grid = {
    'regressor__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10, 25, 50, 100, 200, 500]
}

# 10-fold Cross-validation (same as V1)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"   Testing {len(param_grid['regressor__alpha'])} alpha values with 10-fold CV...")
print(f"   Total fits: {len(param_grid['regressor__alpha']) * 10}")

grid_search.fit(X, y)

# --- 9. Results Analysis ---
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

# --- 10. Train Final Model ---
print("\n7. Training final model on all data...")

final_model = grid_search.best_estimator_
final_model.fit(X, y)

# Get training RMSE
y_train_pred = final_model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
print(f"   Training RMSE: {train_rmse:.4f}")

# --- 11. Generate Predictions ---
print("\n8. Generating predictions on scoring data...")

X_scoring = scoring_df.drop('customer_id', axis=1, errors='ignore')
X_scoring = X_scoring[X.columns]

predictions = final_model.predict(X_scoring)

# --- 12. Create Submission ---
print("\n9. Creating submission file...")

submission_df = pd.DataFrame({
    'customer_id': scoring_df['customer_id'],
    'monthly_spend': predictions
})

# Ensure no negative predictions
submission_df['monthly_spend'] = submission_df['monthly_spend'].apply(lambda x: max(0, x))

print(f"   Prediction mean: {submission_df['monthly_spend'].mean():.2f}")
print(f"   Prediction std: {submission_df['monthly_spend'].std():.2f}")
print(f"   Prediction min: {submission_df['monthly_spend'].min():.2f}")
print(f"   Prediction max: {submission_df['monthly_spend'].max():.2f}")

submission_file_name = 'submission_ridge_v2_enhanced.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\n✓ Submission file '{submission_file_name}' created successfully!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Features: {len(numerical_features)} numerical + {len(categorical_features)} categorical")
print(f"Total Engineered Features: 12")
print(f"Best Alpha: {best_alpha}")
print(f"CV RMSE: {best_rmse:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Submission File: {submission_file_name}")
print("="*80)

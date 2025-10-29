import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    # Local environment
    analysis_df = pd.read_csv('analysis_data.csv')
    scoring_df = pd.read_csv('scoring_data.csv')
except FileNotFoundError:
    # Colab environment from your notebook
    from google.colab import drive
    drive.mount('/content/drive')
    analysis_df = pd.read_csv('/content/drive/MyDrive/APAN5200/PAC/analysis_data.csv')
    scoring_df = pd.read_csv('/content/drive/MyDrive/APAN5200/PAC/scoring_data.csv')

# --- 2. Smart Feature Engineering ---
# We do this *before* the pipeline
def feature_engineer(df):
    # This is likely your most predictive feature.
    # Total spend is almost certainly a function of total transaction value.
    df['total_transaction_value'] = df['num_transactions'] * df['avg_transaction_value']

    # Ratios are great for linear models
    df['income_to_limit_ratio'] = df['annual_income'] / (df['credit_limit'] + 1e-6)
    df['limit_per_card'] = df['credit_limit'] / (df['num_credit_cards'] + 1)

    # Interaction
    df['age_x_tenure'] = df['age'] * df['tenure']

    # Balance per transaction
    df['points_per_transaction'] = df['reward_points_balance'] / (df['num_transactions'] + 1e-6)

    return df

analysis_df = feature_engineer(analysis_df)
scoring_df = feature_engineer(scoring_df)

# --- 3. Define Column Lists ---
# We have new engineered features to add
numerical_features = [
    'age', 'num_children', 'annual_income', 'credit_score', 'num_credit_cards',
    'credit_limit', 'tenure', 'num_transactions', 'avg_transaction_value',
    'online_shopping_freq', 'reward_points_balance', 'travel_frequency',
    'utility_payment_count',
    # New Features
    'total_transaction_value', 'income_to_limit_ratio',
    'limit_per_card', 'age_x_tenure', 'points_per_transaction'
]

# All other non-numeric, non-ID, non-target features
categorical_features = [
    'gender', 'marital_status', 'education_level', 'region',
    'employment_status', 'owns_home', 'has_auto_loan', 'card_type'
]

# --- 4. Build Preprocessing Pipelines ---

# Numerical Pipeline: Impute with median, then use RobustScaler
# This is our NEW way to handle outliers.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Categorical Pipeline: Impute with 'missing' (as its own category), then One-Hot Encode
# handle_unknown='ignore' is crucial for data in the scoring set
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
    remainder='drop' # Drop any columns we didn't specify (like customer_id)
)

# --- 6. Create the Full Model Pipeline ---
# This pipeline does all preprocessing and then models
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(random_state=42)) # Using Ridge to handle multicollinearity
])

# --- 7. Define X and y ---
X = analysis_df.drop(['monthly_spend', 'customer_id'], axis=1, errors='ignore')
y = analysis_df['monthly_spend']

# --- 8. Tune the Model with GridSearchCV ---
# We will tune the 'alpha' parameter of Ridge
# This 'alpha' controls the strength of the regularization (our fix for multicollinearity)
param_grid = {
    'regressor__alpha': [0.1, 1.0, 10, 50, 100, 200]
}

# 10-fold Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Use scoring='neg_root_mean_squared_error' to optimize for RMSE
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1 # Use all cores
)

print("Starting model tuning...")
grid_search.fit(X, y)

print(f"Best parameters found: {grid_search.best_params_}")
# The score will be negative, so we multiply by -1
best_rmse = -grid_search.best_score_
print(f"Best cross-validation RMSE: {best_rmse:.4f}")

# This score should be *significantly* lower than 250.
# If this is below 220, you're ready. If it's close, you can try LASSO
# by replacing 'Ridge()' with 'Lasso()' and tuning its 'alpha'

# --- 9. Train Final Model and Generate Submission ---
print("\nTraining final model on all data...")
# grid_search.best_estimator_ is our final, tuned pipeline
final_model = grid_search.best_estimator_
final_model.fit(X, y)

print("Generating predictions on scoring data...")
# The pipeline will automatically apply all the same transformations
X_scoring = scoring_df.drop('customer_id', axis=1, errors='ignore')

# Align columns to be 100% sure
X_scoring = X_scoring[X.columns]

predictions = final_model.predict(X_scoring)

# --- 10. Create Submission File ---
submission_df = pd.DataFrame({
    'customer_id': scoring_df['customer_id'],
    'monthly_spend': predictions
})

# Ensure no negative predictions (spending can't be negative)
submission_df['monthly_spend'] = submission_df['monthly_spend'].apply(lambda x: max(0, x))

submission_file_name = 'submission_ridge_v1_baseline.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"Submission file '{submission_file_name}' created successfully!")

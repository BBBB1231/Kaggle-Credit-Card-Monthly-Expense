import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the analysis and scoring data"""
    print("Loading data...")
    analysis_data = pd.read_csv('analysis_data.csv')
    scoring_data = pd.read_csv('scoring_data.csv')
    print(f"Analysis data shape: {analysis_data.shape}")
    print(f"Scoring data shape: {scoring_data.shape}")
    return analysis_data, scoring_data

def check_missing_values(df, name="Dataset"):
    """Check for missing values in the dataset"""
    print(f"\n{name} Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    return missing_df

def preprocess_features(train_df, test_df, target_col='monthly_spend'):
    """
    Preprocess features: encode categorical variables and separate features from target
    """
    # Separate features and target
    X_train = train_df.drop([target_col, 'customer_id'], axis=1, errors='ignore')
    y_train = train_df[target_col] if target_col in train_df.columns else None

    # For test data, keep customer_id for submission
    test_customer_ids = test_df['customer_id'] if 'customer_id' in test_df.columns else None
    X_test = test_df.drop(['customer_id'], axis=1, errors='ignore')
    if target_col in X_test.columns:
        X_test = X_test.drop([target_col], axis=1)

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Encode categorical variables using Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to ensure all categories are known
        combined_col = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined_col.astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    return X_train, y_train, X_test, test_customer_ids, numerical_cols, categorical_cols

def apply_imputation(X_train, X_test, method='mean'):
    """
    Apply imputation to handle missing values

    Parameters:
    - method: 'mean', 'median', or 'knn'
    """
    print(f"\nApplying {method} imputation...")

    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError("Method must be 'mean', 'median', or 'knn'")

    # Fit on training data and transform both
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Convert back to DataFrame
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

    return X_train_imputed, X_test_imputed

def apply_feature_scaling(X_train, X_test):
    """
    Apply StandardScaler for feature transformation
    """
    print("\nApplying feature scaling (StandardScaler)...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, scaler

def train_lasso_model(X_train, y_train, alpha=1.0):
    """
    Train LASSO regression model
    """
    print(f"\nTraining LASSO model with alpha={alpha}...")
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)

    # Print number of non-zero coefficients
    non_zero_coefs = np.sum(lasso.coef_ != 0)
    print(f"Number of non-zero coefficients: {non_zero_coefs}/{len(lasso.coef_)}")

    return lasso

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate the model on training and validation sets
    """
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Validation predictions
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Validation R²: {val_r2:.4f}")

    return val_rmse, val_r2

def run_pipeline(imputation_method='mean', alpha=1.0):
    """
    Run the complete pipeline with specified imputation method
    """
    print(f"\n{'='*80}")
    print(f"Running pipeline with {imputation_method.upper()} imputation")
    print(f"{'='*80}")

    # Load data
    analysis_data, scoring_data = load_data()

    # Check missing values
    check_missing_values(analysis_data, "Analysis Data")
    check_missing_values(scoring_data, "Scoring Data")

    # Preprocess features
    X_train_full, y_train_full, X_test, test_customer_ids, num_cols, cat_cols = preprocess_features(
        analysis_data, scoring_data
    )

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"\nTrain set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")

    # Apply imputation
    X_train_imputed, X_val_imputed = apply_imputation(X_train, X_val, method=imputation_method)
    X_train_full_imputed, X_test_imputed = apply_imputation(X_train_full, X_test, method=imputation_method)

    # Apply feature scaling
    X_train_scaled, X_val_scaled, _ = apply_feature_scaling(X_train_imputed, X_val_imputed)
    X_train_full_scaled, X_test_scaled, scaler = apply_feature_scaling(X_train_full_imputed, X_test_imputed)

    # Train LASSO model on training set
    lasso_model = train_lasso_model(X_train_scaled, y_train, alpha=alpha)

    # Evaluate on validation set
    val_rmse, val_r2 = evaluate_model(lasso_model, X_train_scaled, y_train, X_val_scaled, y_val)

    # Retrain on full training data for final predictions
    print("\nRetraining on full training data for final predictions...")
    final_model = train_lasso_model(X_train_full_scaled, y_train_full, alpha=alpha)

    # Make predictions on test set
    y_test_pred = final_model.predict(X_test_scaled)

    # Create submission file
    submission = pd.DataFrame({
        'customer_id': test_customer_ids,
        'monthly_spend': y_test_pred
    })

    submission_filename = f'submission_{imputation_method}.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"\nSubmission file saved: {submission_filename}")

    return {
        'method': imputation_method,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'model': final_model,
        'submission_file': submission_filename
    }

def compare_imputation_methods(alpha=1.0):
    """
    Compare all three imputation methods and select the best one
    """
    methods = ['mean', 'median', 'knn']
    results = []

    for method in methods:
        result = run_pipeline(imputation_method=method, alpha=alpha)
        results.append(result)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Method': r['method'].upper(),
            'Validation RMSE': r['val_rmse'],
            'Validation R²': r['val_r2'],
            'Submission File': r['submission_file']
        }
        for r in results
    ])

    print(f"\n{'='*80}")
    print("COMPARISON OF IMPUTATION METHODS")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))

    # Find best method (lowest RMSE)
    best_idx = comparison_df['Validation RMSE'].idxmin()
    best_method = comparison_df.loc[best_idx, 'Method']
    best_rmse = comparison_df.loc[best_idx, 'Validation RMSE']

    print(f"\n{'='*80}")
    print(f"BEST METHOD: {best_method} with Validation RMSE: {best_rmse:.2f}")
    print(f"{'='*80}")

    return comparison_df, results[best_idx]

if __name__ == "__main__":
    # Compare all imputation methods with LASSO
    comparison_df, best_result = compare_imputation_methods(alpha=1.0)

    print(f"\nBest submission file: {best_result['submission_file']}")
    print("\nAll submission files have been generated. Use the best one for submission!")
